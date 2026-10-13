"""
LoFi: low-frequency RoPE channel selection and mass-corrected merging for
calibration-free KV cache compression.

Uses the sqrt(base) frequency boundary that separates position-invariant
low-frequency channels from position-dependent high-frequency channels
(see :mod:`lofi.frequency`) to select and merge token pairs in the KV cache.
Because the low-frequency channels carry position-invariant content, pair
similarity stays meaningful across passes and at token gaps greater than one,
and merge counts can be folded back into the decode attention logits so a
survivor keeps the attention mass of everything merged into it.

Developed in collaboration with researchers at NVIDIA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LoFiConfig
from .frequency import high_freq_indices, low_freq_indices

# Attention sinks are never compressed: the first tokens absorb a large share
# of attention mass regardless of content, so merging them is uniformly harmful.
N_SINK = 32

# Smallest recent window kept verbatim, even when recent_ratio would ask for less.
MIN_RECENT_WINDOW = 32


@dataclass
class CompressionStats:
    """Per-layer record of what one compression call actually did."""

    seq_len: int
    kept: int
    retention: float
    passes: int
    fallback_evicted: int


@dataclass
class CompressionResult:
    """Outcome of compressing one layer's cache.

    Attributes:
        keys: (1, H, kept, head_dim) compressed keys.
        values: (1, H, kept, head_dim) compressed values.
        merge_counts: (1, kept) float32 multiplicity per surviving slot, or
            None when no merging happened. Feeds the decode logit bias.
        stats: Retention record, or None when compression did not run.
    """

    keys: torch.Tensor
    values: torch.Tensor
    merge_counts: torch.Tensor | None
    stats: CompressionStats | None


@torch.no_grad()
def compress_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    rope_base: float,
    config: LoFiConfig,
    attn_weights: torch.Tensor | None = None,
) -> CompressionResult:
    """Compress a post-prefill KV cache to the configured retention target.

    The cache is split into a sink prefix, a compressible middle region, and a
    verbatim recent window. Only the middle region is touched. Candidate pairs
    are consecutive members of the *active* set rather than of the original
    sequence, so adjacency is recomputed after each pass; a single pass over
    non-overlapping adjacent pairs can remove at most half the region, so
    reaching budgets below 50% retention requires several passes. Passes
    alternate pair parity, and if pairing runs out before the target is met the
    lowest-importance survivors are evicted outright and counted in the stats.

    Args:
        key_states: (1, H, n, head_dim) post-RoPE keys, already expanded to the
            query head count.
        value_states: (1, H, n, head_dim) values, same layout.
        rope_base: The model's RoPE theta, which fixes the frequency boundary.
        config: Compression settings.
        attn_weights: (1, H, n, n) prefill attention probabilities, used as the
            token importance signal. When None, importance is uniform, which
            disables protection and makes merge direction arbitrary.

    Returns:
        A :class:`CompressionResult`. When the sequence already fits the budget
        the inputs are passed through unchanged.
    """
    merge_mode = config.merge_mode
    value_mode = config.value_mode
    key_rescale = config.key_rescale

    bsz, _n_heads, seq_len, head_dim = key_states.shape
    device = key_states.device

    budget = max(4, int(seq_len * config.target_retention))
    if seq_len <= budget:
        return CompressionResult(key_states, value_states, None, None)
    if bsz != 1:
        raise ValueError(f"LoFi compression supports batch size 1, got {bsz}")

    window = min(max(MIN_RECENT_WINDOW, int(seq_len * config.recent_ratio)),
                 seq_len - N_SINK - 1)
    target_middle = max(1, budget - N_SINK - window)
    start, end = N_SINK, seq_len - window
    region_len = end - start

    if region_len <= 1 or target_middle >= region_len:
        # Budget leaves no room to compress: keep sink + recent window only.
        keys = torch.cat([key_states[:, :, :N_SINK, :], key_states[:, :, -window:, :]], dim=2)
        values = torch.cat([value_states[:, :, :N_SINK, :], value_states[:, :, -window:, :]], dim=2)
        return CompressionResult(keys, values, None, None)

    low_idx = low_freq_indices(head_dim, rope_base, device)
    high_idx = high_freq_indices(head_dim, rope_base, device)
    sel_idx = low_idx if config.sim_mode == "low" else torch.arange(head_dim, device=device)

    keys_mid = key_states[0, :, start:end, :].clone()       # (H, region_len, head_dim)
    values_mid = value_states[0, :, start:end, :].clone()

    if attn_weights is not None:
        # Total attention mass received by each key column, averaged over heads.
        importance = attn_weights.sum(dim=2).mean(dim=1)[0, start:end].float().clone()
    else:
        importance = torch.zeros(region_len, device=device)

    counts = torch.ones(region_len, device=device, dtype=torch.float32)
    kept = torch.ones(region_len, dtype=torch.bool, device=device)
    protected = torch.zeros(region_len, dtype=torch.bool, device=device)
    if attn_weights is not None and config.protect_ratio > 0:
        n_protect = max(1, int(region_len * config.protect_ratio))
        protected[importance.topk(n_protect).indices] = True
    active = kept & (~protected)

    need = region_len - target_middle
    parity = 0
    passes = 0
    fallback_evicted = 0

    while need > 0 and passes < config.max_passes:
        act_pos = active.nonzero(as_tuple=False).squeeze(-1)   # ascending positions
        n_active = int(act_pos.numel())
        if n_active < 2:
            break
        left = act_pos[parity:n_active - 1:2]
        right = act_pos[parity + 1:n_active:2]
        n_pairs = min(left.numel(), right.numel())
        if n_pairs == 0:
            parity ^= 1
            passes += 1
            continue
        left = left[:n_pairs]
        right = right[:n_pairs]

        # Score on the current keys: under the content-preserving merge the
        # low-frequency block of a survivor is bit-identical to its original,
        # so similarities stay comparable across passes.
        left_n = F.normalize(keys_mid[:, left, :].index_select(-1, sel_idx).float(), dim=-1)
        right_n = F.normalize(keys_mid[:, right, :].index_select(-1, sel_idx).float(), dim=-1)
        sims = (left_n * right_n).sum(-1).mean(0)              # (n_pairs,)

        take = min(need, n_pairs)
        top_sims, order = sims.topk(take)
        pair_left = left[order]
        pair_right = right[order]
        score = top_sims.clamp(0.0, 1.0)

        # The more important side survives, so attention mass concentrates on
        # the token the model was already reading.
        right_wins = importance[pair_right] > importance[pair_left]
        tgt = torch.where(right_wins, pair_right, pair_left)
        src = torch.where(right_wins, pair_left, pair_right)

        if merge_mode != "evict":
            tgt_keys = keys_mid[:, tgt, :].clone()
            src_keys = keys_mid[:, src, :]
            if merge_mode == "cp":
                alpha_h = (0.2 + 0.3 * score).view(1, -1, 1).to(keys_mid.dtype)
                tgt_high = tgt_keys.index_select(-1, high_idx)
                src_high = src_keys.index_select(-1, high_idx)
                high_norm0 = tgt_high.float().norm(dim=-1, keepdim=True)
                mixed = (1 - alpha_h) * tgt_high + alpha_h * src_high
                if key_rescale == "h":
                    # Restoring only the high-frequency norm leaves the
                    # low-frequency content exactly untouched.
                    mixed_norm = mixed.float().norm(dim=-1, keepdim=True)
                    mixed = (mixed.float() * (high_norm0 / (mixed_norm + 1e-6))).to(keys_mid.dtype)
                tgt_keys[..., high_idx] = mixed
                if key_rescale == "full":
                    norm0 = keys_mid[:, tgt, :].float().norm(dim=-1, keepdim=True)
                    norm1 = tgt_keys.float().norm(dim=-1, keepdim=True)
                    tgt_keys = (tgt_keys.float() * (norm0 / (norm1 + 1e-6))).to(keys_mid.dtype)
            elif merge_mode == "uniform":
                tgt_keys = 0.5 * (tgt_keys + src_keys)
            keys_mid[:, tgt, :] = tgt_keys

            tgt_vals = values_mid[:, tgt, :]
            src_vals = values_mid[:, src, :]
            if value_mode == "adaptive":
                alpha_v = (0.15 + 0.15 * score).view(1, -1, 1).to(values_mid.dtype)
                values_mid[:, tgt, :] = (1 - alpha_v) * tgt_vals + alpha_v * src_vals
            else:
                if value_mode == "imp_mass":
                    w_tgt = counts[tgt] * importance[tgt].clamp_min(1e-6)
                    w_src = counts[src] * importance[src].clamp_min(1e-6)
                else:
                    # Count-weighted: on the first pass this is a 50:50 mean.
                    w_tgt = counts[tgt]
                    w_src = counts[src]
                w_tgt = w_tgt.view(1, -1, 1).to(values_mid.dtype)
                w_src = w_src.view(1, -1, 1).to(values_mid.dtype)
                values_mid[:, tgt, :] = (w_tgt * tgt_vals + w_src * src_vals) / (w_tgt + w_src)

            counts[tgt] = counts[tgt] + counts[src]
            if config.importance_accumulate:
                importance[tgt] = importance[tgt] + importance[src]
        # In "evict" mode the source is simply dropped and counts stay at 1,
        # which makes the log-count logit bias a no-op.

        kept[src] = False
        active[src] = False
        need -= take
        passes += 1
        parity ^= 1

    if need > 0:
        # Pairing exhausted before the budget was met. Evict the least
        # important survivors, preferring unprotected ones.
        pool = (kept & ~protected).nonzero(as_tuple=False).squeeze(-1)
        if pool.numel() < need:
            extra = (kept & protected).nonzero(as_tuple=False).squeeze(-1)
            pool = torch.cat([pool, extra])
        order = importance[pool].argsort()
        drop = pool[order[:need]]
        kept[drop] = False
        fallback_evicted = int(drop.numel())
        need = 0

    keep_idx = kept.nonzero(as_tuple=False).squeeze(-1)
    new_keys = torch.cat(
        [key_states[0, :, :N_SINK, :], keys_mid[:, keep_idx, :], key_states[0, :, end:, :]],
        dim=1,
    ).unsqueeze(0)
    new_values = torch.cat(
        [value_states[0, :, :N_SINK, :], values_mid[:, keep_idx, :], value_states[0, :, end:, :]],
        dim=1,
    ).unsqueeze(0)

    n_recent = new_keys.shape[2] - N_SINK - int(keep_idx.numel())
    merge_counts = torch.cat([
        torch.ones(N_SINK, device=device),
        counts[keep_idx],
        torch.ones(n_recent, device=device),
    ]).view(1, -1)

    stats = CompressionStats(
        seq_len=int(seq_len),
        kept=int(new_keys.shape[2]),
        retention=round(float(new_keys.shape[2]) / seq_len, 4),
        passes=passes,
        fallback_evicted=fallback_evicted,
    )
    if config.debug:
        print(f"[LoFi] {stats}", flush=True)
    return CompressionResult(new_keys, new_values, merge_counts, stats)


def mass_logit_bias(merge_counts: torch.Tensor, n_keys: int) -> torch.Tensor:
    """Additive attention-logit bias that restores merged tokens' mass.

    Adding log(c_j) before the softmax gives exp(q.k + log c_j) = c_j exp(q.k),
    so a slot standing for c_j original tokens attracts c_j times the weight.

    Args:
        merge_counts: (1, L) multiplicities from :func:`compress_kv_cache`.
        n_keys: Current key length, which grows past L as tokens are decoded.

    Returns:
        (1, 1, 1, n_keys) float bias. Tokens decoded after compression have
        count 1 and so receive zero bias.
    """
    counts = merge_counts
    if counts.shape[-1] < n_keys:
        pad = n_keys - counts.shape[-1]
        counts = torch.cat(
            [counts, torch.ones(counts.shape[0], pad, device=counts.device)], dim=-1
        )
    return counts.clamp_min(1.0).log().view(1, 1, 1, -1)

"""LoFi on a Hugging Face causal LM: prefill, compress the KV cache once, decode.

The three steps are separate functions so that each can be inspected:

    pre = prefill(model, input_ids)                 # cache + importance (Section 4.2)
    cache, stats = compress(model, pre, LoFiConfig())   # Eq. (4) + Algorithm 1 per layer and KV head
    tokens = greedy_decode(model, cache, pre.logits, pre.length, max_new_tokens=64)

:func:`generate` wraps them for the two settings of Section 4.1. In the
query-aware setting the whole prompt is prefilled and compressed. In the
query-agnostic setting only the context is prefilled and compressed, and the
query is then prefilled on the compressed cache.

Hugging Face models store keys after RoPE, one per KV head, which is what
LoFi scores. Only batch size 1 is supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .frequency import derotate, low_high_channels, recent_window
from .selection import CRITERIA, OPERATORS, select

try:  # transformers >= 4.36
    from transformers import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None


@dataclass
class LoFiConfig:
    """Settings of LoFi. The defaults are the ones of the paper (Section 4.3).

    Attributes:
        budget: share of the prefilled tokens kept in every layer and KV head,
            sinks and recent window included.
        n_sink: sink tokens ``n_s`` kept at the start.
        n_obs: observation positions ``n_o`` at the end of the prefill, used
            for the importance.
        protect: protected fraction ``p`` of the distant budget.
        operator: ``evict`` (LoFi-Sel) or ``merge`` (LoFi-M, Eq. 10).
        criterion: ``low`` (LoFi). ``full``, ``derotated`` and ``random`` are
            the controls of the selection-rule comparison.
        first_criterion: criterion of the first pass only, if different.
        use_importance: False gives the control without importance.
        max_passes: cap on the number of passes of the pair rule. When the cap
            is reached before the budget, the fallback of line 13 removes the
            remaining tokens. None runs the loop of line 4 without a cap.
        window: recent window in tokens. None applies Eq. (4).
        rope_base: RoPE base. None reads it from the model config; for models
            with rescaled RoPE this is the base before rescaling.
        seed: seed of the ``random`` control.
    """

    budget: float = 0.2
    n_sink: int = 32
    n_obs: int = 32
    protect: float = 0.1
    operator: str = "evict"
    criterion: str = "low"
    first_criterion: Optional[str] = None
    use_importance: bool = True
    max_passes: Optional[int] = None
    window: Optional[int] = None
    rope_base: Optional[float] = None
    seed: int = 0

    def __post_init__(self):
        if self.max_passes is not None and self.max_passes < 0:
            raise ValueError("max_passes must be None or a non-negative integer")
        if not 0.0 < self.budget <= 1.0:
            raise ValueError("budget must be in (0, 1]")
        if self.operator not in OPERATORS:
            raise ValueError(f"operator must be one of {OPERATORS}")
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")


@dataclass
class Layout:
    """Split of ``n`` prefilled tokens into sinks, distant region and recent window."""

    n: int
    budget: int          # B
    n_sink: int          # n_s
    window: int          # W
    n_keep: int          # B - n_s - W, tokens kept from D

    @property
    def n_distant(self) -> int:      # |D|
        return self.n - self.n_sink - self.window

    @property
    def n_remove(self) -> int:       # r
        return self.n_distant - self.n_keep


def layout(n: int, cfg: LoFiConfig, base: float) -> Layout:
    """Budget ``B = floor(budget * n)`` and the window of Eq. (4)."""
    budget = int(math.floor(cfg.budget * n + 1e-9))
    if budget <= cfg.n_sink + 1:
        raise ValueError(
            f"a budget of {budget} tokens leaves nothing to select after {cfg.n_sink} sink tokens "
            f"(n={n}); the selection is not defined for prompts this short"
        )
    if cfg.window is None:
        window = recent_window(base, budget, cfg.n_sink)
    else:
        window = min(int(cfg.window), budget - cfg.n_sink - 1)
    return Layout(n=n, budget=budget, n_sink=cfg.n_sink, window=window, n_keep=budget - cfg.n_sink - window)


@dataclass
class Prefill:
    """Output of :func:`prefill`."""

    cache: object
    logits: torch.Tensor                       # (1, vocab), last prefilled position
    length: int                                # n
    importance: Optional[List[torch.Tensor]]   # per layer, (H_kv, n)


@dataclass
class CompressionStats:
    """What :func:`compress` did. Tensors are stacked over layers: ``(layers, H_kv)``."""

    layout: Layout
    n_protected: int
    passes: torch.Tensor
    pair_removed: torch.Tensor
    fallback_removed: torch.Tensor
    kept: List[torch.Tensor] = field(default_factory=list)     # per layer, (H_kv, B) positions
    counts: List[torch.Tensor] = field(default_factory=list)   # per layer, (H_kv, B), merge only

    def summary(self) -> dict:
        lay = self.layout
        pair, fallback = int(self.pair_removed.sum()), int(self.fallback_removed.sum())
        return {
            "n": lay.n,
            "budget": lay.budget,
            "window": lay.window,
            "window_share_of_budget": lay.window / lay.budget,
            "distant": lay.n_distant,
            "kept_from_distant": lay.n_keep,
            "kept_share_of_distant": lay.n_keep / max(lay.n_distant, 1),
            "protected": self.n_protected,
            "mean_passes": float(self.passes.float().mean()),
            "max_passes": int(self.passes.max()),
            # share of the removals made by the pair rule rather than by the fallback of line 13
            "pair_share": pair / max(pair + fallback, 1),
            "fallback_removed": fallback,
        }


# --------------------------------------------------------------------------- model helpers


def rope_base(model) -> float:
    """RoPE base of the model (before any rescaling)."""
    config = model.config
    base = getattr(config, "rope_theta", None)
    if base is None:
        params = getattr(config, "rope_parameters", None) or {}
        base = params.get("rope_theta")
    if base is None:
        raise ValueError("the model config has no RoPE base; pass LoFiConfig(rope_base=...)")
    return float(base)


def rope_inv_freq(model) -> torch.Tensor:
    """Inverse frequencies the model applies (after rescaling, if any)."""
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor):
            return inv_freq.detach().double().cpu()
    raise ValueError("no rotary embedding with an `inv_freq` buffer found in the model")


def disable_sliding_window(model) -> None:
    """Turn off sliding-window attention (Mistral-7B-v0.1 ships with a 4,096-token window).

    The paper evaluates Mistral-7B and Qwen2-7B with full attention.
    """
    if getattr(model.config, "sliding_window", None) is not None:
        model.config.sliding_window = None
    if getattr(model.config, "use_sliding_window", False):
        model.config.use_sliding_window = False


def cache_layers(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """``[(keys, values)]`` per layer, each ``(1, H_kv, n, d)``, for any cache format."""
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    if hasattr(cache, "layers"):                     # transformers 5.x
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):                  # transformers 4.36 to 4.5x
        return list(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"unsupported cache type {type(cache).__name__}")


def cache_length(cache) -> int:
    return cache_layers(cache)[0][0].shape[-2]


def build_cache(layers: Sequence[Tuple[torch.Tensor, torch.Tensor]], like):
    """A cache of the same kind as ``like`` holding the given layers."""
    if isinstance(like, (tuple, list)):
        return tuple((k, v) for k, v in layers)
    cache = DynamicCache()
    for idx, (k, v) in enumerate(layers):
        cache.update(k, v, idx)
    return cache


def _legacy_rope_modules(model) -> list:
    """Rotary embeddings that slice their table to the cache length (Qwen2 in transformers 4.43).

    These modules return ``cos[:cache_length + q]`` and index it with the
    position ids, which fails once the cache is shorter than the positions.
    Their ``forward`` is replaced by one that returns a table long enough for
    the positions in use. The values of the table are unchanged.
    """
    modules = getattr(model, "_lofi_legacy_rope", None)
    if modules is not None:
        return modules
    modules = [m for m in model.modules() if hasattr(m, "_set_cos_sin_cache") and hasattr(m, "cos_cached")]
    for module in modules:
        module._lofi_min_len = 0

        def forward(x, seq_len=None, _m=module):
            seq_len = max(int(seq_len or 0), _m._lofi_min_len)
            if seq_len > _m.max_seq_len_cached:
                _m._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            return _m.cos_cached[:seq_len].to(dtype=x.dtype), _m.sin_cached[:seq_len].to(dtype=x.dtype)

        module.forward = forward
    model._lofi_legacy_rope = modules
    return modules


def _forward(model, input_ids, cache, position: int, output_attentions: bool = False):
    """One forward pass on top of ``cache`` for tokens at true positions ``position, position+1, ...``.

    After compression the cache is shorter than the number of tokens seen, so
    the positions must be passed explicitly.
    """
    q = input_ids.shape[1]
    device = input_ids.device
    for module in _legacy_rope_modules(model):
        module._lofi_min_len = position + q
    kwargs = dict(input_ids=input_ids, use_cache=True)
    if cache is not None:
        kwargs["past_key_values"] = cache
        kwargs["position_ids"] = torch.arange(position, position + q, device=device).unsqueeze(0)
        kwargs["attention_mask"] = torch.ones(1, cache_length(cache) + q, dtype=torch.long, device=device)
    elif DynamicCache is not None:
        kwargs["past_key_values"] = DynamicCache()
    if output_attentions:
        kwargs["output_attentions"] = True
    return model(**kwargs)


# --------------------------------------------------------------------------- prefill


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, n_obs: int = 32, with_importance: bool = True) -> Prefill:
    """Prefill ``input_ids`` and compute the importance of every cached token.

    ``imp(j) = sum_{i in O} A_ij`` over the last ``n_obs`` prefilled positions
    ``O``, with the attention ``A`` averaged over the query heads that share a
    KV head (Section 4.2). To read the attention of these positions only, the
    prompt is prefilled in two steps: all but the last ``n_obs`` tokens, then
    the last ``n_obs`` tokens with ``output_attentions=True``.

    The model must return attention weights. Transformers 4.43 falls back to
    the eager attention for that call by itself; later versions need
    ``attn_implementation="eager"`` at load time.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape (1, n)")
    n = input_ids.shape[1]

    if not with_importance:
        out = _forward(model, input_ids, None, 0)
        return Prefill(out.past_key_values, out.logits[:, -1], n, None)

    if n <= n_obs:
        raise ValueError(f"the prompt has {n} tokens; the importance needs more than n_obs={n_obs}")
    head = _forward(model, input_ids[:, : n - n_obs], None, 0)
    out = _forward(model, input_ids[:, n - n_obs :], head.past_key_values, n - n_obs, output_attentions=True)

    attentions = out.attentions
    if not attentions or any(a is None for a in attentions):
        raise RuntimeError(
            "the model returned no attention weights; load it with attn_implementation='eager'"
        )
    importance = []
    for attn, (keys, _) in zip(attentions, cache_layers(out.past_key_values)):
        n_query_heads, n_kv_heads = attn.shape[1], keys.shape[1]
        if attn.shape[-1] != n or n_query_heads % n_kv_heads:
            raise RuntimeError(f"unexpected attention shape {tuple(attn.shape)}")
        mass = attn[0].float().sum(dim=1)                                    # (H_q, n): sum over O
        mass = mass.view(n_kv_heads, n_query_heads // n_kv_heads, n).mean(dim=1)   # query heads of a KV head
        importance.append(mass)
    return Prefill(out.past_key_values, out.logits[:, -1], n, importance)


# --------------------------------------------------------------------------- compression


@torch.no_grad()
def compress(model, pre: Prefill, cfg: Optional[LoFiConfig] = None, keep_indices: bool = False):
    """Compress the prefilled cache to the budget, independently per layer and KV head.

    Returns the compressed cache, in which every layer and KV head holds
    exactly ``B`` tokens, and a :class:`CompressionStats`.
    """
    cfg = cfg or LoFiConfig()
    base = cfg.rope_base if cfg.rope_base is not None else rope_base(model)
    n = pre.length
    lay = layout(n, cfg, base)
    if cfg.use_importance and pre.importance is None:
        raise ValueError("prefill was run without importance")

    layers = cache_layers(pre.cache)
    criteria = {cfg.criterion, cfg.first_criterion} - {None}
    inv_freq = rope_inv_freq(model) if "derotated" in criteria else None
    generator = torch.Generator().manual_seed(cfg.seed)

    lo, hi = lay.n_sink, n - lay.window            # D = [lo, hi)
    n_protected = min(int(math.ceil(cfg.protect * lay.n_keep)), lay.n_distant) if cfg.use_importance else 0
    new_layers, passes, pair_removed, fallback_removed = [], [], [], []
    stats = CompressionStats(lay, n_protected, None, None, None)

    for idx, (keys, values) in enumerate(layers):
        if keys.shape[0] != 1 or keys.shape[-2] != n:
            raise ValueError(f"layer {idx}: expected a cache of shape (1, H, {n}, d), got {tuple(keys.shape)}")
        k, v = keys[0], values[0]                                            # (H, n, d)
        n_heads, _, head_dim = k.shape
        device = k.device
        low_idx, high_idx = low_high_channels(base, head_dim, device)

        importance = pre.importance[idx][:, lo:hi].to(device) if cfg.use_importance else None
        derotated = None
        if inv_freq is not None:
            derotated = derotate(k[:, lo:hi], torch.arange(lo, hi), inv_freq)

        out = select(
            k[:, lo:hi], v[:, lo:hi], importance, lay.n_keep, low_idx, high_idx,
            protect=cfg.protect, criterion=cfg.criterion, first_criterion=cfg.first_criterion,
            operator=cfg.operator, derotated_keys=derotated, generator=generator,
            max_passes=cfg.max_passes,
        )

        distant = out.active.nonzero(as_tuple=True)[1].view(n_heads, lay.n_keep) + lo
        sinks = torch.arange(lo, device=device).expand(n_heads, -1)
        recent = torch.arange(hi, n, device=device).expand(n_heads, -1)
        kept = torch.cat([sinks, distant, recent], dim=1)                    # (H, B), ascending

        if cfg.operator == "merge":
            k = torch.cat([k[:, :lo], out.keys, k[:, hi:]], dim=1)
            v = torch.cat([v[:, :lo], out.values, v[:, hi:]], dim=1)
            counts = torch.cat([sinks * 0 + 1, torch.gather(out.counts, 1, distant - lo).long(), recent * 0 + 1], dim=1)
            stats.counts.append(counts.cpu())
        gather_idx = kept.unsqueeze(-1).expand(-1, -1, head_dim)
        new_layers.append((torch.gather(k, 1, gather_idx).unsqueeze(0), torch.gather(v, 1, gather_idx).unsqueeze(0)))

        passes.append(out.passes.cpu())
        pair_removed.append(out.pair_removed.cpu())
        fallback_removed.append(out.fallback_removed.cpu())
        if keep_indices:
            stats.kept.append(kept.cpu())

    stats.passes = torch.stack(passes)
    stats.pair_removed = torch.stack(pair_removed)
    stats.fallback_removed = torch.stack(fallback_removed)
    return build_cache(new_layers, pre.cache), stats


# --------------------------------------------------------------------------- decoding


@torch.no_grad()
def greedy_decode(
    model,
    cache,
    logits: torch.Tensor,
    position: int,
    max_new_tokens: int,
    eos_token_id: Union[int, Sequence[int], None] = None,
) -> List[int]:
    """Greedy decoding on a (compressed) cache.

    Args:
        logits: ``(1, vocab)`` logits of the next token.
        position: true position of the next token, i.e. the number of tokens seen so far.
    """
    stop = set()
    if eos_token_id is not None:
        stop = {int(eos_token_id)} if isinstance(eos_token_id, int) else {int(t) for t in eos_token_id}
    tokens: List[int] = []
    for _ in range(max_new_tokens):
        token = logits.argmax(dim=-1)
        tokens.append(int(token))
        if tokens[-1] in stop or len(tokens) == max_new_tokens:
            break
        out = _forward(model, token.view(1, 1), cache, position)
        logits, cache = out.logits[:, -1], out.past_key_values
        position += 1
    return tokens


@torch.no_grad()
def generate(
    model,
    context_ids: torch.Tensor,
    query_ids: Optional[torch.Tensor] = None,
    cfg: Optional[LoFiConfig] = None,
    max_new_tokens: int = 64,
    eos_token_id: Union[int, Sequence[int], None] = None,
    compress_cache: bool = True,
    keep_indices: bool = False,
):
    """Prefill, compress once, decode greedily.

    Query-aware setting: pass the whole prompt as ``context_ids`` and leave
    ``query_ids`` empty. The cache is compressed after the query was prefilled.

    Query-agnostic setting: pass the context as ``context_ids`` and the rest of
    the prompt as ``query_ids``. The context is compressed first and the query
    is prefilled on the compressed cache.

    ``compress_cache=False`` runs the same two-step protocol on the full cache.
    ``keep_indices=True`` stores the kept positions of every layer and head in
    the returned statistics.

    Returns:
        ``(tokens, stats)``: the generated token ids and the compression
        statistics (None without compression).
    """
    cfg = cfg or LoFiConfig()
    stats = None
    if compress_cache and cfg.budget < 1.0:
        pre = prefill(model, context_ids, cfg.n_obs, with_importance=cfg.use_importance)
        cache, stats = compress(model, pre, cfg, keep_indices=keep_indices)
    else:
        pre = prefill(model, context_ids, with_importance=False)
        cache = pre.cache
    logits, position = pre.logits, pre.length
    if query_ids is not None and query_ids.shape[1] > 0:
        out = _forward(model, query_ids, cache, position)
        logits, cache = out.logits[:, -1], out.past_key_values
        position += query_ids.shape[1]
    return greedy_decode(model, cache, logits, position, max_new_tokens, eos_token_id), stats

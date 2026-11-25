"""Algorithm 1 of the paper: LoFi selection for the distant region of one layer.

The function :func:`select` runs Algorithm 1 for all KV heads of a layer at
once. Each head has its own active set, its own protected set and its own
kept tokens, exactly as if the algorithm were run head by head (the test
suite checks this against the plain per-head implementation in
:mod:`lofi.reference`).

Correspondence with Algorithm 1 (Appendix C):

    line 2   protected set P            -> :func:`protected_mask`
    line 5   s_j for adjacent pairs     -> :func:`pair_scores`
    line 6   local maxima, by score     -> ``cand`` / ``sel`` in :func:`select`
    line 8   target and source          -> ``left_is_target``
    line 9   evict or merge (Eq. 10)    -> ``operator``
    line 13  fallback eviction          -> end of :func:`select`

Points the paper leaves open, and the choice made here:

* Ties. A pair is a local maximum if its score is strictly above the pair on
  its left and at least the pair on its right, so two candidates never share a
  token. Candidates with equal scores are taken left to right.
* Equal importances are resolved in favour of the earlier position, both when
  the protected set is formed and in line 8. With this rule line 8 never
  evicts a protected token.
* Without importance (the "L, no imp." control) nothing is protected and the
  left token of a pair is kept.
* ``max_passes`` caps the number of passes of the loop of line 4. Tokens still
  to be removed when the cap is reached go to the fallback of line 13. Without
  a cap the pair rule always reaches the budget, because the protected set is
  smaller than the number of tokens to keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")

CRITERIA = ("low", "full", "derotated", "random")
OPERATORS = ("evict", "merge")


@dataclass
class SelectionOutput:
    """Result of Algorithm 1 on the distant region of one layer.

    Attributes:
        active: ``(H, N)`` bool, the kept positions of the distant region.
        keys, values: ``(H, N, d)`` tensors. Unchanged under ``evict``; under
            ``merge`` the kept targets carry the merged entries.
        counts: ``(H, N)`` number of original tokens each position stands for.
        passes: ``(H,)`` number of passes that removed at least one token.
        pair_removed: ``(H,)`` tokens removed by the pair rule.
        fallback_removed: ``(H,)`` tokens removed by the fallback of line 13.
    """

    active: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor
    passes: torch.Tensor
    pair_removed: torch.Tensor
    fallback_removed: torch.Tensor


def protected_mask(importance: torch.Tensor, n_keep: int, protect: float) -> torch.Tensor:
    """Line 2: the ``ceil(p * n_keep)`` positions of D with the highest importance, per head.

    Equal importances are resolved in favour of the earlier position.
    """
    h, n = importance.shape
    mask = torch.zeros(h, n, dtype=torch.bool, device=importance.device)
    n_protect = min(int(math.ceil(protect * n_keep)), n) if protect > 0 else 0
    if n_protect > 0:
        order = torch.argsort(importance, dim=1, descending=True, stable=True)
        mask.scatter_(1, order[:, :n_protect], True)
    return mask


def next_prev_active(active: torch.Tensor):
    """For every position, the next and the previous active position of its head.

    Returns ``(nxt, prv)`` of shape ``(H, N)``; ``nxt == N`` and ``prv == -1``
    mean "none".
    """
    h, n = active.shape
    idx = torch.arange(n, device=active.device).expand(h, n)
    fwd = torch.where(active, idx, torch.full_like(idx, n))
    nxt_incl = torch.flip(torch.cummin(torch.flip(fwd, dims=[1]), dim=1).values, dims=[1])
    nxt = torch.cat([nxt_incl[:, 1:], torch.full((h, 1), n, dtype=idx.dtype, device=idx.device)], dim=1)
    bwd = torch.where(active, idx, torch.full_like(idx, -1))
    prv_incl = torch.cummax(bwd, dim=1).values
    prv = torch.cat([torch.full((h, 1), -1, dtype=idx.dtype, device=idx.device), prv_incl[:, :-1]], dim=1)
    return nxt, prv


def pair_scores(features: torch.Tensor, nxt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Eq. (5): cosine of each active token with the next active token of its head.

    Args:
        features: ``(H, N, c)`` L2-normalised key features (for LoFi, the keys
            restricted to the low-frequency channels).
        nxt: ``(H, N)`` next active position, ``N`` if none.
        valid: ``(H, N)`` True where the pair ``(j, nxt[j])`` is scored.

    Returns:
        ``(H, N)`` scores, ``-inf`` where the pair is not valid.
    """
    h, n, c = features.shape
    index = nxt.clamp(max=n - 1).unsqueeze(-1).expand(h, n, c)
    partner = torch.gather(features, 1, index)
    scores = (features * partner).sum(dim=-1)
    return torch.where(valid, scores, torch.full_like(scores, NEG_INF))


def local_maxima(scores: torch.Tensor, nxt: torch.Tensor, prv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Line 6: pairs whose score is a local maximum along the active sequence."""
    n = scores.shape[1]
    neg = torch.full_like(scores, NEG_INF)
    left = torch.where(prv >= 0, torch.gather(scores, 1, prv.clamp(min=0)), neg)
    right = torch.where(nxt < n, torch.gather(scores, 1, nxt.clamp(max=n - 1)), neg)
    return valid & (scores > left) & (scores >= right)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


@torch.no_grad()
def select(
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: Optional[torch.Tensor],
    n_keep: int,
    low_idx: torch.Tensor,
    high_idx: torch.Tensor,
    *,
    protect: float = 0.1,
    criterion: str = "low",
    first_criterion: Optional[str] = None,
    operator: str = "evict",
    derotated_keys: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    max_passes: Optional[int] = None,
) -> SelectionOutput:
    """Run Algorithm 1 on the distant region D of one layer.

    Args:
        keys: ``(H, N, d)`` cached keys of D (as stored, i.e. after RoPE).
        values: ``(H, N, d)`` cached values of D.
        importance: ``(H, N)`` attention mass of each token from the last
            ``n_o`` prefilled positions, or None for the "no importance" control.
        n_keep: number of tokens of D to keep, ``B - n_s - W``.
        low_idx, high_idx: channel indices of L and H.
        protect: protected fraction ``p``.
        criterion: ``low`` (LoFi), ``full``, ``derotated`` or ``random``.
        first_criterion: if given, the criterion of the first pass only
            (the mixed-order controls of the pass analysis).
        operator: ``evict`` (LoFi-Sel) or ``merge`` (LoFi-M, Eq. 10).
        derotated_keys: ``(H, N, d)`` de-rotated keys, required by ``derotated``.
        generator: random generator for the ``random`` criterion.
        max_passes: stop the pair rule after this many passes and let the
            fallback of line 13 remove the rest. None (default) runs Algorithm 1
            as written; every pass removes at least one token per unfinished
            head, so it terminates.
    """
    used = {criterion} if first_criterion is None else {criterion, first_criterion}
    if not used <= set(CRITERIA):
        raise ValueError(f"criterion must be one of {CRITERIA}, got {sorted(used)}")
    if operator not in OPERATORS:
        raise ValueError(f"operator={operator!r} not in {OPERATORS}")
    if "derotated" in used:
        if derotated_keys is None:
            raise ValueError("criterion='derotated' needs derotated_keys")
        if operator == "merge":
            raise ValueError("the merge operator is not defined for the de-rotated control")
    h, n, _ = keys.shape
    if not 1 <= n_keep <= n:
        raise ValueError(f"n_keep={n_keep} outside [1, {n}]")
    device = keys.device

    keys = keys.clone() if operator == "merge" else keys
    values = values.clone() if operator == "merge" else values
    counts = torch.ones(h, n, dtype=torch.float32, device=device)

    if importance is None:
        imp = torch.zeros(h, n, dtype=torch.float32, device=device)
        protected = torch.zeros(h, n, dtype=torch.bool, device=device)
    else:
        imp = importance.float()
        protected = protected_mask(imp, n_keep, protect)

    active = torch.ones(h, n, dtype=torch.bool, device=device)
    remaining = torch.full((h,), n - n_keep, dtype=torch.long, device=device)   # r of Algorithm 1
    passes = torch.zeros(h, dtype=torch.long, device=device)
    pair_removed = torch.zeros(h, dtype=torch.long, device=device)
    rank = torch.arange(n, device=device).unsqueeze(0)

    cache = {}

    def features(name):
        """L2-normalised features of a criterion; recomputed only if a merge can change them."""
        if name in cache:
            return cache[name]
        if name == "low":
            feats = _normalise(keys.index_select(-1, low_idx))
        elif name == "full":
            feats = _normalise(keys)
        else:
            feats = _normalise(derotated_keys)
        if not (operator == "merge" and name == "full"):   # a merge changes the keys outside L only
            cache[name] = feats
        return feats

    pass_idx = -1
    while bool((remaining > 0).any()):
        pass_idx += 1
        if max_passes is not None and pass_idx >= max_passes:
            break
        name = first_criterion if (pass_idx == 0 and first_criterion is not None) else criterion
        nxt, prv = next_prev_active(active)
        partner_protected = torch.gather(protected, 1, nxt.clamp(max=n - 1))
        valid = active & (nxt < n) & ~(protected & partner_protected) & (remaining > 0).unsqueeze(1)

        if name == "random":
            scores = torch.rand(h, n, generator=generator, device="cpu").to(device)
            scores = torch.where(valid, scores, torch.full_like(scores, NEG_INF))
        else:
            scores = pair_scores(features(name), nxt, valid)

        cand = local_maxima(scores, nxt, prv, valid)
        take = torch.minimum(remaining, cand.sum(dim=1))
        if not bool((take > 0).any()):
            break

        # line 7: the first min(r, |M|) candidates in descending order of score
        order = torch.argsort(torch.where(cand, scores, torch.full_like(scores, NEG_INF)),
                              dim=1, descending=True, stable=True)
        sel = torch.zeros_like(cand).scatter_(1, order, rank < take.unsqueeze(1)) & cand

        head, left = sel.nonzero(as_tuple=True)
        right = nxt[head, left]
        left_protected, right_protected = protected[head, left], protected[head, right]
        # line 8: the more important token is the target; a protected token is never the source
        left_is_target = left_protected | (~right_protected & (imp[head, left] >= imp[head, right]))
        target = torch.where(left_is_target, left, right)
        source = torch.where(left_is_target, right, left)

        if operator == "merge":
            _merge(keys, values, counts, head, target, source, scores[head, left], high_idx)
        active[head, source] = False

        remaining -= take
        pair_removed += take
        passes += (take > 0).long()

    fallback_removed = remaining.clone()
    if bool((remaining > 0).any()):
        # line 13: evict the r unprotected positions with the lowest importance
        pool = active & ~protected
        key = torch.where(pool, imp, torch.full_like(imp, float("inf")))
        order = torch.argsort(key, dim=1, stable=True)
        drop = torch.zeros_like(pool).scatter_(1, order, rank < remaining.unsqueeze(1)) & pool
        active &= ~drop

    kept = active.sum(dim=1)
    if not bool((kept == n_keep).all()):
        raise RuntimeError(f"selection kept {kept.tolist()} tokens, expected {n_keep}")
    return SelectionOutput(active, keys, values, counts, passes, pair_removed, fallback_removed)


def _merge(keys, values, counts, head, target, source, score, high_idx) -> None:
    """Eq. (10): content-preserving merge of ``source`` into ``target``, in place.

    The low-frequency channels of the target are left untouched. Its
    high-frequency channels move toward the source and are rescaled to their
    norm before the merge. The value moves toward the source value.
    """
    s = score.clamp(0.0, 1.0).float().unsqueeze(-1)
    alpha = 0.2 + 0.3 * s
    alpha_v = 0.15 + 0.15 * s

    k_target = keys[head, target]
    high_target = k_target.index_select(-1, high_idx).float()
    high_source = keys[head, source].index_select(-1, high_idx).float()
    norm_before = high_target.norm(dim=-1, keepdim=True)
    mixed = (1.0 - alpha) * high_target + alpha * high_source
    mixed = mixed * (norm_before / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    k_target = k_target.clone()
    k_target[:, high_idx] = mixed.to(keys.dtype)
    keys[head, target] = k_target

    v_new = (1.0 - alpha_v) * values[head, target].float() + alpha_v * values[head, source].float()
    values[head, target] = v_new.to(values.dtype)
    counts[head, target] = counts[head, target] + counts[head, source]

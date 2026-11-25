"""Algorithm 1 for one KV head, written line by line with Python loops.

This file exists for reading and for testing. It follows the pseudocode of
Appendix C as literally as possible and is slow. :mod:`lofi.selection` is the
vectorised version used in practice; ``tests/test_selection.py`` checks that
both return the same kept set, pass by pass, on random inputs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def select_one_head(
    keys: torch.Tensor,
    importance: Optional[Sequence[float]],
    n_keep: int,
    low_idx: torch.Tensor,
    protect: float = 0.1,
    max_passes: Optional[int] = None,
) -> Tuple[List[int], int]:
    """LoFi-Sel on the distant region of one KV head.

    Args:
        keys: ``(N, d)`` cached keys of D.
        importance: ``N`` importances, or None.
        n_keep: tokens of D to keep, ``B - n_s - W``.
        low_idx: channel indices of L.
        protect: protected fraction ``p``.
        max_passes: cap on the passes of the pair rule. None runs the loop of
            line 4 as written.

    Returns:
        The kept positions of D in ascending order and the number of passes.
    """
    n = keys.shape[0]
    imp = [0.0] * n if importance is None else [float(x) for x in importance]
    # Eq. (5) needs the keys restricted to L, with unit norm
    low = F.normalize(keys[:, low_idx].float(), dim=-1, eps=1e-8)

    # line 1: r <- |D| - (B - n_s - W)
    r = n - n_keep

    # line 2: P <- the ceil(p (B - n_s - W)) positions of D with the highest imp
    protected = set()
    if importance is not None and protect > 0:
        n_protect = min(int(math.ceil(protect * n_keep)), n)
        protected = set(sorted(range(n), key=lambda j: (-imp[j], j))[:n_protect])

    # line 3: A <- D
    active = list(range(n))
    passes = 0

    # line 4: while r > 0 and A contains an adjacent pair not inside P
    while r > 0 and (max_passes is None or passes < max_passes):
        pairs = [(active[i], active[i + 1]) for i in range(len(active) - 1)]
        valid = [not (j in protected and j2 in protected) for j, j2 in pairs]
        if not any(valid):
            break

        # line 5: s_j <- Eq. (5) for every adjacent (j, j') in A with {j, j'} not inside P
        left_idx = torch.tensor([j for j, _ in pairs])
        right_idx = torch.tensor([j2 for _, j2 in pairs])
        cos = (low[left_idx] * low[right_idx]).sum(dim=-1).tolist()
        score = [c if ok else -math.inf for c, ok in zip(cos, valid)]

        # line 6: M <- pairs whose score is a local maximum, in descending order of s_j
        candidates = []
        for i, ok in enumerate(valid):
            if not ok:
                continue
            left = score[i - 1] if i > 0 else -math.inf
            right = score[i + 1] if i + 1 < len(score) else -math.inf
            if score[i] > left and score[i] >= right:
                candidates.append(i)
        candidates.sort(key=lambda i: (-score[i], i))

        # line 7: for the first min(r, |M|) pairs (j, j') in M
        removed = set()
        for i in candidates[: min(r, len(candidates))]:
            j, j2 = pairs[i]
            # line 8: (t, u) <- (j, j') if imp(j) >= imp(j') or j in P, else (j', j)
            if j in protected:
                u = j2
            elif j2 in protected:
                u = j
            else:
                u = j2 if imp[j] >= imp[j2] else j
            # line 9: evict u;  line 10: A <- A \ {u}; r <- r - 1
            removed.add(u)
        active = [j for j in active if j not in removed]
        r -= len(removed)
        passes += 1

    # line 13: if r > 0, evict the r positions of A \ P with the lowest imp
    if r > 0:
        pool = sorted((j for j in active if j not in protected), key=lambda j: (imp[j], j))
        drop = set(pool[:r])
        active = [j for j in active if j not in drop]

    return active, passes

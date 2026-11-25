"""Algorithm 1 for one KV head, written line by line with Python loops."""

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
    """LoFi-Sel on the distant region of one KV head."""
    n = keys.shape[0]
    imp = [0.0] * n if importance is None else [float(x) for x in importance]
    low = F.normalize(keys[:, low_idx].float(), dim=-1, eps=1e-8)

    r = n - n_keep

    protected = set()
    if importance is not None and protect > 0:
        n_protect = min(int(math.ceil(protect * n_keep)), n)
        protected = set(sorted(range(n), key=lambda j: (-imp[j], j))[:n_protect])

    active = list(range(n))
    passes = 0

    while r > 0 and (max_passes is None or passes < max_passes):
        pairs = [(active[i], active[i + 1]) for i in range(len(active) - 1)]
        valid = [not (j in protected and j2 in protected) for j, j2 in pairs]
        if not any(valid):
            break

        left_idx = torch.tensor([j for j, _ in pairs])
        right_idx = torch.tensor([j2 for _, j2 in pairs])
        cos = (low[left_idx] * low[right_idx]).sum(dim=-1).tolist()
        score = [c if ok else -math.inf for c, ok in zip(cos, valid)]

        candidates = []
        for i, ok in enumerate(valid):
            if not ok:
                continue
            left = score[i - 1] if i > 0 else -math.inf
            right = score[i + 1] if i + 1 < len(score) else -math.inf
            if score[i] > left and score[i] >= right:
                candidates.append(i)
        candidates.sort(key=lambda i: (-score[i], i))

        removed = set()
        for i in candidates[: min(r, len(candidates))]:
            j, j2 = pairs[i]
            if j in protected:
                u = j2
            elif j2 in protected:
                u = j
            else:
                u = j2 if imp[j] >= imp[j2] else j
            removed.add(u)
        active = [j for j in active if j not in removed]
        r -= len(removed)
        passes += 1

    if r > 0:
        pool = sorted((j for j in active if j not in protected), key=lambda j: (imp[j], j))
        drop = set(pool[:r])
        active = [j for j in active if j not in drop]

    return active, passes

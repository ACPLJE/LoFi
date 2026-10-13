"""RoPE frequency partition at the sqrt(base) boundary.

Splits a head's channels into a position-invariant low-frequency set and a
position-dependent high-frequency set. The boundary is derived from the RoPE
base alone, so it needs no calibration data and no per-model tuning.
"""

from __future__ import annotations

import math

import torch

# Channel index sets are reused on every layer and every forward pass, so they
# are cached per (head_dim, rope_base, device) rather than rebuilt each call.
_LOW_FREQ_CACHE: dict[tuple[int, float, str], torch.Tensor] = {}
_HIGH_FREQ_CACHE: dict[tuple[int, float, str], torch.Tensor] = {}


def low_freq_indices(head_dim: int, rope_base: float, device: torch.device) -> torch.Tensor:
    """Channel indices whose RoPE rotation is position-invariant over the cache.

    A channel pair with frequency theta_i = base^(-2i/d) rotates by
    Delta * theta_i over a token distance Delta. Pairs stay within a quarter
    turn (and so keep cos(Delta * theta_i) ~ 1) while Delta * theta_i <= pi/2.
    Taking Delta ~ sqrt(base) as the reference distance gives the pair cutoff
    `kb` below; every pair above it is treated as low-frequency content.

    Returns:
        1-D long tensor of channel indices into the last axis of a
        (..., head_dim) tensor. Under the HF rotate_half layout a pair `i`
        occupies channels (i, i + head_dim // 2), so the returned set is the
        union of two contiguous blocks, sorted ascending.
    """
    key = (head_dim, rope_base, str(device))
    cached = _LOW_FREQ_CACHE.get(key)
    if cached is not None:
        return cached

    half = head_dim // 2
    ref_dist = rope_base**0.5
    if ref_dist <= math.pi / 2:
        kb = 0
    else:
        kb = math.ceil(half * math.log(2.0 * ref_dist / math.pi) / math.log(rope_base))
    kb = max(0, min(kb, half - 1))
    n_low = max(1, half - kb)

    idx = sorted(list(range(half - n_low, half)) + list(range(head_dim - n_low, head_dim)))
    out = torch.tensor(idx, dtype=torch.long, device=device)
    _LOW_FREQ_CACHE[key] = out
    return out


def high_freq_indices(head_dim: int, rope_base: float, device: torch.device) -> torch.Tensor:
    """Complement of :func:`low_freq_indices`: the position-dependent channels.

    Returns:
        1-D long tensor of channel indices, sorted ascending.
    """
    key = (head_dim, rope_base, str(device))
    cached = _HIGH_FREQ_CACHE.get(key)
    if cached is not None:
        return cached

    low = set(low_freq_indices(head_dim, rope_base, device).tolist())
    high = sorted(set(range(head_dim)) - low)
    out = torch.tensor(high, dtype=torch.long, device=device)
    _HIGH_FREQ_CACHE[key] = out
    return out

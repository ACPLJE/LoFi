"""RoPE frequency structure used by LoFi (paper Section 3 and Section 4.2).

* Definition 1: the low-frequency set L holds the channel pairs whose effective
  range ``pi / (2 * theta_i)`` exceeds ``sqrt(base)``.
* Eq. (4): the recent window ``W = min(floor(sqrt(base)), floor((B - n_s) / 2))``.

Everything here follows from ``(base, d)`` alone. No data is involved.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def rope_thetas(base: float, head_dim: int) -> torch.Tensor:
    """``theta_i = base^(-2i/d)`` for pair ``i`` in ``[0, d/2)`` (float64)."""
    i = torch.arange(head_dim // 2, dtype=torch.float64)
    return float(base) ** (-2.0 * i / head_dim)


def effective_ranges(base: float, head_dim: int) -> torch.Tensor:
    """``Delta*_i = pi / (2 * theta_i)``: distance at which pair ``i`` has turned a quarter."""
    return (math.pi / 2.0) / rope_thetas(base, head_dim)


def low_frequency_pairs(base: float, head_dim: int) -> torch.Tensor:
    """Definition 1: indices ``i`` of the channel pairs with ``Delta*_i > sqrt(base)``."""
    ranges = effective_ranges(base, head_dim)
    return torch.nonzero(ranges > math.sqrt(float(base)), as_tuple=False).squeeze(-1)


def pair_to_channels(pairs: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Flat channel indices of the given pairs.

    Hugging Face models apply RoPE with the ``rotate_half`` layout, in which
    pair ``i`` occupies channels ``(i, i + d/2)``.
    """
    half = head_dim // 2
    return torch.sort(torch.cat([pairs, pairs + half])).values


def low_high_channels(
    base: float, head_dim: int, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Channel indices of L and of its complement H, both sorted ascending."""
    low = pair_to_channels(low_frequency_pairs(base, head_dim), head_dim)
    mask = torch.ones(head_dim, dtype=torch.bool)
    mask[low] = False
    high = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if device is not None:
        low, high = low.to(device), high.to(device)
    return low, high


def recent_window(base: float, budget: int, n_sink: int) -> int:
    """Eq. (4): ``W = min(floor(sqrt(base)), floor((B - n_s) / 2))``."""
    return max(0, min(int(math.floor(math.sqrt(float(base)))), (int(budget) - int(n_sink)) // 2))


def default_inv_freq(base: float, head_dim: int) -> torch.Tensor:
    """Inverse frequencies of unscaled RoPE, ``base^(-2i/d)``."""
    return rope_thetas(base, head_dim)


def derotate(keys: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Undo the RoPE rotation of cached keys (used only by the ``derotated`` control).

    Args:
        keys: ``(..., n, d)`` cached keys, ``rotate_half`` layout.
        positions: ``(n,)`` absolute positions of the keys.
        inv_freq: ``(d/2,)`` inverse frequencies the model applies. Pass the
            model's own values when its RoPE is rescaled (e.g. Llama-3.1).

    Returns:
        Keys rotated back to position 0, same shape, float32.
    """
    k = keys.float()
    angles = positions.to(k.device).double()[:, None] * inv_freq.to(k.device).double()[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).float()
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).float()
    half = k.shape[-1] // 2
    rot_half = torch.cat([-k[..., half:], k[..., :half]], dim=-1)
    # forward rotation is k*cos + rotate_half(k)*sin, so the inverse flips the sign of sin
    return k * cos - rot_half * sin

"""Definition 1, Eq. (4) and the constants of Table 1."""

import math

import pytest
import torch

from lofi.frequency import (
    derotate,
    effective_ranges,
    low_frequency_pairs,
    low_high_channels,
    recent_window,
    rope_thetas,
)

# (base, |L|, floor(sqrt(base))) for d = 128: Mistral-7B, Llama-3 / 3.1, Qwen2 / 2.5
TABLE_1 = [(1e4, 35, 100), (5e5, 34, 707), (1e6, 34, 1000)]


@pytest.mark.parametrize("base,size,window", TABLE_1)
def test_table_1_constants(base, size, window):
    assert low_frequency_pairs(base, 128).numel() == size
    assert int(math.floor(math.sqrt(base))) == window


@pytest.mark.parametrize("base,size,_", TABLE_1)
def test_low_set_is_definition_1(base, size, _):
    pairs = low_frequency_pairs(base, 128)
    ranges = effective_ranges(base, 128)
    # theta decreases with the pair index, so L is the block of the slowest pairs
    assert pairs.tolist() == list(range(64 - size, 64))
    assert float(ranges[pairs[0]]) > math.sqrt(base) >= float(ranges[pairs[0] - 1])
    assert torch.allclose(ranges * rope_thetas(base, 128), torch.full((64,), math.pi / 2, dtype=torch.float64))


@pytest.mark.parametrize("base,size,_", TABLE_1)
def test_channel_layout(base, size, _):
    low, high = low_high_channels(base, 128)
    assert low.numel() == 2 * size and high.numel() == 128 - 2 * size
    assert sorted(low.tolist() + high.tolist()) == list(range(128))
    pairs = low_frequency_pairs(base, 128).tolist()
    assert low.tolist() == pairs + [p + 64 for p in pairs]      # rotate_half layout: (i, i + d/2)


def test_window_eq_4():
    # Llama-3-8B, 6,500 prefilled tokens, 20% budget: the cap is active
    assert recent_window(5e5, 1300, 32) == 634
    assert round(100 * 634 / (6500 - 32 - 634), 1) == 10.9
    # Mistral-7B, 5,400 prefilled tokens: W = floor(sqrt(base))
    assert recent_window(1e4, 1080, 32) == 100
    assert round(100 * (1080 - 32 - 100) / (5400 - 32 - 100), 1) == 18.0
    assert round(100 / 1080, 2) == 0.09


@pytest.mark.parametrize("base", [1e4, 5e5, 1e6])
def test_window_cap_threshold(base):
    """At a 20% budget the cap binds for prompts shorter than 10 sqrt(base) + 160 tokens."""
    root = int(math.floor(math.sqrt(base)))
    for n in range(200, 12000, 37):
        budget = n // 5
        window = recent_window(base, budget, 32)
        assert window <= (budget - 32) // 2            # the selection decides at least half of B - n_s
        assert window == (root if n >= 10 * root + 160 else (budget - 32) // 2)


def _rope(keys, positions, inv_freq):
    """RoPE as applied by Hugging Face models (rotate_half layout)."""
    angles = positions.double()[:, None] * inv_freq.double()[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    half = keys.shape[-1] // 2
    rotated = torch.cat([-keys[..., half:], keys[..., :half]], dim=-1)
    return keys * cos + rotated * sin


def test_derotate_inverts_rope():
    torch.manual_seed(0)
    inv_freq = rope_thetas(5e5, 128)
    positions = torch.arange(40, 4040, 7)
    plain = torch.randn(3, positions.numel(), 128, dtype=torch.float64)
    cached = _rope(plain, positions, inv_freq)
    assert torch.allclose(derotate(cached, positions, inv_freq).double(), plain, atol=1e-4)


@pytest.mark.parametrize("base,_,__", TABLE_1)
def test_quarter_turn_at_sqrt_base(base, _, __):
    """At distance sqrt(base), the pairs of L have turned less than a quarter and those of H at least one.

    The same content cached at two positions is used: the cosine of pair i is then cos(distance * theta_i).
    """
    torch.manual_seed(0)
    inv_freq = rope_thetas(base, 128)
    distance = math.sqrt(base)
    content = torch.randn(16, 128, dtype=torch.float64)
    near = _rope(content, torch.zeros(16, dtype=torch.float64), inv_freq)
    far = _rope(content, torch.full((16,), distance, dtype=torch.float64), inv_freq)

    def pair_cosine(i):
        a, b = near[:, [i, i + 64]], far[:, [i, i + 64]]
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    low = set(low_frequency_pairs(base, 128).tolist())
    for i in range(64):
        expected = math.cos(distance * float(inv_freq[i]))
        assert torch.allclose(pair_cosine(i), torch.full((16,), expected, dtype=torch.float64), atol=1e-9)
        assert (distance * float(inv_freq[i]) < math.pi / 2) == (i in low)

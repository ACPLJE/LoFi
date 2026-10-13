"""Properties of the sqrt(base) frequency partition."""

import math

import pytest
import torch

from lofi.frequency import high_freq_indices, low_freq_indices

CPU = torch.device("cpu")


@pytest.mark.parametrize("head_dim,rope_base", [(128, 10000.0), (128, 500000.0),
                                                (64, 10000.0), (256, 1000000.0)])
def test_partition_is_a_partition(head_dim, rope_base):
    low = low_freq_indices(head_dim, rope_base, CPU)
    high = high_freq_indices(head_dim, rope_base, CPU)
    assert set(low.tolist()).isdisjoint(high.tolist())
    assert sorted(low.tolist() + high.tolist()) == list(range(head_dim))


@pytest.mark.parametrize("head_dim,rope_base", [(128, 10000.0), (128, 500000.0),
                                                (64, 10000.0), (256, 1000000.0)])
def test_low_freq_respects_rotate_half_pairing(head_dim, rope_base):
    """Under rotate_half, pair i is channels (i, i + head_dim/2).

    A pair must be wholly low- or wholly high-frequency; splitting one would
    rotate half a pair and mix the two regimes.
    """
    low = set(low_freq_indices(head_dim, rope_base, CPU).tolist())
    half = head_dim // 2
    for i in range(half):
        assert (i in low) == (i + half in low), f"pair {i} is split"


@pytest.mark.parametrize("head_dim,rope_base", [(128, 10000.0), (128, 500000.0),
                                                (64, 10000.0), (256, 1000000.0)])
def test_low_freq_pairs_are_the_slowest(head_dim, rope_base):
    """theta_i decreases with i, so the low-frequency set is a suffix of pairs."""
    low = sorted(low_freq_indices(head_dim, rope_base, CPU).tolist())
    half = head_dim // 2
    pairs = sorted({i % half for i in low})
    assert pairs == list(range(half - len(pairs), half))


def test_boundary_matches_the_quarter_turn_criterion():
    """The cutoff pair is the one that just stays inside a quarter turn at
    Delta = sqrt(base)."""
    head_dim, rope_base = 128, 500000.0
    low = sorted(low_freq_indices(head_dim, rope_base, CPU).tolist())
    half = head_dim // 2
    n_low = len(low) // 2
    first_low_pair = half - n_low

    delta = rope_base**0.5
    theta = lambda i: rope_base ** (-2.0 * i / head_dim)  # noqa: E731
    assert delta * theta(first_low_pair) <= math.pi / 2
    if first_low_pair > 0:
        assert delta * theta(first_low_pair - 1) > math.pi / 2


def test_small_base_degenerates_to_all_low():
    """With base <= (pi/2)^2 no pair completes a quarter turn, so nothing is
    treated as position-dependent high frequency."""
    low = low_freq_indices(128, 2.0, CPU)
    assert low.numel() == 128


def test_indices_are_cached_per_key():
    a = low_freq_indices(128, 500000.0, CPU)
    b = low_freq_indices(128, 500000.0, CPU)
    assert a is b

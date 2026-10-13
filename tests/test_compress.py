"""Behaviour of the compression core, on synthetic caches (CPU only)."""

import pytest
import torch

from lofi.config import PRESETS, LoFiConfig
from lofi.core import N_SINK, compress_kv_cache, mass_logit_bias
from lofi.frequency import low_freq_indices

HEAD_DIM = 128
ROPE_BASE = 500000.0


def make_cache(seq_len: int, n_heads: int = 4, seed: int = 0):
    """Random (keys, values, attn_weights) with the shapes a prefill produces."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, n_heads, seq_len, HEAD_DIM, generator=gen)
    values = torch.randn(1, n_heads, seq_len, HEAD_DIM, generator=gen)
    logits = torch.randn(1, n_heads, seq_len, seq_len, generator=gen)
    attn = torch.softmax(logits.tril().masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1), float("-inf")
    ), dim=-1)
    return keys, values, attn


def compress(seq_len=1024, config=None, with_attn=True, seed=0):
    keys, values, attn = make_cache(seq_len, seed=seed)
    config = config or LoFiConfig()
    return compress_kv_cache(
        keys, values, ROPE_BASE, config,
        attn_weights=attn if with_attn else None,
    )


@pytest.mark.parametrize("target", [0.2, 0.3, 0.5])
def test_reaches_the_configured_budget(target):
    config = LoFiConfig(hh_ratio=target / 2, recent_ratio=target / 2)
    result = compress(seq_len=2048, config=config)
    budget = max(4, int(2048 * target))
    assert result.keys.shape[2] == budget
    assert result.values.shape[2] == budget
    assert result.stats.retention == pytest.approx(budget / 2048, abs=1e-3)


def test_multiple_passes_are_needed_below_half_retention():
    """A single pass cannot remove more than half the compressible region."""
    result = compress(seq_len=2048, config=LoFiConfig(hh_ratio=0.1, recent_ratio=0.1))
    assert result.stats.passes > 1


def test_single_pass_misses_a_low_budget_and_the_fallback_covers_it():
    config = LoFiConfig(hh_ratio=0.1, recent_ratio=0.1, max_passes=1)
    result = compress(seq_len=2048, config=config)
    assert result.stats.passes == 1
    assert result.stats.fallback_evicted > 0
    # The budget is still met, but by eviction rather than merging.
    assert result.keys.shape[2] == max(4, int(2048 * 0.2))


def test_passthrough_when_already_within_budget():
    keys, values, attn = make_cache(64)
    config = LoFiConfig(hh_ratio=0.5, recent_ratio=0.5)
    result = compress_kv_cache(keys, values, ROPE_BASE, config, attn)
    assert result.keys is keys
    assert result.values is values
    assert result.merge_counts is None
    assert result.stats is None


def test_merge_counts_conserve_token_mass():
    """With no fallback eviction every original token is represented exactly
    once in the counts."""
    result = compress(seq_len=2048, config=LoFiConfig(hh_ratio=0.15, recent_ratio=0.15))
    assert result.stats.fallback_evicted == 0
    assert result.merge_counts.sum().item() == pytest.approx(2048.0)


def test_sink_and_recent_slots_have_unit_count():
    result = compress(seq_len=2048)
    counts = result.merge_counts[0]
    assert torch.all(counts[:N_SINK] == 1.0)
    window = max(32, int(2048 * 0.1))
    assert torch.all(counts[-window:] == 1.0)


def test_content_preserving_merge_leaves_low_freq_channels_exact():
    """Every surviving key's low-frequency block must still be bit-identical to
    some original key's, which is what makes the similarity score stable."""
    keys, values, attn = make_cache(1024)
    config = LoFiConfig(hh_ratio=0.15, recent_ratio=0.15)
    result = compress_kv_cache(keys, values, ROPE_BASE, config, attn)

    low = low_freq_indices(HEAD_DIM, ROPE_BASE, keys.device)
    original = keys[0, 0].index_select(-1, low)
    compressed = result.keys[0, 0].index_select(-1, low)
    for row in compressed:
        assert torch.isclose(original, row, atol=0).all(dim=-1).any(), \
            "a surviving key's low-frequency block was modified"


def test_evict_mode_keeps_all_counts_at_one():
    config = LoFiConfig(**PRESETS["select_evict"], hh_ratio=0.1, recent_ratio=0.1)
    result = compress(seq_len=2048, config=config)
    assert torch.all(result.merge_counts == 1.0)


def test_evict_mode_does_not_touch_surviving_vectors():
    keys, values, attn = make_cache(1024)
    config = LoFiConfig(**PRESETS["select_evict"], hh_ratio=0.15, recent_ratio=0.15)
    result = compress_kv_cache(keys, values, ROPE_BASE, config, attn)
    for row in result.keys[0, 0]:
        assert torch.isclose(keys[0, 0], row, atol=0).all(dim=-1).any()


def test_uniform_merge_changes_low_freq_channels():
    """The contrast to the content-preserving merge: a naive average corrupts
    the position-invariant block too."""
    keys, values, attn = make_cache(1024)
    config = LoFiConfig(**PRESETS["uniform_merge"], hh_ratio=0.15, recent_ratio=0.15)
    result = compress_kv_cache(keys, values, ROPE_BASE, config, attn)

    low = low_freq_indices(HEAD_DIM, ROPE_BASE, keys.device)
    original = keys[0, 0].index_select(-1, low)
    compressed = result.keys[0, 0].index_select(-1, low)
    untouched = [bool(torch.isclose(original, row, atol=0).all(dim=-1).any())
                 for row in compressed]
    assert not all(untouched)


def test_disabled_config_is_rejected_by_the_core():
    """`disabled` is honoured by the attention wrapper, not here: calling the
    core directly still compresses, so the flag cannot be silently ignored."""
    config = LoFiConfig(disabled=True, hh_ratio=0.1, recent_ratio=0.1)
    result = compress(seq_len=1024, config=config)
    assert result.stats is not None


def test_batch_size_above_one_is_rejected():
    keys, values, attn = make_cache(512)
    keys = keys.repeat(2, 1, 1, 1)
    values = values.repeat(2, 1, 1, 1)
    attn = attn.repeat(2, 1, 1, 1)
    with pytest.raises(ValueError, match="batch size 1"):
        compress_kv_cache(keys, values, ROPE_BASE, LoFiConfig(), attn)


def test_works_without_importance_signal():
    result = compress(seq_len=1024, with_attn=False)
    assert result.keys.shape[2] == max(4, int(1024 * 0.2))


def test_mass_logit_bias_is_log_of_counts_and_pads_with_zero():
    counts = torch.tensor([[1.0, 2.0, 4.0]])
    bias = mass_logit_bias(counts, n_keys=5)
    assert bias.shape == (1, 1, 1, 5)
    expected = torch.tensor([0.0, torch.log(torch.tensor(2.0)),
                             torch.log(torch.tensor(4.0)), 0.0, 0.0])
    assert torch.allclose(bias.flatten(), expected)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_runs_and_hits_the_budget(preset):
    if preset == "full_cache":
        pytest.skip("full_cache is handled by the attention wrapper")
    config = LoFiConfig(**PRESETS[preset], hh_ratio=0.1, recent_ratio=0.1)
    result = compress(seq_len=2048, config=config)
    assert result.keys.shape[2] == max(4, int(2048 * 0.2))


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="merge_mode"):
        LoFiConfig(merge_mode="bogus")
    with pytest.raises(ValueError, match="protect_ratio"):
        LoFiConfig(protect_ratio=1.0)
    with pytest.raises(ValueError, match="max_passes"):
        LoFiConfig(max_passes=0)

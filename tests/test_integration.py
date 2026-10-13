"""End-to-end checks on a tiny randomly-initialised model (CPU only).

These exercise the attention patch itself: that compression fires once at
prefill, that decoding over a compressed cache advances positions correctly,
and that the mass-correction bias is wired into the logits.
"""

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from lofi.config import PRESETS, LoFiConfig
from lofi.models.llama import LoFiLlamaAttention, LoFiLlamaForCausalLM, reset_caches

SEQ_LEN = 1024
TARGET = 0.2


def build_model(lofi: LoFiConfig) -> LoFiLlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=4096, rope_theta=500000.0,
    )
    config.lofi_config = lofi
    torch.manual_seed(0)
    return LoFiLlamaForCausalLM(config).eval()


def layers(model) -> list[LoFiLlamaAttention]:
    return [m for m in model.modules() if isinstance(m, LoFiLlamaAttention)]


@pytest.fixture(scope="module")
def prefilled():
    """Prefill once and reuse: building the model is the slow part."""
    model = build_model(LoFiConfig(hh_ratio=0.1, recent_ratio=0.1))
    ids = torch.randint(0, 256, (1, SEQ_LEN), generator=torch.Generator().manual_seed(1))
    reset_caches(model)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return model, ids, out


def test_prefill_compresses_every_layer_to_the_budget(prefilled):
    model, ids, out = prefilled
    expected = max(4, int(SEQ_LEN * TARGET))
    for keys, values in out.past_key_values:
        assert keys.shape[-2] == expected
        assert values.shape[-2] == expected
    for layer in layers(model):
        assert layer.compressed
        assert layer.stats.kept == expected


def test_decode_over_compressed_cache_stays_finite(prefilled):
    model, ids, out = prefilled
    past = out.past_key_values
    token = out.logits[:, -1:].argmax(-1)
    start = past[0][0].shape[-2]
    with torch.no_grad():
        for step in range(5):
            # True positions continue past the prompt even though the cache is
            # far shorter, which is why positions are passed explicitly.
            positions = torch.tensor([[SEQ_LEN + step]])
            out = model(input_ids=token, past_key_values=past,
                        position_ids=positions, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1:].argmax(-1)
            assert torch.isfinite(out.logits).all()
    assert past[0][0].shape[-2] == start + 5


def test_compression_fires_only_once(prefilled):
    model, ids, out = prefilled
    before = [layer.stats for layer in layers(model)]
    past = out.past_key_values
    token = out.logits[:, -1:].argmax(-1)
    with torch.no_grad():
        model(input_ids=token, past_key_values=past,
              position_ids=torch.tensor([[SEQ_LEN]]), use_cache=True)
    assert [layer.stats for layer in layers(model)] == before


def test_merge_counts_account_for_every_token_except_evicted(prefilled):
    model, _ids, _out = prefilled
    for layer in layers(model):
        total = float(layer.merge_counts.sum())
        assert total == pytest.approx(SEQ_LEN - layer.stats.fallback_evicted)


def test_reset_clears_state_between_examples(prefilled):
    model, _ids, _out = prefilled
    reset_caches(model)
    for layer in layers(model):
        assert not layer.compressed
        assert layer.merge_counts is None
        assert layer.stats is None


def test_disabled_config_skips_compression():
    model = build_model(LoFiConfig(**PRESETS["full_cache"],
                                   hh_ratio=0.1, recent_ratio=0.1))
    ids = torch.randint(0, 256, (1, 512), generator=torch.Generator().manual_seed(2))
    reset_caches(model)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    for keys, _values in out.past_key_values:
        assert keys.shape[-2] == 512
    for layer in layers(model):
        assert not layer.compressed


def test_mass_correction_changes_decode_logits():
    """The log-count bias must actually reach the softmax; with it disabled the
    same compressed cache produces different next-token logits."""
    outputs = {}
    for mass_mode in ("logn", "none"):
        model = build_model(LoFiConfig(hh_ratio=0.1, recent_ratio=0.1,
                                       mass_mode=mass_mode))
        ids = torch.randint(0, 256, (1, SEQ_LEN),
                            generator=torch.Generator().manual_seed(1))
        reset_caches(model)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            token = out.logits[:, -1:].argmax(-1)
            out = model(input_ids=token, past_key_values=out.past_key_values,
                        position_ids=torch.tensor([[SEQ_LEN]]), use_cache=True)
        outputs[mass_mode] = out.logits.float()
    assert not torch.allclose(outputs["logn"], outputs["none"])

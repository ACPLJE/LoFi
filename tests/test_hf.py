"""End-to-end checks on small, randomly initialised Hugging Face models (CPU, no download)."""

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from lofi import LoFiConfig, compress, disable_sliding_window, generate, greedy_decode, layout, prefill, rope_base
from lofi.compress import _forward, cache_layers, rope_inv_freq
from lofi.frequency import derotate, low_high_channels
from lofi.reference import select_one_head

VOCAB = 97
SMALL = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)


def build(kind, eager=True):
    torch.manual_seed(0)
    if kind == "llama":            # RoPE base of Llama-3
        config = transformers.LlamaConfig(rope_theta=500000.0, **SMALL)
        cls = transformers.LlamaForCausalLM
    elif kind == "llama31":        # rescaled RoPE, as in Llama-3.1
        scaling = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                       original_max_position_embeddings=256)
        try:
            config = transformers.LlamaConfig(rope_theta=500000.0, rope_scaling=scaling, **SMALL)
            assert rope_base_of(config) == 500000.0
        except Exception:
            config = transformers.LlamaConfig(rope_parameters=dict(rope_theta=500000.0, **scaling), **SMALL)
        cls = transformers.LlamaForCausalLM
    elif kind == "mistral":        # ships with a sliding window, as Mistral-7B-v0.1
        config = transformers.MistralConfig(rope_theta=10000.0, sliding_window=64, **SMALL)
        cls = transformers.MistralForCausalLM
    else:                          # qwen2
        config = transformers.Qwen2Config(rope_theta=1000000.0, **SMALL)
        cls = transformers.Qwen2ForCausalLM
    if eager:
        config._attn_implementation = "eager"  # attention weights are needed for the importance
    model = cls(config).eval()
    disable_sliding_window(model)
    return model


def rope_base_of(config):
    base = getattr(config, "rope_theta", None)
    return base if base is not None else config.rope_parameters["rope_theta"]


KINDS = ["llama", "llama31", "mistral", "qwen2"]


@pytest.fixture(scope="module", params=KINDS)
def model(request):
    return build(request.param)


def prompt(n, seed=0):
    return torch.randint(0, VOCAB, (1, n), generator=torch.Generator().manual_seed(seed))


CFG = LoFiConfig(budget=0.2, n_sink=8, n_obs=8)     # small constants for 400-token prompts


def test_rope_base_is_read_from_the_config(model):
    assert rope_base(model) in (10000.0, 500000.0, 1000000.0)


def test_importance_is_the_attention_of_the_last_positions(model):
    ids = prompt(200)
    pre = prefill(model, ids, n_obs=8)
    full = model(ids, output_attentions=True, use_cache=True)
    assert len(pre.importance) == 3
    for imp, attn, (keys, _), (ref_keys, _) in zip(
        pre.importance, full.attentions, cache_layers(pre.cache), cache_layers(full.past_key_values)
    ):
        expected = attn[0, :, -8:, :].sum(dim=1).view(2, 2, 200).mean(dim=1)     # two query heads per KV head
        assert imp.shape == (2, 200)
        assert torch.allclose(imp, expected, atol=1e-5)
        assert torch.allclose(imp.sum(dim=-1), torch.full((2,), 8.0), atol=1e-4)  # 8 observation rows
        assert torch.allclose(keys, ref_keys, atol=1e-5)                          # two-step prefill = one-step prefill
    assert torch.allclose(pre.logits, full.logits[:, -1], atol=1e-4)


def test_cached_keys_are_rotated_keys(model):
    """The cache holds keys after RoPE: de-rotating layer 0 gives the projection of the embeddings."""
    ids = prompt(300)
    pre = prefill(model, ids, with_importance=False)
    keys = cache_layers(pre.cache)[0][0][0]                                       # (H_kv, n, d)
    layer = model.model.layers[0]
    hidden = layer.input_layernorm(model.model.embed_tokens(ids))
    plain = layer.self_attn.k_proj(hidden).view(300, 2, 16).transpose(0, 1)
    assert torch.allclose(derotate(keys, torch.arange(300), rope_inv_freq(model)), plain, atol=1e-4)
    assert not torch.allclose(keys, plain, atol=1e-2)


def test_compressed_cache_holds_the_selected_tokens(model):
    ids = prompt(400)
    pre = prefill(model, ids, n_obs=CFG.n_obs)
    cache, stats = compress(model, pre, CFG, keep_indices=True)
    lay = layout(400, CFG, rope_base(model))
    assert lay.budget == 80 and lay.window == min(int(math.sqrt(rope_base(model))), 36)
    assert lay.n_sink + lay.n_keep + lay.window == 80
    low, _ = low_high_channels(rope_base(model), 16)
    for idx, ((k, v), (k0, v0)) in enumerate(zip(cache_layers(cache), cache_layers(pre.cache))):
        assert k.shape == (1, 2, 80, 16) and v.shape == (1, 2, 80, 16)            # exactly B per layer and head
        kept = stats.kept[idx]
        for h in range(2):
            positions = kept[h].tolist()
            assert positions == sorted(set(positions))
            assert positions[:8] == list(range(8))                                # sinks
            assert positions[80 - lay.window:] == list(range(400 - lay.window, 400))   # recent window
            assert torch.equal(k[0, h], k0[0, h, kept[h]]) and torch.equal(v[0, h], v0[0, h, kept[h]])
            # same kept set as Algorithm 1 run line by line on this head
            lo, hi = 8, 400 - lay.window
            ref, passes = select_one_head(k0[0, h, lo:hi], pre.importance[idx][h, lo:hi].tolist(), lay.n_keep, low)
            assert positions[8:80 - lay.window] == [lo + j for j in ref]
            assert int(stats.passes[idx, h]) == passes
    assert stats.fallback_removed.sum() == 0
    summary = stats.summary()
    assert summary["budget"] == 80 and summary["kept_from_distant"] == lay.n_keep


def test_heads_keep_different_tokens(model):
    pre = prefill(model, prompt(400), n_obs=CFG.n_obs)
    _, stats = compress(model, pre, CFG, keep_indices=True)
    assert any(not torch.equal(kept[0], kept[1]) for kept in stats.kept)


def test_full_cache_path_is_plain_greedy_decoding(model):
    ids = prompt(120, seed=1)
    expected, seq = [], ids
    for _ in range(12):                                   # greedy decoding without any cache
        nxt = model(seq).logits[:, -1].argmax(dim=-1)
        expected.append(int(nxt))
        seq = torch.cat([seq, nxt.view(1, 1)], dim=1)
    tokens, stats = generate(model, ids, max_new_tokens=12, compress_cache=False)
    assert stats is None and tokens == expected
    # context and query prefilled in two steps on the full cache give the same output
    tokens, _ = generate(model, ids[:, :90], ids[:, 90:], max_new_tokens=12, compress_cache=False)
    assert tokens == expected


def test_positions_continue_after_compression(model):
    """A token decoded on the compressed cache is at its true position, not at the cache length."""
    ids = prompt(400, seed=2)
    pre = prefill(model, ids, n_obs=CFG.n_obs)
    cache, _ = compress(model, pre, CFG)
    token = torch.tensor([[5]])
    out = _forward(model, token, cache, 400)
    new_key = cache_layers(out.past_key_values)[0][0][0, :, -1]                   # layer 0 depends on the token only
    full = prefill(model, torch.cat([ids, token], dim=1), with_importance=False)
    assert torch.allclose(new_key, cache_layers(full.cache)[0][0][0, :, -1], atol=1e-5)
    assert cache_layers(out.past_key_values)[0][0].shape[-2] == 81


def test_decoding_on_an_evicted_cache_is_masked_attention(model):
    """Independent check of the decoding path.

    If every head drops the same tokens, decoding on the shortened cache must
    give the logits of the full cache with these tokens masked out.
    """
    from lofi.compress import build_cache

    ids, query = prompt(300, seed=10), prompt(6, seed=11)
    keep = torch.randperm(300, generator=torch.Generator().manual_seed(12))[:60].sort().values
    pre = prefill(model, ids, with_importance=False)
    short = build_cache([(k[:, :, keep], v[:, :, keep]) for k, v in cache_layers(pre.cache)], pre.cache)
    logits_short = _forward(model, query, short, 300).logits

    mask = torch.zeros(1, 306, dtype=torch.long)
    mask[0, keep] = 1
    mask[0, 300:] = 1
    logits_masked = model(input_ids=query, past_key_values=pre.cache, attention_mask=mask, use_cache=True).logits
    assert torch.allclose(logits_short, logits_masked, atol=1e-4)
    full = prefill(model, torch.cat([ids, query], dim=1), with_importance=False)
    assert not torch.allclose(logits_short[:, -1], full.logits, atol=1e-3)       # and it differs from the full cache


@pytest.mark.parametrize("operator", ["evict", "merge"])
def test_query_aware_and_query_agnostic_generation(model, operator):
    cfg = LoFiConfig(budget=0.2, n_sink=8, n_obs=8, operator=operator)
    ids = prompt(420, seed=3)
    aware, stats_aware = generate(model, ids, cfg=cfg, max_new_tokens=6)
    agnostic, stats_agnostic = generate(model, ids[:, :400], ids[:, 400:], cfg=cfg, max_new_tokens=6)
    assert len(aware) == 6 and len(agnostic) == 6
    assert stats_aware.layout.n == 420 and stats_agnostic.layout.n == 400       # the query is not part of the compressed prefill
    assert stats_agnostic.layout.budget == 80
    if operator == "merge":
        assert all(int(c.sum(dim=1)[0]) == 400 for c in stats_agnostic.counts)


def test_query_agnostic_compression_does_not_depend_on_the_query(model):
    context = prompt(400, seed=4)
    queries = [prompt(40, seed=5), prompt(40, seed=6)]
    agnostic = [generate(model, context, q, cfg=CFG, max_new_tokens=2, keep_indices=True)[1] for q in queries]
    assert all(torch.equal(a, b) for a, b in zip(agnostic[0].kept, agnostic[1].kept))
    # with the query in the prefill, the tokens kept from the context change with the query
    aware = [generate(model, torch.cat([context, q], dim=1), cfg=CFG, max_new_tokens=2, keep_indices=True)[1]
             for q in queries]
    assert any(not torch.equal(a, b) for a, b in zip(aware[0].kept, aware[1].kept))


def test_merge_keeps_low_frequency_channels(model):
    cfg = LoFiConfig(budget=0.2, n_sink=8, n_obs=8, operator="merge")
    pre = prefill(model, prompt(400, seed=7), n_obs=8)
    merged, stats_m = compress(model, pre, cfg, keep_indices=True)
    evicted, stats_e = compress(model, pre, CFG, keep_indices=True)
    low, high = low_high_channels(rope_base(model), 16)
    for (km, vm), (ke, ve), a, b in zip(cache_layers(merged), cache_layers(evicted), stats_m.kept, stats_e.kept):
        assert torch.equal(a, b)                                                  # same positions
        assert torch.equal(km[..., low], ke[..., low])                            # same low-frequency channels
        assert torch.allclose(km[..., high].norm(dim=-1), ke[..., high].norm(dim=-1), rtol=1e-4, atol=1e-5)
        assert not torch.equal(vm, ve)


@pytest.mark.parametrize("kwargs", [
    dict(criterion="full"), dict(criterion="derotated"), dict(criterion="random"),
    dict(use_importance=False), dict(criterion="low", first_criterion="full"), dict(window=8),
])
def test_controls(model, kwargs):
    cfg = LoFiConfig(budget=0.2, n_sink=8, n_obs=8, **kwargs)
    tokens, stats = generate(model, prompt(400, seed=8), cfg=cfg, max_new_tokens=3)
    assert len(tokens) == 3 and stats.layout.budget == 80
    if "window" in kwargs:
        assert stats.layout.window == 8 and stats.layout.n_keep == 64


def test_eos_stops_decoding(model):
    ids = prompt(120, seed=9)
    tokens, _ = generate(model, ids, max_new_tokens=8, compress_cache=False)
    stopped, _ = generate(model, ids, max_new_tokens=8, compress_cache=False, eos_token_id=[tokens[2]])
    assert stopped == tokens[: tokens.index(tokens[2]) + 1]


def test_short_prompts_are_rejected(model):
    with pytest.raises(ValueError):
        generate(model, prompt(100), cfg=LoFiConfig())          # B = 20 < n_s = 32
    with pytest.raises(ValueError):
        prefill(model, prompt(8), n_obs=8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("operator", ["evict", "merge"])
def test_reduced_precision_cache_keeps_its_dtype(dtype, operator):
    """Scores and merges are computed in float32; the cache keeps the dtype of the model."""
    model = build("llama").to(dtype)
    ids = prompt(400, seed=13)
    try:
        pre = prefill(model, ids, n_obs=8)
    except RuntimeError as error:               # some CPU builds lack kernels for this dtype
        pytest.skip(str(error))
    cfg = LoFiConfig(budget=0.2, n_sink=8, n_obs=8, operator=operator)
    cache, stats = compress(model, pre, cfg)
    for keys, values in cache_layers(cache):
        assert keys.dtype == dtype and values.dtype == dtype and keys.shape[-2] == 80
        assert torch.isfinite(keys.float()).all() and torch.isfinite(values.float()).all()
    tokens = greedy_decode(model, cache, pre.logits, pre.length, max_new_tokens=3)
    assert len(tokens) == 3


@pytest.mark.parametrize("kind", KINDS)
def test_default_attention_implementation(kind):
    """Without attn_implementation="eager": Transformers 4.43 falls back to eager for the observation
    step, so the result is identical; versions that return no attention weights get a clear error."""
    eager, default = build(kind), build(kind, eager=False)
    default.load_state_dict(eager.state_dict())
    ids = prompt(400, seed=14)
    reference = prefill(eager, ids, n_obs=8)
    try:
        pre = prefill(default, ids, n_obs=8)
    except RuntimeError as error:
        assert "attn_implementation" in str(error)
        return
    for a, b in zip(pre.importance, reference.importance):
        assert torch.allclose(a, b, atol=1e-5)
    _, stats_default = compress(default, pre, CFG, keep_indices=True)
    _, stats_eager = compress(eager, reference, CFG, keep_indices=True)
    assert all(torch.equal(a, b) for a, b in zip(stats_default.kept, stats_eager.kept))
    tokens_default, _ = generate(default, ids[:, :380], ids[:, 380:], cfg=CFG, max_new_tokens=5)
    tokens_eager, _ = generate(eager, ids[:, :380], ids[:, 380:], cfg=CFG, max_new_tokens=5)
    assert tokens_default == tokens_eager


@pytest.mark.parametrize("cap", [1, 3])
def test_pass_cap_in_the_config(model, cap):
    cfg = LoFiConfig(budget=0.2, n_sink=8, n_obs=8, max_passes=cap)
    pre = prefill(model, prompt(400, seed=15), n_obs=8)
    cache, stats = compress(model, pre, cfg, keep_indices=True)
    _, free = compress(model, pre, CFG)
    lay = stats.layout
    low, _ = low_high_channels(rope_base(model), 16)
    assert (stats.passes <= cap).all()
    assert (stats.pair_removed + stats.fallback_removed == lay.n_remove).all()
    summary = stats.summary()
    binds = bool((free.passes > cap).any())
    assert (summary["pair_share"] < 1.0) == binds and free.summary()["pair_share"] == 1.0
    for idx, (k, _) in enumerate(cache_layers(cache)):
        assert k.shape[-2] == lay.budget                                         # exactly B with the fallback
        k0 = cache_layers(pre.cache)[idx][0]
        lo, hi = 8, 400 - lay.window
        for h in range(2):
            ref, _ = select_one_head(k0[0, h, lo:hi], pre.importance[idx][h, lo:hi].tolist(), lay.n_keep, low,
                                     max_passes=cap)
            assert stats.kept[idx][h, 8:lay.budget - lay.window].tolist() == [lo + j for j in ref]

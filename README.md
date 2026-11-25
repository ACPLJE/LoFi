# LoFi

Reference implementation of **LoFi: Low-Frequency Key Similarity for
Query-Agnostic KV Cache Eviction**.

LoFi compresses the KV cache of a RoPE-based language model once, after
prefill, independently for each layer and KV head. It keeps the first tokens
and a recent window, and in between it repeatedly removes the less important
token of the adjacent pairs whose keys are most similar on the low-frequency
RoPE channels. The channel set and the window follow from the RoPE base of the
model, so there is no calibration data and no search per model, task, or
budget.

This repository contains the method: the constants derived from RoPE, the
selection rule (Algorithm 1), the two operators (LoFi-Sel and LoFi-M), the
controls of the selection-rule comparison, and the prefill, compression and
decoding path on Hugging Face models for the query-aware and the
query-agnostic setting. The benchmark harness and the baselines are not part
of it.

## Paper to code

| Paper | Code |
|---|---|
| Definition 1, low-frequency set `L = {i : π / (2θ_i) > √base}` | `lofi/frequency.py: low_frequency_pairs`, `low_high_channels` |
| Eq. (4), window `W = min(⌊√base⌋, ⌊(B − n_s)/2⌋)` | `lofi/frequency.py: recent_window`, `lofi/compress.py: layout` |
| Importance, attention mass from the last `n_o` prefilled positions | `lofi/compress.py: prefill` |
| Eq. (5), cosine of adjacent keys on `L` | `lofi/selection.py: pair_scores` |
| Algorithm 1, all KV heads of a layer at once | `lofi/selection.py: select` |
| Algorithm 1, one head, line by line | `lofi/reference.py: select_one_head` |
| Eq. (10), content-preserving merge (LoFi-M) | `lofi/selection.py: _merge` |
| Query-aware and query-agnostic compression (Section 4.1) | `lofi/compress.py: generate` |
| Controls: random pairing, full-channel, de-rotated, no importance | `LoFiConfig(criterion=..., use_importance=...)` |

`lofi/reference.py` follows the pseudocode with Python loops and is meant to be
read next to the paper. `lofi/selection.py` is the vectorised version; the
tests check that both keep the same tokens in the same number of passes.

## Install

```bash
pip install -r requirements.txt
```

The paper uses Hugging Face Transformers 4.43.3, which `requirements.txt`
pins.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lofi import LoFiConfig, generate, disable_sliding_window

name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
disable_sliding_window(model)          # full attention, as in the paper (matters for Mistral-7B-v0.1)

context_ids = tokenizer(context, return_tensors="pt").input_ids.cuda()
query_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

# query-agnostic: compress the context, then prefill the query on the compressed cache
tokens, stats = generate(model, context_ids, query_ids, cfg=LoFiConfig(budget=0.2), max_new_tokens=64)

# query-aware: prefill context and query together, then compress
prompt_ids = torch.cat([context_ids, query_ids], dim=1)
tokens, stats = generate(model, prompt_ids, cfg=LoFiConfig(budget=0.2), max_new_tokens=64)

print(tokenizer.decode(tokens), stats.summary())
```

`stats.summary()` reports the budget `B`, the window `W` and its share of the
budget, the share of the distant region that is kept, the share of the
removals made by the pair rule, and the number of passes. The three steps are also available separately (`prefill`, `compress`,
`greedy_decode`), and `keep_indices=True` returns the kept positions of every
layer and head in `stats.kept`.

`scripts/compression_stats.py` prints these quantities over a set of prompts,
for one or several values of `max_passes`:

```bash
python scripts/compression_stats.py --model Qwen/Qwen2-7B-Instruct \
    --data contexts.jsonl --field context --max-tokens 7500 --max-passes none 8 10
```

A command-line example:

```bash
python examples/demo.py --model Qwen/Qwen2-7B-Instruct --context-file doc.txt \
    --question "Who signed the contract?" --setting agnostic --budget 0.2 --compare-full
```

## Settings

`LoFiConfig` holds the constants of the paper as defaults.

| Field | Default | Meaning |
|---|---|---|
| `budget` | `0.2` | share of the prefilled tokens kept per layer and KV head, sinks and window included |
| `n_sink` | `32` | sink tokens `n_s` |
| `n_obs` | `32` | observation positions `n_o` for the importance |
| `protect` | `0.1` | protected fraction `p` of the distant budget |
| `operator` | `"evict"` | `"evict"` is LoFi-Sel, `"merge"` is LoFi-M |
| `criterion` | `"low"` | `"low"` is LoFi; `"full"`, `"derotated"`, `"random"` are the controls |
| `first_criterion` | `None` | criterion of the first pass only (mixed-order controls) |
| `use_importance` | `True` | `False` is the control without importance |
| `max_passes` | `None` | cap on the passes of the pair rule; `None` is no cap |
| `window` | `None` | `None` applies Eq. (4); an integer fixes the window |
| `rope_base` | `None` | `None` reads the base from the model config |

The constants of Table 1 are computed from the model config and are not stored
anywhere:

| RoPE base | head dimension | size of `L` | `⌊√base⌋` |
|---|---|---|---|
| 10,000 | 128 | 35 | 100 |
| 500,000 | 128 | 34 | 707 |
| 1,000,000 | 128 | 34 | 1,000 |

For a model with rescaled RoPE (Llama-3.1), `L` and `W` use the base before
rescaling, and the de-rotated control uses the frequencies the model applies.

## Implementation notes

Points that the pseudocode leaves open, and what this code does.

- **Budget.** `B = ⌊budget · n⌋`, where `n` is the number of tokens prefilled
  before compression. Every layer and KV head ends with exactly `B` tokens. In
  the query-agnostic setting `n` is the length of the context, and the query
  tokens prefilled afterwards are all kept.
- **Local maxima and ties.** A pair is a candidate if its score is strictly
  above the pair on its left and at least the pair on its right, so two
  candidates never share a token. Candidates with equal scores are taken from
  left to right. Equal importances are resolved in favour of the earlier
  position, both for the protected set and for the choice of the surviving
  token.
- **Passes and fallback (lines 4 and 13).** Without a cap, the loop of line 4
  always reaches the budget: the protected set holds a tenth of the distant
  budget, so an adjacent pair outside it exists whenever tokens remain to be
  removed, and the fallback of line 13 is not executed.
  `LoFiConfig(max_passes=k)` caps the loop at `k` passes; the tokens that
  remain to be removed are then evicted by the fallback, lowest importance
  first and never from the protected set. `stats.summary()["pair_share"]`
  reports the share of the removals made by the pair rule.
- **Without importance.** Nothing is protected and the earlier token of a pair
  is kept.
- **Merge.** LoFi-M keeps the same positions as LoFi-Sel, because the merge
  does not change the low-frequency channels on which pairs are scored. The
  number of tokens merged into each kept position is returned in
  `stats.counts`. The mass correction of Eq. (11) is not applied.
- **Precision.** Scores and merges are computed in float32 and the cache keeps
  the dtype of the model.
- **Positions.** After compression the cache is shorter than the number of
  tokens seen, so decoding passes the true positions explicitly. Qwen2 in
  Transformers 4.43 slices its RoPE table to the cache length; the table
  lookup is extended at run time, without changing its values
  (`lofi/compress.py: _legacy_rope_modules`).
- **Attention weights.** The importance reads the attention of the last `n_o`
  positions with `output_attentions=True`. To avoid materialising the full
  attention matrix, the prompt is prefilled in two steps: all but the last
  `n_o` tokens, then the last `n_o` tokens. Transformers 4.43 switches to the
  eager attention for the second step by itself. Later versions need
  `attn_implementation="eager"` at load time.
- **Short prompts.** If `B ≤ n_s + 1` the selection is not defined and
  `compress` raises an error.
- **Batch size.** One prompt at a time.

## Tests

```bash
python -m pytest tests -q
```

The tests run on CPU in under a minute and download nothing. They use small,
randomly initialised Llama, Mistral and Qwen2 models, and a Llama model with
rescaled RoPE.

- `tests/test_frequency.py`: the constants of Table 1, Definition 1 (at
  distance `√base` the pairs of `L` have turned less than a quarter and all
  others at least a quarter), Eq. (4) and the prompt length below which its
  cap binds, de-rotation.
- `tests/test_selection.py`: exactly `B − n_s − W` tokens per head, protected
  tokens never removed, agreement of the vectorised selection with the
  line-by-line one including tied scores and importances, independence of the
  heads, the pass cap and the fallback against the line-by-line version,
  Eq. (10) on a single pair, low-frequency channels bit-identical and
  high-frequency norm preserved after merging, the controls.
- `tests/test_hf.py`: importance equal to the attention of a one-step prefill,
  cached keys equal to rotated projections, kept sets equal to Algorithm 1 run
  head by head, decoding on a shortened cache equal to masked attention on the
  full cache, full-cache path equal to greedy decoding without a cache,
  query-agnostic compression independent of the query, identical results
  with the default and the eager attention implementation, fp16 and bf16
  caches.

The suite passes on Transformers 4.43.3 and 5.24.1 (the latter with
`attn_implementation="eager"`).

## License

Apache License 2.0, see `LICENSE`.

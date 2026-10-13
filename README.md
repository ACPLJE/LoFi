# LoFi

Calibration-free KV cache compression via low-frequency RoPE channels.

RoPE rotates each channel pair at its own frequency, so over a token distance
`Δ` a pair with frequency `θ_i` turns by `Δ·θ_i`. Pairs slow enough to stay
inside a quarter turn carry **content** that barely moves with position; fast
pairs carry **position**. LoFi puts the boundary at `Δ ≈ √base`, which follows
from the model's RoPE base alone — no calibration data, no profiling pass, no
per-model tuning.

That split is used twice:

- **LoFi-Sel.** Candidate token pairs are scored by cosine similarity
  restricted to the low-frequency channels. Because those channels are
  position-invariant, the score reflects content rather than proximity, so it
  stays meaningful across merge passes and at token gaps greater than one.
- **Mass-corrected content-preserving merge.** When two tokens merge, the
  surviving key keeps its low-frequency block exactly and blends only its
  high-frequency block, with the norm restored on that block alone. Values are
  combined by a count-weighted average. Each slot tracks how many original
  tokens it stands for, and `log c` is added to that key's attention logits at
  decode time — since `exp(q·k + log c) = c·exp(q·k)`, a merged slot attracts
  the attention mass of all its constituents instead of just one.

Compression runs once, right after prefill. Decoding then proceeds over the
smaller cache.

## Install

```bash
git clone <this-repo> && cd lofi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.12, PyTorch 2.7.1 (CUDA 11.8), Transformers 4.57.6, single
GPU, fp16, batch size 1. The vendored attention in `third_party/` depends on
tuple-style `past_key_values`, so the pinned Transformers version matters.

Run the test suite — it needs no GPU and no model download:

```bash
pytest
```

## Quick start

Compress a cache directly:

```python
import torch
from lofi import LoFiConfig, compress_kv_cache

config = LoFiConfig(hh_ratio=0.1, recent_ratio=0.1)   # keep 20%
result = compress_kv_cache(
    keys, values,                 # (1, H, n, head_dim), post-RoPE
    rope_base=500000.0,
    config=config,
    attn_weights=attn,            # (1, H, n, n) prefill attention
)
result.keys, result.values        # (1, H, kept, head_dim)
result.merge_counts               # (1, kept) multiplicities
result.stats.retention            # measured, not requested
```

Or load a patched model:

```python
from transformers import AutoConfig
from lofi import LoFiConfig
from lofi.models.llama import LoFiLlamaForCausalLM, reset_caches

hf_config = AutoConfig.from_pretrained(model_path)
hf_config.lofi_config = LoFiConfig(hh_ratio=0.1, recent_ratio=0.1)
model = LoFiLlamaForCausalLM.from_pretrained(
    model_path, config=hf_config, torch_dtype=torch.float16,
).cuda().eval()

reset_caches(model)   # call once per example; state is per-sequence
```

Three-task smoke test, about ten minutes on one GPU:

```bash
python scripts/run_longbench.py \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --out-dir results/smoke --limit 20 \
  --tasks gov_report hotpotqa passage_retrieval_en
python -m eval.score_longbench results/smoke
```

## Check the retention before trusting any accuracy number

The configured ratio is a **target, not a guarantee**. A single pass over
non-overlapping adjacent pairs can remove at most half of the compressible
region, so a budget below ~50% retention is only reached if enough passes run
and the eviction fallback fires. Measure it:

```bash
python scripts/audit_retention.py \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --hh-ratio 0.1 --recent-ratio 0.1 \
  --ctx-lens 4096 8192
```

The script reports measured per-layer cache length and **exits non-zero** if it
misses the target by more than 5 points. An accuracy number paired with a
compression ratio that was never verified is not comparable to a baseline at
the same nominal ratio. `scripts/run_ablations.sh` runs this audit first for
exactly this reason.

## Reproducing the tables

Every setting below shares the same prompts, truncation, decoding and
compression budget; only the component under test changes, so the differences
are attributable.

| Preset | What it changes | Isolates |
|---|---|---|
| `main` | — (the method) | — |
| `no_mass` | mass correction off, adaptive value blend | the `log c` correction |
| `select_evict` | source evicted instead of merged | selection vs. merge operator |
| `uniform_merge` | naive 50:50 merge over all channels | the merge operator |
| `full_sim` | candidates scored on all channels | the selection criterion |
| `full_cache` | no compression | the reference point |

Full grid across all 16 LongBench tasks:

```bash
MODEL=meta-llama/Meta-Llama-3-8B-Instruct bash scripts/run_ablations.sh
```

One setting at a time:

```bash
python scripts/run_longbench.py \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --out-dir results/main_r0.2 \
  --preset main --hh-ratio 0.1 --recent-ratio 0.1
python -m eval.score_longbench results/main_r0.2
```

Sweep the budget:

```bash
for r in 0.05 0.1 0.15 0.2; do
  python scripts/run_longbench.py --model-path "$MODEL" \
    --out-dir "results/main_r$(python3 -c "print(2*$r)")" \
    --preset main --hh-ratio "$r" --recent-ratio "$r"
done
```

Efficiency (TTFT, time per output token, peak memory, measured retention):

```bash
python scripts/benchmark_efficiency.py \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --presets full_cache main select_evict \
  --context-len 7500 --gen-lens 128 512 2048
```

`eval.score_longbench` prints per-task scores and the macro average and writes
`scores.json`. This repository ships **no precomputed scores**: generate them
with the commands above and report what you measure.

## Layout

```
lofi/
  frequency.py      √base boundary → low / high channel index sets
  core.py           selection, merge, mass correction
  config.py         every knob, plus the ablation presets
  models/llama.py   LLaMA-3 attention patch
eval/               official LongBench metrics and scorer
scripts/            LongBench runner, retention audit, efficiency benchmark
configs/            LongBench prompt templates and generation lengths
third_party/        vendored LLaMA attention (see NOTICE)
tests/              CPU-only; no GPU or model download needed
```

## Scope

- **LLaMA-3 only.** The attention patch covers the LLaMA-3 family. The core in
  `lofi/core.py` is model-agnostic (it takes tensors in and returns tensors
  out), so another architecture needs only a new patch in `lofi/models/`.
- **Batch size 1**, matching the evaluation harness. `compress_kv_cache`
  raises on larger batches rather than silently mis-slicing.
- **Eager attention**, because the prefill attention probabilities are the
  token importance signal.
- RoPE scaling is disabled at load time; the frequency partition assumes the
  model's plain base frequency.
- The softmax accumulates in fp16 to match the pipeline this method was
  developed in. Widening it changes results slightly.

## Third-party code

The vendored LLaMA attention, the LongBench metrics and scorer, and the prompt
templates come from other projects and keep their original license headers.
See [`NOTICE`](NOTICE) for the full attribution list.

This project is licensed under Apache-2.0 ([`LICENSE`](LICENSE)), which is
compatible with the Transformers-derived code it vendors.

## Citation

```bibtex
@article{lofi,
  title   = {LoFi: calibration-free KV cache compression via low-frequency RoPE channels},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

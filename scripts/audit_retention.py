#!/usr/bin/env python
"""Measure the retention a LoFi setting actually achieves.

The configured ratio is a target, not a guarantee: a single pass over
non-overlapping adjacent pairs can remove at most half of the compressible
region, so a setting that asks for less than ~50% retention only reaches it if
enough passes run and the eviction fallback fires. This script reports the
measured per-layer cache length for one prefill forward per sample, and exits
non-zero when the measurement misses the target.

Run this before quoting any compression ratio alongside an accuracy number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lofi import config as lofi_config  # noqa: E402
from lofi.models import llama as lofi_llama  # noqa: E402

FILLER = (
    "The rotary position embedding assigns each channel pair a distinct "
    "rotation frequency, so the key vector of every token is modulated "
    "by its absolute position before being cached for later attention. "
)

# How far the measured retention may drift from the target before it counts
# as a mismatch.
TOLERANCE = 0.05


def build_input_ids(tokenizer, ctx_len: int, text: str | None) -> torch.Tensor:
    """Tile `text` (or synthetic filler) to exactly `ctx_len` tokens."""
    ids = tokenizer(text or FILLER * 400, return_tensors="pt").input_ids[0]
    while ids.numel() < ctx_len:
        ids = torch.cat([ids, ids], dim=0)
    return ids[:ctx_len].unsqueeze(0)


def load_texts(jsonl: Path | None, n_samples: int) -> list[str | None]:
    """Read up to `n_samples` contexts from a LongBench-style JSONL."""
    if jsonl is None:
        return [None] * n_samples
    texts: list[str | None] = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append((record.get("context", "") + "\n"
                          + record.get("input", "")).strip())
            if len(texts) >= n_samples:
                break
    return texts or [None] * n_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--ctx-lens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="LongBench-style JSONL for realistic contexts "
                             "(default: synthetic filler text)")
    parser.add_argument("--out", type=Path, default=Path("retention_audit.json"))
    lofi_config.add_arguments(parser)
    args = parser.parse_args()

    lofi = lofi_config.from_args(args)
    target = lofi.target_retention

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, cache_dir=args.cache_dir
    )
    config = _prepare_config(args.model_path, lofi, args.cache_dir)
    model = lofi_llama.LoFiLlamaForCausalLM.from_pretrained(
        args.model_path, config=config, cache_dir=args.cache_dir,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).cuda().eval()

    texts = load_texts(args.jsonl, args.n_samples)
    results: dict[int, dict] = {}

    for ctx in args.ctx_lens:
        per_sample = []
        for text in texts:
            lofi_llama.reset_caches(model)
            ids = build_input_ids(tokenizer, ctx, text).cuda()
            with torch.no_grad():
                model(input_ids=ids, use_cache=True)

            stats = [m.stats for m in model.modules()
                     if isinstance(m, lofi_llama.LoFiLlamaAttention) and m.stats is not None]
            if not stats:
                print(f"[warn] compression never fired at ctx={ctx} "
                      f"(sequence already within budget?)")
                continue
            kept = [s.kept for s in stats]
            per_sample.append({
                "in_len": stats[0].seq_len,
                "kept_mean": sum(kept) / len(kept),
                "kept_min": min(kept),
                "kept_max": max(kept),
                "n_layers_fired": len(stats),
                "retention": sum(kept) / (len(kept) * stats[0].seq_len),
                "passes_max": max(s.passes for s in stats),
                "fallback_evicted_total": sum(s.fallback_evicted for s in stats),
            })
            del ids
            torch.cuda.empty_cache()

        if per_sample:
            mean = sum(s["retention"] for s in per_sample) / len(per_sample)
            results[ctx] = {"samples": per_sample, "mean_retention": mean}
            verdict = "ok" if abs(mean - target) <= TOLERANCE else "MISMATCH"
            print(f"[ctx={ctx}] target={target:.2f} measured={mean:.4f}  {verdict}")

    args.out.write_text(json.dumps({
        "model_path": args.model_path,
        "lofi": lofi.to_dict(),
        "target_retention": target,
        "tolerance": TOLERANCE,
        "results": {str(k): v for k, v in results.items()},
    }, indent=2))
    print(f"wrote {args.out}")

    bad = [ctx for ctx, r in results.items()
           if abs(r["mean_retention"] - target) > TOLERANCE]
    if bad:
        print(f"\nmeasured retention misses the target by more than "
              f"{TOLERANCE:.0%} at ctx={bad}. Accuracy numbers taken at this "
              f"setting are not at the compression ratio they claim.")
        raise SystemExit(1)


def _prepare_config(model_path: str, lofi: lofi_config.LoFiConfig, cache_dir):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
    if getattr(config, "rope_scaling", None) is not None:
        config.rope_scaling = None
    for attr, default in (("rope_theta", 500000.0), ("rope_scaling", None),
                          ("pretraining_tp", 1), ("attention_dropout", 0.0)):
        if not hasattr(config, attr):
            setattr(config, attr, default)
    config.lofi_config = lofi
    return config


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Measure prefill, decode and memory cost of LoFi settings.

Reports, per (preset, generation length):

  prefill_s     prefill forward including the one-time compression
  ttft_s        time to first token (prefill + first decode step)
  tpot_ms       mean time per output token, first step excluded
  decode_tok_s  1 / tpot
  peak_*_gb     torch peak allocation after prefill and over the whole run
  retention     measured cache length after prefill, over prompt length

All presets share one model load and one prompt, so the deltas isolate the
compression setting rather than load order or input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lofi import config as lofi_config  # noqa: E402
from lofi.models import llama as lofi_llama  # noqa: E402

FILLER = (
    "Long-context inference stresses the key-value cache because every "
    "generated token attends to all previously cached positions, and the "
    "memory footprint grows linearly with the prompt length. "
)


def build_input_ids(tokenizer, ctx_len: int) -> torch.Tensor:
    ids = tokenizer(FILLER * 400, return_tensors="pt").input_ids[0]
    while ids.numel() < ctx_len:
        ids = torch.cat([ids, ids], dim=0)
    return ids[:ctx_len].unsqueeze(0)


def _cache_len(past) -> int:
    try:
        return int(past[0][0].shape[-2])
    except (TypeError, IndexError, AttributeError):
        return -1


@torch.no_grad()
def measure(model, input_ids: torch.Tensor, gen_len: int) -> dict[str, Any]:
    """Time one prefill plus `gen_len` greedy decode steps."""
    lofi_llama.reset_caches(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - start
    peak_prefill = torch.cuda.max_memory_allocated() / 1024**3

    past = out.past_key_values
    cache_len = _cache_len(past)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    step_times: list[float] = []
    for _ in range(gen_len):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    peak_total = torch.cuda.max_memory_allocated() / 1024**3
    # The first decode step is excluded from TPOT: it is charged to TTFT.
    tail = step_times[1:] or step_times
    tpot = sum(tail) / len(tail)

    result = {
        "prefill_s": prefill_s,
        "ttft_s": prefill_s + step_times[0],
        "tpot_ms": 1000 * tpot,
        "decode_tok_s": 1.0 / tpot,
        "peak_prefill_gb": peak_prefill,
        "peak_total_gb": peak_total,
        "cache_len_after_prefill": cache_len,
        "retention": cache_len / input_ids.shape[-1] if cache_len > 0 else None,
    }
    del past, out
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--presets", nargs="+", default=["full_cache", "main"],
                        choices=sorted(lofi_config.PRESETS),
                        help="settings to compare, measured in one process")
    parser.add_argument("--context-len", type=int, default=7500)
    parser.add_argument("--gen-lens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--hh-ratio", type=float, default=0.1)
    parser.add_argument("--recent-ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed runs before measuring, to settle kernels")
    parser.add_argument("--out", type=Path, default=Path("efficiency.json"))
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, cache_dir=args.cache_dir
    )

    config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    if getattr(config, "rope_scaling", None) is not None:
        config.rope_scaling = None
    for attr, default in (("rope_theta", 500000.0), ("rope_scaling", None),
                          ("pretraining_tp", 1), ("attention_dropout", 0.0)):
        if not hasattr(config, attr):
            setattr(config, attr, default)
    config.lofi_config = lofi_config.LoFiConfig(
        hh_ratio=args.hh_ratio, recent_ratio=args.recent_ratio
    )

    model = lofi_llama.LoFiLlamaForCausalLM.from_pretrained(
        args.model_path, config=config, cache_dir=args.cache_dir,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).cuda().eval()

    input_ids = build_input_ids(tokenizer, args.context_len).cuda()
    rows: list[dict[str, Any]] = []

    for preset in args.presets:
        settings = lofi_config.LoFiConfig(
            hh_ratio=args.hh_ratio, recent_ratio=args.recent_ratio,
            **lofi_config.PRESETS[preset],
        )
        # Swapping the config in place keeps one set of weights for all presets.
        for module in model.modules():
            if isinstance(module, lofi_llama.LoFiLlamaAttention):
                module.lofi_config = settings

        for gen_len in args.gen_lens:
            for _ in range(args.warmup):
                measure(model, input_ids, min(gen_len, 8))
            row = {"preset": preset, "context_len": args.context_len,
                   "gen_len": gen_len, **measure(model, input_ids, gen_len)}
            rows.append(row)
            print(f"{preset:12s} gen={gen_len:5d}  "
                  f"ttft={row['ttft_s']:6.3f}s  "
                  f"tpot={row['tpot_ms']:6.2f}ms  "
                  f"{row['decode_tok_s']:6.1f} tok/s  "
                  f"peak={row['peak_total_gb']:5.2f}GiB  "
                  f"retention={row['retention']}")

    args.out.write_text(json.dumps({
        "model_path": args.model_path,
        "context_len": args.context_len,
        "hh_ratio": args.hh_ratio,
        "recent_ratio": args.recent_ratio,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

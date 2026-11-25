"""Window, kept share, pair share and number of passes of LoFi on a set of prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lofi import LoFiConfig, compress, disable_sliding_window, layout, prefill, rope_base


def read_texts(args) -> list:
    texts = []
    for path in args.text_file or []:
        with open(path, encoding="utf-8") as handle:
            texts.append(handle.read())
    if args.data:
        with open(args.data, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    texts.append(json.loads(line)[args.field])
    if not texts:
        raise SystemExit("no prompts: pass --data file.jsonl or --text-file file.txt")
    return texts[: args.limit] if args.limit else texts


def parse_cap(value: str):
    return None if value.lower() in ("none", "inf", "0") else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", help="JSON-lines file with one prompt per line")
    parser.add_argument("--field", default="context", help="field of each JSON object that holds the text")
    parser.add_argument("--text-file", nargs="*", help="plain-text files, one prompt each")
    parser.add_argument("--limit", type=int, default=None, help="use the first N prompts")
    parser.add_argument("--max-tokens", type=int, default=7500, help="truncate longer prompts in the middle")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--max-passes", nargs="*", default=["none"], help="caps to compare; 'none' is no cap")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn", default=None, help="attn_implementation; 'eager' is needed on transformers >= 4.48")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype or ("float16" if device == "cuda" else "float32"))
    kwargs = dict(torch_dtype=dtype)
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(device).eval()
    disable_sliding_window(model)
    base = rope_base(model)

    caps = [parse_cap(v) for v in args.max_passes]
    rows = {cap: dict(n=[], window=[], share=[], kept=[], pair=0, fallback=0, passes=[]) for cap in caps}
    skipped = 0
    for text in read_texts(args):
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] > args.max_tokens:
            half = args.max_tokens // 2
            ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
        try:
            layout(ids.shape[1], LoFiConfig(budget=args.budget), base)
        except ValueError:
            skipped += 1
            continue
        pre = prefill(model, ids.to(device))
        for cap in caps:
            _, stats = compress(model, pre, LoFiConfig(budget=args.budget, max_passes=cap))
            lay, row = stats.layout, rows[cap]
            row["n"].append(lay.n)
            row["window"].append(lay.window)
            row["share"].append(lay.window / lay.budget)
            row["kept"].append(lay.n_keep / lay.n_distant)
            row["pair"] += int(stats.pair_removed.sum())
            row["fallback"] += int(stats.fallback_removed.sum())
            row["passes"].append(stats.passes.flatten())
        del pre

    def mean(values):
        return sum(values) / len(values)

    count = len(rows[caps[0]]["n"])
    if count == 0:
        raise SystemExit("every prompt was too short for this budget")
    print(f"model: {args.model} | RoPE base {base:g} | budget {args.budget:.0%} | "
          f"{count} prompts, mean length {mean(rows[caps[0]]['n']):.0f} tokens"
          + (f" | {skipped} skipped (too short)" if skipped else ""))
    print(f"{'max_passes':>10} {'W':>7} {'W/B':>6} {'kept (%)':>9} {'pair (%)':>9} {'passes mean / max':>18}")
    for cap in caps:
        row = rows[cap]
        passes = torch.cat(row["passes"]).float()
        pair = 100.0 * row["pair"] / max(row["pair"] + row["fallback"], 1)
        print(f"{'none' if cap is None else cap:>10} {mean(row['window']):>7.0f} {mean(row['share']):>6.2f} "
              f"{100 * mean(row['kept']):>9.1f} {pair:>9.1f} {passes.mean():>11.1f} / {int(passes.max())}")


if __name__ == "__main__":
    main()

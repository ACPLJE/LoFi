"""Run LoFi on one context and one question, and print what was kept and what was generated.

    python examples/demo.py --model Qwen/Qwen2-7B-Instruct \
        --context-file my_document.txt --question "Who signed the contract?" \
        --setting agnostic --budget 0.2 --compare-full

``--setting agnostic`` compresses the cache of the context before the question
is seen and then prefills the question on the compressed cache.
``--setting aware`` prefills context and question together and compresses
afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lofi import LoFiConfig, disable_sliding_window, generate  # noqa: E402

INSTRUCTION = "\n\nAnswer the question based on the text above. Give only the answer.\nQuestion: "


def split_prompt(tokenizer, context: str, question: str, use_template: bool):
    """Token ids of the part of the prompt up to the end of the context, and of the rest."""
    context = context.strip()
    tail = INSTRUCTION + question.strip()
    if use_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": context + tail}], tokenize=False, add_generation_prompt=True
        )
        cut = text.index(context) + len(context)
        head, tail, special = text[:cut], text[cut:], False      # the template adds its own special tokens
    else:
        head, tail, special = context, tail + "\nAnswer:", True
    context_ids = tokenizer(head, add_special_tokens=special, return_tensors="pt").input_ids
    query_ids = tokenizer(tail, add_special_tokens=False, return_tensors="pt").input_ids
    return context_ids, query_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--setting", choices=["agnostic", "aware"], default="agnostic")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--operator", choices=["evict", "merge"], default="evict", help="LoFi-Sel or LoFi-M")
    parser.add_argument("--criterion", choices=["low", "full", "derotated", "random"], default="low")
    parser.add_argument("--no-importance", action="store_true")
    parser.add_argument("--window", type=int, default=None, help="recent window; default: Eq. (4)")
    parser.add_argument("--max-passes", type=int, default=None, help="cap on the passes of the pair rule")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="truncate the context in the middle")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn", default=None, help="attn_implementation; 'eager' is needed on transformers >= 4.48")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--compare-full", action="store_true", help="also decode on the full cache")
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

    with open(args.context_file, encoding="utf-8") as handle:
        context = handle.read()
    if args.max_context_tokens:
        ids = tokenizer(context, add_special_tokens=False).input_ids
        if len(ids) > args.max_context_tokens:
            half = args.max_context_tokens // 2
            context = tokenizer.decode(ids[:half]) + tokenizer.decode(ids[-half:])
    context_ids, query_ids = split_prompt(tokenizer, context, args.question, not args.no_chat_template)
    context_ids, query_ids = context_ids.to(device), query_ids.to(device)

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    cfg = LoFiConfig(budget=args.budget, operator=args.operator, criterion=args.criterion,
                     use_importance=not args.no_importance, window=args.window, max_passes=args.max_passes)

    def run(compress_cache: bool):
        start = time.time()
        if args.setting == "aware":
            prompt, query = torch.cat([context_ids, query_ids], dim=1), None
        else:
            prompt, query = context_ids, query_ids
        tokens, stats = generate(model, prompt, query, cfg=cfg, max_new_tokens=args.max_new_tokens,
                                 eos_token_id=eos, compress_cache=compress_cache)
        return tokenizer.decode(tokens, skip_special_tokens=True).strip(), stats, time.time() - start

    print(f"model: {args.model} | setting: query-{args.setting} | "
          f"context: {context_ids.shape[1]} tokens | query: {query_ids.shape[1]} tokens")
    text, stats, seconds = run(True)
    print(json.dumps(stats.summary(), indent=2))
    print(f"[LoFi, {args.budget:.0%} kept, {seconds:.1f}s] {text}")
    if args.compare_full:
        text, _, seconds = run(False)
        print(f"[full cache, {seconds:.1f}s] {text}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate LongBench predictions for a LLaMA-3 model under LoFi compression.

Writes one ``<task>.jsonl`` per task plus an ``args.json`` manifest, in the
layout :mod:`eval.score_longbench` expects.

Prompt construction, middle truncation, and greedy decoding follow the
reference pipeline this method was developed in, so results are comparable
with numbers produced there.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lofi import config as lofi_config  # noqa: E402
from lofi.models.llama import LoFiLlamaForCausalLM, reset_caches  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

# LongBench's own convention: on these tasks the instruction-style wrapper
# hurts, so the raw formatted prompt is used.
NO_CHAT_TEMPLATE_TASKS = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")

# LongBench-E has no length-stratified split for these three.
NO_E_SPLIT_TASKS = ("narrativeqa", "qmsum", "musique")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_config(model_path: str, lofi: lofi_config.LoFiConfig):
    """Load the HF config and attach LoFi settings.

    RoPE scaling is disabled because the frequency partition and the cached
    rotary tables here assume the model's plain base frequency.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    if getattr(config, "rope_scaling", None) is not None:
        config.rope_scaling = None
    for attr, default in (("rope_theta", 500000.0), ("rope_scaling", None),
                          ("pretraining_tp", 1), ("attention_dropout", 0.0)):
        if not hasattr(config, attr):
            setattr(config, attr, default)
    config.lofi_config = lofi
    return config


def truncate_middle(tokenizer, prompt: str, max_length: int) -> str:
    """Keep the first and last `max_length // 2` tokens.

    Both ends carry instructions on most LongBench tasks, so the middle is the
    safe place to cut.
    """
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))


@torch.no_grad()
def generate_task(model, tokenizer, data, task: str, prompt_format: str,
                  max_gen: int, max_length: int, device,
                  use_chat_template: bool = False) -> list[dict[str, Any]]:
    """Run greedy generation over one task's examples."""
    preds: list[dict[str, Any]] = []
    times: list[float] = []
    peak_mems: list[float] = []

    for example in tqdm(data, desc=task):
        reset_caches(model)
        prompt = truncate_middle(tokenizer, prompt_format.format(**example), max_length)
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_gen, "num_beams": 1,
            "do_sample": False, "temperature": 1.0,
        }
        if task == "samsum":
            # Without this the model loops on "\nDialogue" instead of stopping.
            gen_kwargs.update(
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id,
                              tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )
        output = model.generate(**inputs, **gen_kwargs)[0]

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peak_mems.append(torch.cuda.max_memory_allocated() / (1024**3))

        preds.append({
            "pred": tokenizer.decode(output[context_length:], skip_special_tokens=True),
            "answers": example["answers"],
            "all_classes": example["all_classes"],
            "length": example["length"],
        })

    if times:
        print(f"  {task}: {len(times)} examples, "
              f"{sum(times) / len(times) * 1000:.0f} ms/example, "
              f"peak {max(peak_mems):.2f} GiB")
    return preds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True,
                        help="HF hub id or local directory of a LLaMA-3 model")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="directory to write <task>.jsonl into")
    parser.add_argument("--cache-dir", default=None,
                        help="HF cache directory (default: HF_HOME / ~/.cache)")
    parser.add_argument("--tasks", nargs="+", default=LONGBENCH_TASKS,
                        metavar="TASK", help="subset of LongBench tasks to run")
    parser.add_argument("--max-input-length", type=int, default=None,
                        help="truncation budget in tokens "
                             "(default: looked up in configs/model2maxlen.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N examples per task (smoke tests)")
    parser.add_argument("--longbench-e", action="store_true",
                        help="use the length-stratified LongBench-E splits")
    parser.add_argument("--chat-template", action="store_true",
                        help="wrap prompts in the tokenizer's chat template. Off by "
                             "default, matching the reference pipeline")
    parser.add_argument("--seed", type=int, default=42)
    lofi_config.add_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lofi = lofi_config.from_args(args)
    seed_everything(args.seed)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = Path(args.model_path).name

    max_length = args.max_input_length
    if max_length is None:
        table = json.loads((CONFIG_DIR / "model2maxlen.json").read_text())
        if model_name not in table:
            raise SystemExit(
                f"no truncation budget known for {model_name!r}; "
                f"pass --max-input-length (configs/model2maxlen.json has: "
                f"{', '.join(sorted(table))})"
            )
        max_length = table[model_name]

    dataset2prompt = json.loads((CONFIG_DIR / "dataset2prompt.json").read_text())
    dataset2maxlen = json.loads((CONFIG_DIR / "dataset2maxlen.json").read_text())

    print(f"loading {args.model_path} (fp16)")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True, cache_dir=args.cache_dir
    )
    config = prepare_config(args.model_path, lofi)
    model = LoFiLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps({
        "model_path": args.model_path,
        "model_name": model_name,
        "max_input_length": max_length,
        "longbench_e": args.longbench_e,
        "chat_template": args.chat_template,
        "limit": args.limit,
        "seed": args.seed,
        "preset": args.preset,
        "lofi": lofi.to_dict(),
    }, indent=2))

    print(f"target retention {lofi.target_retention:.2f} "
          f"(hh={lofi.hh_ratio}, recent={lofi.recent_ratio}), preset={args.preset}")

    for task in args.tasks:
        if task not in dataset2prompt:
            raise SystemExit(f"unknown LongBench task: {task}")
        split_name = task
        if args.longbench_e and task not in NO_E_SPLIT_TASKS:
            split_name = f"{task}_e"
        data = load_dataset("THUDM/LongBench", split_name, split="test")
        if args.limit is not None:
            data = data.select(range(min(args.limit, len(data))))

        use_template = args.chat_template and task not in NO_CHAT_TEMPLATE_TASKS
        preds = generate_task(model, tokenizer, data, task, dataset2prompt[task],
                              dataset2maxlen[task], max_length, device,
                              use_chat_template=use_template)
        out_path = args.out_dir / f"{task}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for record in preds:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  wrote {out_path}")

    print(f"\ndone. score with:\n  python -m eval.score_longbench {args.out_dir}")


if __name__ == "__main__":
    main()

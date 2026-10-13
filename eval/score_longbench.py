"""Score LongBench prediction files with the official metrics.

Modified from https://github.com/THUDM/LongBench (LongBench/eval.py): the
scoring functions are unchanged; input/output paths are command-line arguments
instead of being derived from a fixed ``pred/`` layout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .longbench_metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

DATASET2METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Tasks where the model tends to keep generating past the answer, so only the
# first non-empty line is scored.
_FIRST_LINE_ONLY = ("trec", "triviaqa", "samsum", "lsht")


def _example_score(dataset: str, prediction: str, ground_truths: list[str],
                   all_classes: Any) -> float:
    if dataset in _FIRST_LINE_ONLY:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return max(
        (DATASET2METRIC[dataset](prediction, gt, all_classes=all_classes)
         for gt in ground_truths),
        default=0.0,
    )


def score(dataset: str, predictions: list[str], answers: list[list[str]],
          all_classes: Any) -> float:
    """Mean score over all examples, as a percentage rounded to 2 decimals."""
    total = sum(
        _example_score(dataset, pred, gts, all_classes)
        for pred, gts in zip(predictions, answers, strict=True)
    )
    return round(100 * total / len(predictions), 2)


def score_by_length(dataset: str, predictions: list[str], answers: list[list[str]],
                    lengths: list[int], all_classes: Any) -> dict[str, float]:
    """LongBench-E scoring: the same metric, bucketed by input length."""
    if len(lengths) != len(predictions):
        raise ValueError(
            f"{dataset}: {len(predictions)} predictions but {len(lengths)} lengths; "
            "LongBench-E scoring needs a 'length' field on every record"
        )
    buckets: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": [], "total": []}
    for pred, gts, length in zip(predictions, answers, lengths, strict=True):
        value = _example_score(dataset, pred, gts, all_classes)
        if length < 4000:
            buckets["0-4k"].append(value)
        elif length < 8000:
            buckets["4-8k"].append(value)
        else:
            buckets["8k+"].append(value)
        buckets["total"].append(value)
    return {k: round(100 * float(np.mean(v)), 2) if v else 0.0 for k, v in buckets.items()}


def score_directory(pred_dir: Path, longbench_e: bool = False) -> dict[str, Any]:
    """Score every ``<task>.jsonl`` in `pred_dir`, keyed by task name."""
    scores: dict[str, Any] = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        dataset = path.stem
        if dataset not in DATASET2METRIC:
            continue
        predictions: list[str] = []
        answers: list[list[str]] = []
        lengths: list[int] = []
        all_classes: Any = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                predictions.append(record["pred"])
                answers.append(record["answers"])
                all_classes = record["all_classes"]
                if "length" in record:
                    lengths.append(record["length"])
        if not predictions:
            continue
        if longbench_e:
            scores[dataset] = score_by_length(dataset, predictions, answers,
                                              lengths, all_classes)
        else:
            scores[dataset] = score(dataset, predictions, answers, all_classes)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path,
                       help="directory of <task>.jsonl prediction files")
    parser.add_argument("--longbench-e", action="store_true",
                       help="score LongBench-E, bucketing by input length")
    parser.add_argument("--out", type=Path, default=None,
                       help="output path (default: <pred_dir>/scores.json)")
    args = parser.parse_args()

    scores = score_directory(args.pred_dir, args.longbench_e)
    if not scores:
        raise SystemExit(f"no scorable .jsonl files in {args.pred_dir}")

    out = args.out or args.pred_dir / "scores.json"
    out.write_text(json.dumps(scores, ensure_ascii=False, indent=2))

    for task, value in scores.items():
        print(f"{task:24s} {value}")
    if not args.longbench_e:
        macro = sum(scores.values()) / len(scores)
        print(f"{'macro average':24s} {macro:.2f}  ({len(scores)} tasks)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

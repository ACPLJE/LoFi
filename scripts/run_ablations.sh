#!/usr/bin/env bash
# Run the ablation grid on LongBench and score every run.
#
# Each preset shares the prompt format, truncation, decoding and compression
# budget, so the only thing that varies is the setting under test:
#
#   main           low-frequency selection + content-preserving merge + mass
#                  correction (the method)
#   no_mass        mass correction removed      -> its contribution
#   select_evict   source evicted, not merged   -> selection vs merge operator
#   uniform_merge  naive 50:50 full-channel merge -> merge operator
#   full_sim       candidates scored on all channels -> selection criterion
#   full_cache     no compression (reference)
#
# Usage:
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct bash scripts/run_ablations.sh
#
# Environment:
#   MODEL       HF id or local path          (default: Meta-Llama-3-8B-Instruct)
#   OUT         output root                  (default: results)
#   HH, RECENT  budget split                 (default: 0.1 / 0.1, i.e. 20% kept)
#   GPU         CUDA_VISIBLE_DEVICES         (default: 0)
#   PRESETS     space-separated preset list  (default: all six)
#   LIMIT       examples per task            (default: unset, i.e. all)

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
OUT="${OUT:-results}"
HH="${HH:-0.1}"
RECENT="${RECENT:-0.1}"
GPU="${GPU:-0}"
PRESETS="${PRESETS:-main no_mass select_evict uniform_merge full_sim full_cache}"

cd "$(dirname "$0")/.."

extra=()
if [[ -n "${LIMIT:-}" ]]; then
  extra+=(--limit "$LIMIT")
fi

# Confirm the budget is actually reached before producing any accuracy number:
# the target ratio is not self-enforcing (see scripts/audit_retention.py).
echo "=== retention audit: target $(python3 -c "print($HH + $RECENT)") ==="
CUDA_VISIBLE_DEVICES="$GPU" python scripts/audit_retention.py \
  --model-path "$MODEL" --hh-ratio "$HH" --recent-ratio "$RECENT" \
  --out "$OUT/retention_audit.json"

for preset in $PRESETS; do
  run_dir="$OUT/${preset}_r$(python3 -c "print($HH + $RECENT)")"
  echo
  echo "=== $preset -> $run_dir ==="
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_longbench.py \
    --model-path "$MODEL" \
    --out-dir "$run_dir" \
    --preset "$preset" \
    --hh-ratio "$HH" \
    --recent-ratio "$RECENT" \
    "${extra[@]+"${extra[@]}"}"
  python -m eval.score_longbench "$run_dir"
done

echo
echo "all runs complete. Per-run scores are in $OUT/*/scores.json"

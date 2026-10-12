#!/bin/bash
#SBATCH --job-name=stuttermark-eval
#SBATCH --partition=nlp
#SBATCH --account=swabhas_1625
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run eval suite E1-E8 (E7 derives from E6) for one model on one A100; finished conditions are skipped on resubmit.
# sbatch --export=ALL,CONFIG=configs/eval/olmo7b.yaml,STUTTERMARK_HF_HOME=/scratch1/$USER/.cache/huggingface scripts/eval_suite.sbatch
set -euo pipefail

: "${CONFIG:?set CONFIG, e.g. configs/eval/olmo7b.yaml}"
: "${STUTTERMARK_HF_HOME:?set STUTTERMARK_HF_HOME to a scratch path (home quota is too small for model caches)}"
PROJECT="${PROJECT:-$PWD}"
export HF_HOME="$STUTTERMARK_HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE"

cd "$PROJECT"
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: ${SLURMD_NODENAME:-unknown}  Config: $CONFIG  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for exp in E1 E2 E3 E4; do
  echo "=== $exp ==="
  uv run python scripts/eval_capability.py --config "$CONFIG" --exp "$exp"
done
for exp in E5 E6; do
  echo "=== $exp ==="
  uv run python scripts/eval_generation.py --config "$CONFIG" --exp "$exp"
done
echo "=== E8 ==="
uv run python scripts/eval_capability.py --config "$CONFIG" --exp E8

if [[ -n "${JUDGE_BASE_URL:-}" && -n "${JUDGE_API_KEY:-}" ]]; then
  echo "=== E5 judge ==="
  uv run python scripts/eval_judge.py --config "$CONFIG"
else
  echo "=== E5 judge skipped (JUDGE_BASE_URL / JUDGE_API_KEY not set) ==="
fi

echo "=== Report ==="
uv run python scripts/eval_report.py --config "$CONFIG"
echo "=== Job finished: $(date) ==="

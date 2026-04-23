#!/bin/bash
#SBATCH --job-name=hct-train
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
# Default log settings for direct sbatch use. `./hct train` overrides these at submit time.
#SBATCH --output=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-train-%j.out
#SBATCH --error=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-train-%j.err

set -euo pipefail

HCT_DIR_RAW="${SLURM_SUBMIT_DIR:-$(pwd)}"
HCT_DIR="${HCT_DIR_RAW/#\/rdcw\/fs2\/bmansfeld\/Active/\/storage1\/fs1\/bmansfeld\/Active}"
export HCT_BASE_DIR="$HCT_DIR"
cd "$HCT_DIR"
mkdir -p "$HCT_DIR/logs"

echo "Submit dir: $HCT_DIR_RAW"
echo "Project root: $HCT_DIR"
echo "Running dataset-centric training in batch mode..."
echo "Python: $(command -v python)"
if [[ -n "${model_name:-}" ]]; then
  echo "Model name: $model_name"
fi

CMD=(python "$HCT_DIR/scripts/04_train.py" --batch)

if [[ -n "${epochs:-}" ]]; then
  CMD+=(--epochs "$epochs")
fi
if [[ -n "${batch_size:-}" ]]; then
  CMD+=(--batch-size "$batch_size")
fi
if [[ -n "${learning_rate:-}" ]]; then
  CMD+=(--learning-rate "$learning_rate")
fi
if [[ -n "${patience:-}" ]]; then
  CMD+=(--patience "$patience")
fi

if [[ -n "${model_name:-}" ]]; then
  CMD+=(--model-name "$model_name")
fi

"${CMD[@]}"

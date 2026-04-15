#!/bin/bash
#SBATCH --job-name=hct-train
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-train-%j.out
#SBATCH --error=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-train-%j.err
#SBATCH --container-image=shafayasghar/hct-train:latest
#SBATCH --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active

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

CMD=(
  python "$HCT_DIR/scripts/04_train.py"
  --batch
  --epochs "${epochs:-30}"
  --batch-size "${batch_size:-8}"
  --learning-rate "${learning_rate:-1e-4}"
  --val-split 0.2
  --patience "${patience:-10}"
  --img-size 128
  --wmse-alpha 50.0
  --dice-weight 0.5
  --fn-weight 2.0
)

if [[ -n "${model_name:-}" ]]; then
  CMD+=(--model-name "$model_name")
fi

"${CMD[@]}"

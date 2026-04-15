#!/bin/bash
#SBATCH --job-name=hct-measure
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-measure-%j.out
#SBATCH --error=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/hct-measure-%j.err
#SBATCH --container-image=shafayasghar/hct-train:latest
#SBATCH --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active

set -euo pipefail

if [[ -z "${dataset:-}" || -z "${model:-}" ]]; then
  echo "Set dataset and model before submitting this job."
  echo 'Example: dataset=F2_001,F2_002 model=my_model sbatch slurm/measure.sh'
  exit 1
fi

HCT_DIR_RAW="${SLURM_SUBMIT_DIR:-$(pwd)}"
HCT_DIR="${HCT_DIR_RAW/#\/rdcw\/fs2\/bmansfeld\/Active/\/storage1\/fs1\/bmansfeld\/Active}"
export HCT_BASE_DIR="$HCT_DIR"
cd "$HCT_DIR"
mkdir -p "$HCT_DIR/logs"

echo "Submit dir: $HCT_DIR_RAW"
echo "Project root: $HCT_DIR"
echo "Dataset: $dataset"
echo "Model: $model"
echo "Python: $(command -v python)"

CMD=(
  python "$HCT_DIR/scripts/05_measure.py"
  --batch
  --datasets "$dataset"
  --model "$model"
  --max-frames "${max_frames:-0}"
  --num-tips "${num_tips:-2}"
  --min-dist "${min_dist:-20}"
  --interval-min "${interval_min:-30}"
)

if [[ -n "${genotype_filter:-}" ]]; then
  CMD+=(--genotype-filter "$genotype_filter")
fi

"${CMD[@]}"

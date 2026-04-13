#!/bin/bash
#SBATCH --job-name=hct-train
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=hct-train-%j.log
#SBATCH --container-image=shafayasghar/hct-train:latest
#SBATCH --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active

# Auto-detect project root from this script's location
HCT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export HCT_BASE_DIR="$HCT_DIR"

echo "Project root: $HCT_DIR"
echo "Running 04_train.py..."

python "$HCT_DIR/scripts/04_train.py"

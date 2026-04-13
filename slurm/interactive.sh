#!/bin/bash
# Interactive GPU session for testing container + data mounts
# Run from the project directory: bash slurm/interactive.sh
#
# Once inside, set up the environment:
#   export HCT_BASE_DIR=$(pwd)
#   python scripts/05_measure.py   # or any script

srun -p general-gpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32GB \
  --time=01:00:00 \
  --container-image="shafayasghar/hct-train:latest" \
  --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active \
  --pty /bin/bash

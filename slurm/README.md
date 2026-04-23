# SLURM usage

The training and measurement entrypoints now use the dataset-centric batch flags.

## Train

Generate training data first:

```bash
./hct masks --datasets F2_001,F2_002 --augment true --augmentations-per-image 4 --genotype-filter 0,4
```

Submit training:

```bash
./hct train my_custom_model --epochs 40 --patience 12 --gpus 1 --cpus 8 --ram 64GB --time 06:00:00
```

If you need a specific GPU-capable image, pass it explicitly:

```bash
./hct train my_custom_model --container-image your/image:tag --container-mounts /storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active
```

This runs `scripts/04_train.py --batch ...` inside the training container and saves the model in `data/models/`.
Logs go to:

- `logs/hct-train-<JOBID>.out`
- `logs/hct-train-<JOBID>.err`

The `./hct` entrypoint also overrides the log path at submit time so log following works from either `/rdcw/...` or `/storage1/...`.

## Measure

Set the dataset and model name when submitting:

```bash
./hct measure F2_001,F2_002 my_custom_model --genotype-filter 0,4 --interval-min 30 --time 03:00:00
```

This writes the current measurement outputs to:

- `data/datasets/<dataset>/output/tip_distances.csv`
- `data/datasets/<dataset>/output/plots/`

```bash
./hct measure F2_001 my_custom_model
```

## Direct `sbatch`

You can still call `sbatch slurm/train.sh` or `sbatch slurm/measure.sh` directly, but you must provide container flags when doing so:

- `--container-image`
- `--container-mounts`

Example:

```bash
sbatch --container-image your/image:tag --container-mounts /storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active slurm/train.sh
```

If your environment differs from this repo owner's Compute2 setup, prefer `./hct train ...` / `./hct measure ...` or edit the fallback directives.

You can also override the container settings used by `./hct` with:

```bash
export HCT_CONTAINER_IMAGE=your/image:tag
export HCT_CONTAINER_MOUNTS=/path1:/path1,/path2:/path2
```

## Monitor jobs

```bash
squeue -u $USER
tail -f logs/hct-train-JOBID.out
tail -f logs/hct-train-JOBID.err
tail -f logs/hct-measure-JOBID.out
tail -f logs/hct-measure-JOBID.err
scancel JOBID
```

# SLURM usage

The training and measurement entrypoints now use the dataset-centric batch flags.

## Train

Generate training data first:

```bash
./hct masks --batch --datasets F2_001,F2_002 --augment true --augmentations-per-image 4 --genotype-filter 0,4
```

Submit training:

```bash
sbatch slurm/train.sh
```

This runs `scripts/04_train.py --batch ...` inside the training container and saves the model in `data/models/`.
Logs go to:

- `logs/hct-train-<JOBID>.out`
- `logs/hct-train-<JOBID>.err`

To choose your own model folder name at submit time:

```bash
./train my_custom_model --epochs 40 --patience 12 --gpus 1 --cpus 8 --ram 64GB
```

Convenience wrapper:

```bash
./train
./train my_custom_model
```

## Measure

Set the dataset and model name when submitting:

```bash
./measure F2_001,F2_002 my_custom_model --genotype-filter 0,4 --interval-min 30
```

This writes the current measurement outputs to:

- `data/datasets/<dataset>/output/tip_distances.csv`
- `data/datasets/<dataset>/output/plots/`

Convenience wrapper:

```bash
./measure F2_001 my_custom_model
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

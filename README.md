# Hierarchical Circadian Tracker

This repo now uses a dataset-centric workflow. The unit of work is a dataset under `data/datasets/<dataset_id>/`, not an experiment id or a shared `_config.py`.

## Layout

```text
data/
  datasets/
    F2_001/
      raw/
      crops.json
      annotations.json
      output/
        tip_distances.csv
        plots/
  training/
    images/
    masks/
    manifest.json
  models/
    F2_001+F2_002_20260413/
      best.keras
      training_info.json
  exports/
    approved_distances.csv
    approved_distances.xls
    export_metadata.json
    summary/
```

## Typical workflow

1. Put images in `data/datasets/<dataset_id>/raw/`.
2. Run `./hct crop` and pick a dataset.
3. Run `./hct annotate` and annotate tips.
4. Run `./hct masks` to generate `data/training/`.
5. Run `./hct train` to submit training.
6. Run `./hct measure <dataset_or_csv> <model>` to measure one or more datasets with one model.
7. Run `./hct export` to combine measured datasets into `data/exports/`.
8. Run `./hct summary` after placing a Biodare detrended CSV in `data/exports/`.

The scripts now only prompt for required choices like dataset/model selection. Other settings stay on defaults unless you override them with CLI flags.

## Commands

```bash
./hct crop --help
./hct annotate --help
./hct masks --help
./hct train --help
./hct measure --help
./hct export --help
./hct summary --help
```

## Batch examples

```bash
./hct masks --batch --datasets F2_001,F2_002 --augment true --augmentations-per-image 4 --genotype-filter 0,4
./hct train my_model --epochs 40 --patience 12 --gpus 1 --cpus 8 --ram 64GB
./hct measure F2_001,F2_002 my_model --genotype-filter 0,4 --interval-min 30
./hct export --batch --datasets F2_001,F2_002 --genotype-names 0=WT,4=F2 --exclude-genotypes 9
./hct summary --batch --input detrended_results.csv --exclude-samples 3,15,22
```

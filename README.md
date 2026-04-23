# Hierarchical Circadian Tracker (HCT)

Pipeline for measuring leaf tip distances from timelapse images and exporting results for BioDare.

All commands go through the `./hct` script at the repo root.

---

## Directory Layout

```
data/
├── datasets/              # Your data lives here — one folder per experiment
│   └── Example_experiment/
│       ├── raw/           # Drop your timelapse frames here
│       ├── crops.json     # Created by "crop"
│       ├── annotations.json
│       └── output/
│           └── my_model/
│               ├── tip_distances.csv
│               └── plots/
├── models/                # Trained models
│   └── my_model/
│       ├── best.keras
│       └── training_info.json
└── exports/               # Final output
    ├── approved_distances.xls   # Upload this to BioDare
    └── summary/                 # Plots from BioDare detrended output
```

The dataset ID is just the folder name under `data/datasets/` (e.g. `F2_run1`).

---

## Standard Workflow

If a model already exists, you only need steps 1–2, then measure → export → summary.

### 1. Put your images in place

Create `data/datasets/<your_id>/raw/` and drop your timelapse frames there. Files are sorted by filename, so name them numerically (`frame_001.tif`, `frame_002.tif`, ...). Supported formats: PNG, JPG, TIF, BMP.

### 2. Crop — draw bounding boxes

```bash
./hct crop
```

Opens a blended composite of your frames. Draw a box around each plant, press a number key (`0`–`9`) to assign its genotype, then `s` to save.

| Key | Action |
|---|---|
| drag | draw a new box |
| right-click / tiny-drag | select an existing box |
| `0`–`9` | assign genotype to selected box |
| `d` | delete selected box |
| `u` | undo last box |
| `c` | clear all |
| `s` / Enter | save |
| `q` / Esc | cancel |

| Flag | Default | Description |
|---|---|---|
| `--dataset ID` | — | Skip the selection prompt |
| `--max-frames N` | 180 | Frames to use for the composite |

### 3. Measure — run inference (GPU, via SLURM)

```bash
./hct measure <dataset_csv> <model>
```

Runs the model on every frame of every plant and writes `tip_distances.csv` + per-plant plots to `data/datasets/<id>/output/<model_name>/`. Submits to SLURM and streams the log — `Ctrl+C` detaches but the job keeps running.

```bash
./hct measure F2_run1 my_model
./hct measure F2_run1,F2_run2 my_model --interval-min 30 --genotype-filter 0,4
```

| Flag | Default | Description |
|---|---|---|
| `--interval-min N` | 30 | Minutes between frames (sets time axis) |
| `--num-tips N` | 2 | Number of tips to detect per frame |
| `--min-dist N` | auto | Min distance in the resized model input image |
| `--max-frames N` | 0 (all) | Limit frames |
| `--genotype-filter 0,4` | — | Only measure these genotypes |
| `--gpus N` | 1 | GPUs to request |
| `--cpus N` | 4 | CPUs to request |
| `--ram SIZE` | 32GB | RAM to request |

### 4. Export — combine into a single file

```bash
./hct export
```

Merges one selected measured output from each dataset into `data/exports/approved_distances.xls` (and `.csv`). Upload the `.xls` to BioDare.

```bash
./hct export --datasets F2_run1,F2_run2 --genotype-names 0=WT,4=Mutant --exclude-plants g4_2
```

| Flag | Default | Description |
|---|---|---|
| `--datasets ID1,ID2` | — | Skip the selection prompt |
| `--genotype-names 0=WT,4=Mut` | — | Column header names |
| `--exclude-genotypes 9` | — | Drop entire genotypes |
| `--exclude-plants g4_2,g3_*` | — | Drop individual plants (`*` wildcard) |
| `--interval-min N` | 30 | Minutes between frames |

### 5. Summary — plot BioDare detrended output

Place the detrended CSV from BioDare in `data/exports/`, then:

```bash
./hct summary
./hct summary --input detrended_results.csv --exclude-samples 3,15 --use-sem
```

Plots go to `data/exports/summary/`.

| Flag | Default | Description |
|---|---|---|
| `--input file.csv` | auto-detect | Detrended CSV filename |
| `--exclude-samples 3,15` | — | Drop samples by BioDare sample number |
| `--use-sem` | STD bands | Plot SEM instead of STD |
| `--colors M82=tab:blue,...` | built-in colors | Override genotype colors |

---

## Training a New Model (optional)

Only needed if you're training from scratch or retraining on new data. 

### Annotate — click leaf tips

```bash
./hct annotate
```

For each plant, shows a random sample of frames. Click the leaf tip(s) on each frame, then save.

| Key | Action |
|---|---|
| left-click | place a tip |
| `s` | save and go to next frame |
| `r` | reset clicks on current frame |
| `n` | skip frame, replace with another |
| `x` | skip frame |
| `q` | quit |

| Flag | Default | Description |
|---|---|---|
| `--dataset ID` | — | Skip the selection prompt |
| `--images-per-plant N` | 5 | Target annotations per plant |
| `--display-scale N` | 2 | Zoom level for the window |

### Masks — generate training data

```bash
./hct masks
./hct masks --datasets F2_run1,F2_run2 --augment true --augmentations-per-image 4
```

Converts annotations into image/mask pairs in `data/training/`. **Overwrites the training folder every run.** Include multiple datasets to pool their annotations.

| Flag | Default | Description |
|---|---|---|
| `--datasets ID1,ID2` | — | Skip the selection prompt |
| `--augment true/false` | true | Enable augmentation |
| `--augmentations-per-image N` | 4 | Augmented copies per image |
| `--genotype-filter 0,4` | — | Only include these genotypes |

### Train — train the model (GPU, via SLURM)

```bash
./hct train my_model
./hct train my_model --epochs 60 --patience 15
```

Submits to SLURM, streams the log. Model is saved to `data/models/<name>/`. Name it something descriptive (e.g. `F2_run1+F2_run2`) — if omitted, a timestamp is used.

| Flag | Default | Description |
|---|---|---|
| `model_name` | timestamp | Folder name under `data/models/` |
| `--epochs N` | 30 | Max epochs |
| `--patience N` | 10 | Early stopping — epochs without val improvement |
| `--gpus N` | 1 | GPUs to request |
| `--cpus N` | 8 | CPUs to request |
| `--ram SIZE` | 64GB | RAM to request |
| `--time HH:MM:SS` | 04:00:00 | SLURM wall time |
| `--batch-size N` | 8 | Batch size |
| `--learning-rate X` | 0.0001 | Learning rate |

---

## Notes

- Genotype numbers (`0`–`9`) are assigned at crop time. Pick a scheme and be consistent — you can give them names later with `--genotype-names` at export.
- Re-cropping after annotating is safe: annotations are remapped to new boxes if positions match, dropped if boxes were removed.
- `./hct <command> --help` shows all options for any command.
- `./hct train` and `./hct measure` override the log paths at submit time so they work whether you entered the repo via `/rdcw/...` or `/storage1/...`.
- If your Compute2 container image or mounts differ, set `HCT_CONTAINER_IMAGE` and/or `HCT_CONTAINER_MOUNTS` before running `./hct train` or `./hct measure`.

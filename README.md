# Hierarchical Circadian Tracker

Automated leaf tip tracking for measuring circadian rhythm in plants. Uses a U-Net neural network to detect leaf tip positions across time-lapse image sequences and measure the distance between tips over time.

## Requirements

- Python 3.10+
- A virtual environment is recommended

### Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Quick Start

The pipeline has 7 numbered scripts that run in order. All configuration lives in `scripts/_config.py` -- edit that file before running anything.

### 1. Set up your experiment

Place your raw time-lapse images (PNG or JPG) into a subfolder under `data/raw/`:

```
data/raw/my_dataset/
    frame_0000.jpg
    frame_0001.jpg
    frame_0002.jpg
    ...
```

Then edit `scripts/_config.py`:

```python
EXPERIMENT_ID = "my_experiment"    # name for this experiment
DATASET_ID    = "my_dataset"       # folder name under data/raw/
```

### 2. Define crop regions (`01_crop.py`)

```bash
python scripts/01_crop.py
```

This opens an interactive window showing a composite of all your frames blended together. You will draw a rectangle around each plant:

- **Click and drag** to draw a bounding box around a plant
- Press **1**, **2**, **3**, or **4** to assign a genotype number to that box
- Repeat for every plant in the image
- Press **s** or **Enter** to save
- Press **d** to delete the last box, **u** to undo, **c** to clear all
- Press **q** or **Esc** to cancel without saving

Replicate numbers are assigned automatically in draw order per genotype (g1_r01, g1_r02, g2_r01, etc.).

**Output:** `data/experiments/{EXPERIMENT_ID}/logs/01_crop.json`

**Key config options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `BLEND_METHOD` | `"max"` | How frames are blended (`"max"` or `"mean"`) |
| `MAX_FRAMES` | `180` | Limit to first N frames (0 = all) |
| `EXPECTED_PLANTS` | `0` | Enforce plant count (0 = no enforcement) |

### 3. Annotate leaf tips (`02_annotate.py`)

```bash
python scripts/02_annotate.py
```

For each plant, the script shows a zoomed crop at several time points. You click on each visible leaf tip.

- **Left-click** to place a tip marker (numbered circles appear)
- Press **s** to save annotations for this frame and move to the next
- Press **r** to reset the current frame's annotations
- Press **n** to skip this frame (picks a replacement)
- Press **x** to skip without replacement
- Press **q** to quit (saves progress)

**Output:** `data/experiments/{EXPERIMENT_ID}/logs/02_annotate.json`

**Key config options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGES_PER_FOLDER` | `3` | How many frames to annotate per plant |
| `DISPLAY_SCALE` | `3` | Zoom factor for the annotation window |

### 4. Generate training data (`03_masks.py`)

```bash
python scripts/03_masks.py
```

Converts your annotations into training image + mask pairs. Each mask is a Gaussian heatmap centered on the annotated tips. Augmented copies (rotated, flipped, brightness-shifted) are created automatically.

This step combines annotations from **multiple experiments** -- set which ones in config:

```python
TRAINING_EXPERIMENTS = ["my_experiment", "another_experiment"]
```

**Output:**
- `data/training/images/` -- cropped leaf images (PNG)
- `data/training/masks/` -- corresponding heatmap masks (PNG)
- `data/training/training_manifest.json`

**Key config options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAINING_EXPERIMENTS` | `[]` | Which experiments to combine for training |
| `SIGMA` | `4` | Gaussian kernel size for heatmaps |
| `AUGMENT` | `True` | Enable data augmentation |
| `AUGS_PER_IMAGE` | `4` | Augmented copies per annotated image |

### 5. Measure leaf tip distances (`05_measure.py`)

```bash
python scripts/05_measure.py
```

Runs the trained model on every frame of your experiment, detects leaf tips, and measures the distance between them over time. Produces a CSV and per-plant plots.

To process multiple experiments at once, set:

```python
MEASURE_EXPERIMENTS = ["experiment_1", "experiment_2"]
```

Leave it empty to just process the single `EXPERIMENT_ID`.

**Output (per experiment):**
- `data/experiments/{EXPERIMENT_ID}/output/tip_distances.csv`
- `data/experiments/{EXPERIMENT_ID}/output/{label}_tip_distance.png` (one plot per plant)

**Output (multi-experiment):**
- `data/experiments/{first_experiment}/output/combined/combined_tip_distances.csv`

**Key config options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MEASURE_EXPERIMENTS` | `[]` | Experiments to process together (empty = just EXPERIMENT_ID) |
| `MODEL_PATH` | `""` | Path to a specific model (empty = auto-find latest) |
| `MODEL_MODE` | `"shared"` | `"shared"` = one model for all, `"auto"` = each experiment's own |
| `MEASURE_MAX_FRAMES` | `0` | Limit to first N frames (0 = all) |
| `NUM_TIPS` | `2` | Number of leaf tips to detect per plant |
| `MIN_DIST` | `20` | Minimum pixel distance between detected tips |
| `INTERVAL_MIN` | `30` | Minutes between frames (for time axis) |

### 6. Export formatted data (`06_export.py`)

```bash
python scripts/06_export.py
```

Converts the raw tip distances CSV into a formatted spreadsheet with genotype-based column names (e.g., `M82_1`, `Penelli_2`). Outputs both CSV and Excel 2003 (.xls) format.

Map genotype numbers to names in config:

```python
GENOTYPE_NAMES = {1: "M82", 2: "Penelli", 3: "pimpi", 4: "F2"}
```

**Output (single experiment):**
- `data/experiments/{EXPERIMENT_ID}/output/approved_distances.csv`
- `data/experiments/{EXPERIMENT_ID}/output/approved_distances.xls`

**Output (multi-experiment):**
- `data/experiments/{first_experiment}/output/combined/approved_distances.csv`
- `data/experiments/{first_experiment}/output/combined/approved_distances.xls`

Column format: `time, GenotypeName_1, GenotypeName_2, ...`
Multi-experiment column format: `time, ExperimentID | GenotypeName_1, ...`

#### Excluding plants

Use the `EXCLUDE_PLANTS` dict in `_config.py`:

```python
EXCLUDE_PLANTS = {
    "*":      ["g4_2", "g4_8"],            # exclude from ALL experiments
    "F2_1":   ["g4_3", "g4_12"],           # exclude only from F2_1
    "F2_2_1": ["g3_*"],                    # exclude all genotype 3 from F2_2_1
}
```

Patterns:
- `"g4_2"` -- exact match (genotype 4, replicate 2)
- `"g3_*"` -- wildcard (all plants of genotype 3)
- `"*"` key -- applies to every experiment

### 7. Summary plots (`07_summary.py`)

```bash
python scripts/07_summary.py
```

Reads a Biodare-detrended CSV (you upload `approved_distances.csv` to [Biodare](https://biodare2.ed.ac.uk/) and download the detrended result). Place the detrended file in the experiment's `output/` folder.

Generates per-genotype mean +/- spread plots and a combined overlay plot.

**Output:**
- `data/experiments/{EXPERIMENT_ID}/output/summary/{genotype}_summary.png`
- `data/experiments/{EXPERIMENT_ID}/output/summary/all_genotypes_summary.png`

## Helper Scripts

| Script | Description | Interactive? |
|--------|-------------|:---:|
| `check_model.py` | Visually inspect model predictions frame-by-frame with peak markers | Yes |
| `make_videos.py` | Generate MP4 videos showing crops with heatmap overlays | No |
| `filter_approved.py` | Remove specific sample columns from approved_distances.csv | No |
| `temp_mask_overlay_viewer.py` | Browse training image/mask pairs for QA | Yes |

## Configuration

All config lives in `scripts/_config.py`. Every script reads from this file.

### Per-script overrides

Each script has an `_OVERRIDE` dict at the top. To change a value for just that script without editing `_config.py`:

```python
# At the top of 05_measure.py:
_OVERRIDE = {"MEASURE_MAX_FRAMES": 50, "NUM_TIPS": 2}
```

### Environment variable

Set `HCT_BASE_DIR` to override the project root path (used for running in Docker or on a remote server):

```bash
export HCT_BASE_DIR=/path/to/project
```

## Data Layout

```
data/
  raw/
    {DATASET_ID}/              # your raw time-lapse images
  training/
    images/                    # generated by 03_masks.py
    masks/                     # generated by 03_masks.py
    training_manifest.json
  experiments/
    {EXPERIMENT_ID}/
      logs/
        01_crop.json           # crop definitions
        02_annotate.json       # tip annotations
      models/
        {YYYYMMDD_HHMMSS}/     # one folder per training run
          best.keras           # best model by validation dice score
          final.keras          # model at last epoch
          training_info.json   # metrics and hyperparameters
      output/
        tip_distances.csv      # raw measurements
        approved_distances.csv # formatted export
        approved_distances.xls # Excel 2003 export
        combined/              # multi-experiment outputs
        summary/               # Biodare summary plots
```

## Training a Model

If you need to train or retrain the model (most users will use a pre-trained model), follow these steps after completing steps 1-3 above for at least one experiment.

### Prerequisites

- A GPU is strongly recommended (CPU training is very slow)
- At least ~50 annotated images across your experiments
- More annotations from diverse genotypes/conditions will improve the model

### Configure training

In `_config.py`, set which experiments to use for training:

```python
TRAINING_EXPERIMENTS = ["experiment_1", "experiment_2"]
```

### Generate training data

```bash
python scripts/03_masks.py
```

This wipes and regenerates `data/training/` from the listed experiments.

### Run training

```bash
python scripts/04_train.py
```

The script trains a U-Net model and saves checkpoints to `data/experiments/{EXPERIMENT_ID}/models/{timestamp}/`.

Training stops early if validation performance plateaus (default patience: 10 epochs). The best model by validation dice score is saved as `best.keras`.

**Key config options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | `128` | Input image size (square) |
| `BATCH_SIZE` | `8` | Training batch size |
| `EPOCHS` | `30` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Adam optimizer learning rate |
| `VAL_SPLIT` | `0.2` | Fraction held for validation |
| `PATIENCE` | `10` | Early stopping patience |

### Model architecture

The model is a 4-level U-Net:
- **Encoder:** 64 -> 128 -> 256 -> 512 channels, each with 2x Conv2D + BatchNorm + ReLU, downsampled with MaxPool
- **Decoder:** mirrors the encoder with skip connections, upsampled with UpSampling2D
- **Output:** single-channel sigmoid heatmap (same size as input)

### Loss function

A combined loss of three components:
- **Weighted MSE** (alpha=50) -- upweights tip pixels so the model focuses on them
- **Soft Dice** (weight=0.5) -- encourages overlap with ground truth regions
- **False-Negative penalty** (weight=2.0) -- heavily penalizes missed tips

### Verifying the model

After training, use the interactive viewer to inspect predictions:

```bash
python scripts/check_model.py
```

Arrow keys navigate frames and plants. Look for accurate peak detection and consistent tracking across frames.

### Tips for better models

- Annotate more images if predictions are poor (especially edge cases)
- Include images from multiple genotypes and experimental conditions
- If tips are frequently confused with other features, increase `SIGMA` for broader heatmaps
- If tips are missed, try increasing `FN_WEIGHT`
- If there are false positives, try increasing `WMSE_ALPHA`

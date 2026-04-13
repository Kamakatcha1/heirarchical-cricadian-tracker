# Running on the WashU Compute2 Cluster

## Prerequisites

- WashU RIS compute2 access (watch the [onboarding video](https://wustl.app.box.com/s/f05ay70rmd12jp6h3ce41p2x3zxwg63h/file/1987582952643), then ask Ben to add you)
- VPN connected ([WashU VPN](https://it.wustl.edu/items/connect/))

## First-Time Setup

### 1. Copy the project to your work folder

SSH into the cluster:

```bash
ssh YOUR_WUSTL_KEY@c2-login-001.ris.wustl.edu
```

Create your work directory and copy the project:

```bash
mkdir -p /storage1/fs1/bmansfeld/Active/work/YOUR_NAME
cp -r /storage1/fs1/bmansfeld/Active/work/a.shafay/hct /storage1/fs1/bmansfeld/Active/work/YOUR_NAME/hct
cd /storage1/fs1/bmansfeld/Active/work/YOUR_NAME/hct
```

### 2. Set up Python for GUI scripts

On the **RIS virtual desktop**, open a terminal and run:

```bash
cd /storage1/fs1/bmansfeld/Active/work/YOUR_NAME/hct
bash slurm/setup_venv.sh
```

This creates a Python virtual environment with all dependencies. You only need to do this once.

## Typical Workflow

### Step 1: Prepare your raw data

Place your time-lapse images in:

```
data/raw/YOUR_DATASET_NAME/
    frame_0000.jpg
    frame_0001.jpg
    ...
```

### Step 2: Edit config

Edit `scripts/_config.py`:

```python
EXPERIMENT_ID = "your_experiment"
DATASET_ID    = "YOUR_DATASET_NAME"
```

Change `BASE_DIR` to point to your project folder (or leave it -- the SLURM scripts set `HCT_BASE_DIR` automatically):

### Step 3: Crop and annotate (virtual desktop)

On the RIS virtual desktop:

```bash
cd /storage1/fs1/bmansfeld/Active/work/YOUR_NAME/hct
source venv/bin/activate
python scripts/01_crop.py       # draw boxes around plants
python scripts/02_annotate.py   # click leaf tips
python scripts/03_masks.py      # generate training data (if training)
```

### Step 4: Train (if needed)

Most users will use an existing trained model. If you need to train:

```bash
sbatch slurm/train.sh
```

### Step 5: Measure

```bash
sbatch slurm/measure.sh
```

### Step 6: Export (can run on virtual desktop or as batch)

On the virtual desktop:

```bash
source venv/bin/activate
python scripts/06_export.py
python scripts/07_summary.py
```

## Monitoring Jobs

```bash
# Check your running jobs
squeue -u $USER

# View job output (replace JOBID with the number from squeue)
cat hct-train-JOBID.log
cat hct-measure-JOBID.log

# Cancel a job
scancel JOBID
```

## Interactive GPU Session

For testing or debugging, get an interactive session inside the Docker container:

```bash
cd /storage1/fs1/bmansfeld/Active/work/YOUR_NAME/hct
bash slurm/interactive.sh
```

Once inside:

```bash
export HCT_BASE_DIR=$(pwd)
python scripts/05_measure.py
```

## Troubleshooting

**"No module named tensorflow"** -- You're running a GPU script outside the Docker container. Use `sbatch` or the interactive session instead.

**"Crop log not found"** -- You haven't run `01_crop.py` for this experiment yet. Run it on the virtual desktop first.

**"Model not found"** -- No trained model exists for this experiment. Either train one (`sbatch slurm/train.sh`) or set `MODEL_PATH` in `_config.py` to point to an existing model.

**Job stuck in queue** -- Check `squeue -u $USER`. The `general-gpu` partition may be busy. You can check with `sinfo -p general-gpu`.

**Container image not found** -- The Docker image may need to be re-pulled. Ask the project maintainer to verify `shafayasghar/hct-train:latest` is on Docker Hub.

## Resource Limits

| Partition | Max GPUs | Max CPUs | Max Memory |
|-----------|----------|----------|------------|
| general-gpu | 8 | - | 80 GB/GPU |
| general-cpu | - | 128 | 2 TB |
| general-interactive | - | 8 | 64 GB |

The lab has a $2,800/year subsidy for shared partitions. Use the `artsci` partition when available (free for Arts & Sciences labs).

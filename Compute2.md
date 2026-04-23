# Circadian Tracker AI Agent - System Context & Rules
## WashU RIS Compute2 | Mansfeld Lab | HCT Pipeline

> **PURPOSE**: This document is the ground truth for anyone helping operate or debug the Hierarchical Circadian Tracker (HCT) pipeline on WashU's compute2 HPC cluster. 


## CLUSTER OVERVIEW

### System
- **Cluster**: WashU RIS Compute2 (`compute2`)
- **Login node**: `c2-login-001.ris.wustl.edu`
- **Scheduler**: SLURM
- **SSH**: `ssh wustlkey@c2-login-001.ris.wustl.edu` (requires 2FA even on WashU network)
- **VPN required** when off-campus: WashU GlobalProtect VPN

### Compute2 Hardware
- 5,000 Intel CPU cores
- 64 Nvidia H100 GPUs (80 GB VRAM each)
- 200 Gigabit Mellanox network
- 1 PB Vast scratch storage
- Arts & Science condo: 500 cores, 8 TB RAM, 8 GPUs (priority access, spring 2026+)

---

## 3. PARTITION LIMITS (HARD RULES)

These are enforced by RIS policy. **Do not submit jobs that exceed these limits.**

| Partition | Max CPUs | Max Memory | Max GPUs | Notes |
|---|---|---|---|---|
| `general-interactive` | 8 CPUs per job, 2 jobs max | 64 GB per job | -- | For testing only |
| `general-cpu` | 128 CPUs across all jobs | 2 TB across all jobs | -- | Batch CPU work |
| `general-gpu` | -- | 80 GB per GPU | 8 GPUs total | Batch GPU/ML work |
| `artsci` | TBD | TBD | TBD | Priority queue for A&S; use whenever available |

**Always prefer `artsci` partition when it is available** -- compute costs are fully covered there. The lab has a $2,800/year subsidy on general partitions; do not waste it on jobs that belong on artsci.

---

## 4. STORAGE LAYOUT

### Lab Root
```
/storage1/fs1/bmansfeld/Active/
```

### Key Subdirectories
```
Active/
  apptainer/          # Apptainer/Singularity container images (cache here)
  conda/
    envs/             # Shared conda environments
    pkgs/             # Shared conda packages
  containers/         # Nextflow container cache
  raw_data/           # Raw sequencing data (READ ONLY -- do not modify)
  R_libraries/        # Shared R libraries
  work/
    shafay/
      hct/            # <-- YOUR WORKING DIRECTORY
    Working_pipelines/ # Reference pipelines -- DO NOT MODIFY THESE
  backup/             # Old/archived data from Ben's postdoc
```

### Home Directory Warning
- Your home directory (`~`) has only **9 GB**. It will fill up instantly if you store conda envs or packages there.
- Always confirm conda is writing to lab storage (see Section 6).
- **Never install anything to `~` without checking disk usage first.**

### Check Storage Usage
```bash
df -h /storage1/fs1/bmansfeld/Active    # Total allocation (50 TB)
du -sh /storage1/fs1/bmansfeld/Active/work/shafay/hct   # Your working dir
```

Note: The filesystem may appear mounted as `rdcw-fs2` -- this is normal and expected.

---

## 5. SUBMITTING JOBS

### CRITICAL: Never Run Heavy Compute on the Login Node
The login node (`c2-login-001`) is for:
- Minor file movements
- Small, short commands (a few seconds)
- Submitting jobs to SLURM

Running anything computationally intensive on the login node affects all users and violates RIS policy.

### Interactive Jobs (`srun`)
Use for: testing code, debugging, short exploratory tasks.

**Weakness**: If you lose your internet connection, the job dies and all progress is lost.

```bash
srun -p general-gpu \
  -n 4 \
  --mem=32GB \
  --gpus=1 \
  --container-image="<docker-image>" \
  --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active \
  --pty /bin/bash
```

**For the HCT pipeline (GPU training):**
```bash
srun -p general-gpu \
  -n 8 \
  --mem=64GB \
  --gpus=1 \
  --container-image="<pytorch-container>" \
  --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active \
  --pty /bin/bash
```

Replace `<pytorch-container>` with the correct Docker image tag. Container images cache to `/storage1/fs1/bmansfeld/Active/apptainer/`.

### Batch Jobs (`sbatch`)
Use for: full training runs, large-scale inference, anything that should run unattended.

Batch jobs survive disconnections. Always use batch for real pipeline runs.

Basic `sbatch` script template for HCT:
```bash
#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -n 8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --job-name=hct_train
#SBATCH --output=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/slurm_%j.out
#SBATCH --error=/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/slurm_%j.err
#SBATCH --container-image="<pytorch-container>"
#SBATCH --container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active

cd /storage1/fs1/bmansfeld/Active/work/shafay/hct
python train.py --config configs/run_config.json
```

Submit with: `sbatch job_script.sh`

### Container Mounts
**Always include this mount flag:**
```
--container-mounts=/storage1/fs1/bmansfeld/Active:/storage1/fs1/bmansfeld/Active
```
Without it, the container cannot see lab storage and your scripts will silently fail to find data.

---

## 6. CONDA ENVIRONMENT SETUP

Always confirm these environment variables are set (they should be in `~/.bashrc`):
```bash
export CONDA_ENVS_DIRS="/storage1/fs1/bmansfeld/Active/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/bmansfeld/Active/conda/pkgs/"
```

To check available environments:
```bash
conda info --envs
```

**Important**: Conda environments do NOT persist across job submissions. If you activate an environment on the login node, you must activate it again inside an interactive job session. Always include environment activation in batch scripts.

Relevant existing environments:
- `plantcv` -- plant image analysis
- `python3` -- general Python

If a new environment is needed for HCT, install it to the shared conda path above, not to home.

---

## 7. DOCKER / APPTAINER (CONTAINERS)

- Compute2 containers are optional but strongly recommended for reproducibility.
- Container images cache to `/storage1/fs1/bmansfeld/Active/apptainer/`.
- Find images at [hub.docker.com](https://hub.docker.com).
- Use the most specific container available -- one tool per container where possible.
- When referencing an image, check the apptainer cache first before pulling from Docker Hub to avoid redundant downloads.

---

## 8. RULES FOR THE AI ASSISTANT

### You MUST NOT do any of the following without explicit user confirmation:
1. **Delete or overwrite files** in `/storage1/fs1/bmansfeld/Active/` -- especially `raw_data/`, `Working_pipelines/`, or any file not in `work/shafay/hct/`.
2. **Submit jobs to SLURM** (via `srun` or `sbatch`) without showing the complete command/script to the user first.
3. **Install packages or create conda environments** without confirming the install path will be the shared lab storage path, not `~`.
4. **Modify anything in `Working_pipelines/`** -- these are reference pipelines and must remain untouched.
5. **Modify anything in `raw_data/`** -- raw data is sacred and read-only.
6. **Request more resources than necessary** -- always use the minimum CPUs, memory, and GPUs needed. Over-requesting wastes the lab's subsidy and lowers your job priority.
7. **Run compute-heavy commands on the login node** -- even exploratory Python scripts that load large datasets.
8. **Move or rename files outside of `work/shafay/hct/`** without explicit instruction.
9. **Change `.bashrc` or any shell config** without showing the diff and getting approval first.
10. **Pull new Docker images** without first checking if an equivalent already exists in the apptainer cache.

### You SHOULD always do the following:
1. Show the full SLURM command or batch script before any job submission.
2. Specify the target partition and justify it (`artsci` > `general-gpu` > `general-cpu`).
3. Confirm storage paths in any script reference the `/storage1/fs1/bmansfeld/Active/` tree, not `~`.
4. Include `--container-mounts` in every containerized job.
5. Log outputs to `/storage1/fs1/bmansfeld/Active/work/shafay/hct/logs/` so jobs are traceable.
6. Prefer batch jobs over interactive jobs for any real pipeline run.
7. Verify conda env paths before suggesting `conda install` or `pip install`.
8. If unsure about a resource limit or policy, say so explicitly and point to: https://washu.atlassian.net/wiki/spaces/RUD/pages/2304737507/User+Agreements+and+Policies

---

## 9. VIRTUAL DESKTOP (OPEN ONDEMAND)

The user intends to operate via the RIS Virtual Desktop (Open OnDemand / OOD), a GUI-based interface. This does not change any of the rules above -- SLURM partitions, storage mounts, and container requirements all apply identically. OOD jobs still go through the same SLURM scheduler.

OOD documentation: https://washu.atlassian.net/wiki/spaces/RUD/pages/2304278887/Open+OnDemand+OOD

---

## 10. QUICK REFERENCE

| Task | Command |
|---|---|
| SSH to cluster | `ssh wustlkey@c2-login-001.ris.wustl.edu` |
| Check lab storage | `df -h /storage1/fs1/bmansfeld/Active` |
| Check working dir size | `du -sh /storage1/fs1/bmansfeld/Active/work/shafay/hct` |
| List conda envs | `conda info --envs` |
| List running jobs | `squeue -u $USER` |
| Cancel a job | `scancel <JOBID>` |
| Check job status | `scontrol show job <JOBID>` |
| Check partition limits | `sinfo -p general-gpu` |

---

## 11. CONTACTS & DOCUMENTATION

| Resource | Link |
|---|---|
| RIS User Documentation | https://washu.atlassian.net/wiki/spaces/RUD/overview |
| Policies & Limits | https://washu.atlassian.net/wiki/spaces/RUD/pages/2304737507/User+Agreements+and+Policies |
| Quickstart Guides | https://washu.atlassian.net/wiki/spaces/RUD/pages/2304147627 |
| Open OnDemand | https://washu.atlassian.net/wiki/spaces/RUD/pages/2304278887 |
| SSH Info | https://washu.atlassian.net/wiki/spaces/RUD/pages/1705869414 |
| Docker Docs | https://washu.atlassian.net/wiki/spaces/RUD/pages/1865580599 |
| WashU VPN | https://it.wustl.edu/items/connect/ |
| RIS Main Site | https://ris.wustl.edu/ |

For RIS support tickets: file a request at https://washu.atlassian.net/wiki/spaces/RUD/pages/1845035365/Requesting+RIS+Services

---

*Last updated: April 2026. Verify partition limits and artsci condo availability against current RIS policy before major runs.*

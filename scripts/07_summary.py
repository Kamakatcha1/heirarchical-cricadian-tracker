# ============================================================
# Genotype summary plots from Biodare detrended CSV.
# Produces per-genotype + combined overlay plots.
# ============================================================
_OVERRIDE = {}
# ============================================================

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_DIR  = _config.EXPERIMENT_DIR
EXCLUDE_SAMPLES = _config.EXCLUDE_SAMPLES

# ---- Settings ----
USE_STD = True          # True = mean +/- STD,  False = mean +/- SEM
COLORS = {"M82": "tab:blue", "Penelli": "tab:orange", "Pimpi": "tab:green"}


# ---- Parsing ----

def parse_biodare_csv(path: Path) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    """Parse Biodare detrended CSV.

    Returns (time_hours, {genotype_name: [array_per_replicate, ...]}).
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find the Label row
    label_row = None
    label_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip().lower().startswith("label"):
            label_row = row
            label_idx = i
            break
    if label_row is None:
        raise ValueError("Could not find 'Label:' row in CSV.")

    # Parse genotype from labels like "1.[B2] M82" or "26.[AA2] Pimpi"
    labels = label_row[1:]  # skip "Label:" cell
    sample_nums: list[int] = []      # sample number per column
    genotype_names: list[str] = []   # genotype name per column
    col_indices: list[int] = []      # original column index (1-based in CSV row)
    for col_offset, lab in enumerate(labels):
        lab = lab.strip()
        if not lab:
            continue
        # Extract sample number: "1.[B2] M82" -> 1
        num_match = re.match(r'(\d+)\.', lab)
        sample_num = int(num_match.group(1)) if num_match else col_offset + 1
        # Skip excluded samples
        if sample_num in EXCLUDE_SAMPLES:
            continue
        sample_nums.append(sample_num)
        col_indices.append(col_offset + 1)  # +1 because col 0 is time
        # Extract the name after the bracket: "1.[B2] M82" -> "M82"
        m = re.search(r'\]\s*(.+)$', lab)
        if m:
            genotype_names.append(m.group(1).strip())
        else:
            genotype_names.append(lab)

    if EXCLUDE_SAMPLES:
        print(f"  Excluding samples: {sorted(EXCLUDE_SAMPLES)}")

    # Data rows start after label row
    times = []
    traces: list[list[float]] = [[] for _ in genotype_names]

    for row in rows[label_idx + 1:]:
        if not row or not row[0].strip():
            continue
        try:
            t = float(row[0])
        except ValueError:
            continue
        times.append(t)
        for i, ci in enumerate(col_indices):
            cell = row[ci].strip() if ci < len(row) else ""
            try:
                traces[i].append(float(cell))
            except ValueError:
                traces[i].append(float("nan"))

    time_arr = np.array(times)

    # Group by genotype name
    grouped: dict[str, list[np.ndarray]] = {}
    for name, data in zip(genotype_names, traces):
        grouped.setdefault(name, []).append(np.array(data))

    return time_arr, grouped


# ---- Stats ----

def summarize(replicates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and spread (STD or SEM) across replicates."""
    stacked = np.array(replicates)  # shape: (n_reps, n_timepoints)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    if USE_STD:
        spread = std
    else:
        n = np.sum(np.isfinite(stacked), axis=0).astype(float)
        n[n == 0] = 1
        spread = std / np.sqrt(n)
    return mean, spread


# ---- Plotting ----

def plot_one(name: str, hours: np.ndarray, mean: np.ndarray, spread: np.ndarray,
             color: str, n_reps: int, out_dir: Path) -> None:
    spread_label = "STD" if USE_STD else "SEM"
    plt.figure(figsize=(9, 4))
    plt.plot(hours, mean, color=color, linewidth=2, label=name)
    plt.fill_between(hours, mean - spread, mean + spread, color=color, alpha=0.2)
    plt.xlabel("Hours")
    plt.ylabel("Detrended value")
    plt.title(f"{name}: mean +/- {spread_label} (n={n_reps})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{name.lower()}_summary.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  {out_path.name}")


def plot_all(all_data: dict, hours: np.ndarray, out_dir: Path) -> None:
    spread_label = "STD" if USE_STD else "SEM"
    plt.figure(figsize=(10, 5))
    for name, (mean, spread, n_reps) in all_data.items():
        color = COLORS.get(name, "tab:gray")
        plt.plot(hours, mean, color=color, linewidth=2,
                 label=f"{name} (n={n_reps})")
        plt.fill_between(hours, mean - spread, mean + spread, color=color, alpha=0.2)
    plt.xlabel("Hours")
    plt.ylabel("Detrended value")
    plt.title(f"All genotypes: mean +/- {spread_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "all_genotypes_summary.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  {out_path.name}")


# ---- Main ----

def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)
    output_dir = experiment_dir / "output"

    # Find the biodare detrended CSV
    candidates = list(output_dir.glob("*[Dd]etrended*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No detrended CSV found in {output_dir}")
    csv_path = candidates[0]
    print(f"Input: {csv_path.name}")

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    hours, grouped = parse_biodare_csv(csv_path)
    print(f"Timepoints: {len(hours)} ({hours[0]:.1f}h - {hours[-1]:.1f}h)")

    all_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for name, reps in grouped.items():
        n_reps = len(reps)
        print(f"\n  {name}: {n_reps} replicates")
        mean, spread = summarize(reps)
        all_data[name] = (mean, spread, n_reps)
        color = COLORS.get(name, "tab:gray")
        plot_one(name, hours, mean, spread, color, n_reps, summary_dir)

    print()
    plot_all(all_data, hours, summary_dir)

    print(f"\nAll plots saved to: {summary_dir}")


if __name__ == "__main__":
    main()

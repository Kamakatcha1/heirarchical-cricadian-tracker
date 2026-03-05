# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# ============================================================
_OVERRIDE = {}
# ============================================================

import csv
from collections import defaultdict
from pathlib import Path

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_ID  = _config.EXPERIMENT_ID
EXPERIMENT_DIR = _config.EXPERIMENT_DIR
INTERVAL_MIN   = _config.INTERVAL_MIN
GENOTYPE_NAMES = _config.GENOTYPE_NAMES
EXCLUDE_PLANTS = _config.EXCLUDE_PLANTS


def load_tip_distances(csv_path: Path) -> dict[str, dict]:
    """Load tip_distances.csv, group by plant label."""
    plants: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in plants:
                plants[label] = {
                    "genotype": int(row["genotype"]),
                    "replicate": row["replicate"],
                    "frames": [],
                }
            val = row["tip_distance_px"]
            plants[label]["frames"].append(float(val) if val != "nan" else float("nan"))
    return plants


def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)
    output_dir = experiment_dir / "output"
    csv_path = output_dir / "tip_distances.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No tip_distances.csv found: {csv_path}\nRun 05_measure.py first.")

    plants = load_tip_distances(csv_path)
    all_labels = list(plants.keys())
    n_frames = len(plants[all_labels[0]]["frames"]) if all_labels else 0

    # Apply exclude list
    exclude_set = set(e.lower() for e in EXCLUDE_PLANTS)
    included = [l for l in all_labels if l.lower() not in exclude_set]
    excluded = [l for l in all_labels if l.lower() in exclude_set]

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Plants: {len(all_labels)}, Frames: {n_frames}")
    if excluded:
        print(f"Excluded: {', '.join(excluded)}")
    print(f"Exporting: {len(included)} plants")

    if not included:
        print("No plants to export.")
        return

    # Sort: by genotype, then replicate
    included.sort(key=lambda l: (plants[l]["genotype"], plants[l]["replicate"]))

    # Build column names: GenotypeName_N (numbered per genotype)
    geno_counter: dict[int, int] = defaultdict(int)
    columns = []
    for label in included:
        geno = plants[label]["genotype"]
        geno_counter[geno] += 1
        geno_name = GENOTYPE_NAMES.get(geno, f"G{geno}")
        columns.append(f"{geno_name}_{geno_counter[geno]}")

    # Time axis in hours
    hours = [i * INTERVAL_MIN / 60.0 for i in range(n_frames)]

    # Write CSV
    export_path = output_dir / "approved_distances.csv"
    with open(export_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + columns)
        for i in range(n_frames):
            row = [f"{hours[i]:.1f}"]
            for label in included:
                val = plants[label]["frames"][i]
                row.append(f"{val:.3f}" if val == val else "")  # nan check
            writer.writerow(row)

    print(f"\nExported -> {export_path}")
    print(f"Columns: time, {', '.join(columns)}")


if __name__ == "__main__":
    main()

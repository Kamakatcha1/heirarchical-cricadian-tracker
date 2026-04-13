# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# ============================================================
_OVERRIDE = {}
# ============================================================

import csv
from collections import defaultdict
from pathlib import Path

import xlwt

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_ID      = _config.EXPERIMENT_ID
EXPERIMENT_DIR     = _config.EXPERIMENT_DIR
EXPERIMENTS_DIR    = _config.EXPERIMENTS_DIR
INTERVAL_MIN       = _config.INTERVAL_MIN
GENOTYPE_NAMES     = _config.GENOTYPE_NAMES
EXCLUDE_PLANTS     = _config.EXCLUDE_PLANTS
MEASURE_EXPERIMENTS = _config.MEASURE_EXPERIMENTS


def load_tip_distances(csv_path: Path, exp_id: str) -> dict[str, dict]:
    """Load tip_distances.csv, group by plant label.
    Prefixes labels with exp_id to avoid collisions across experiments.
    """
    plants: dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row["label"]
            label = f"{exp_id} | {raw_label}"
            if label not in plants:
                plants[label] = {
                    "genotype": int(row["genotype"]),
                    "replicate": row["replicate"],
                    "experiment_id": exp_id,
                    "raw_label": raw_label,
                    "frames": [],
                }
            val = row["tip_distance_px"]
            plants[label]["frames"].append(float(val) if val != "nan" else float("nan"))
    return plants


def load_tip_distances_single(csv_path: Path) -> dict[str, dict]:
    """Load tip_distances.csv for single-experiment mode (no prefix)."""
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


def is_excluded(label: str, raw_label: str, exp_id: str,
                exclude_cfg: dict | list) -> bool:
    """Check if a plant should be excluded.

    Supports old list format:  ["g4_2", "g4_8"]
    And new dict format:       {"*": ["g3_*", "g4_2"], "F2_2_1": ["g4_14"]}
        Keys are experiment IDs ("*" = all experiments).
        Values are label patterns:
            "g4_2"  = exact match (genotype 4, replicate 2)
            "g3_*"  = wildcard (all of genotype 3)
    """
    if isinstance(exclude_cfg, list):
        return label.lower() in set(e.lower() for e in exclude_cfg)

    # raw_label is e.g. "g4_2"
    # Extract genotype prefix (e.g. "g4") and replicate (e.g. "2")
    parts = raw_label.split("_", 1)
    raw_geno = parts[0] if parts else ""       # "g4"

    for cfg_exp, patterns in exclude_cfg.items():
        if cfg_exp != "*" and cfg_exp != exp_id:
            continue
        for pat in patterns:
            pat = str(pat)
            if pat.endswith("_*"):
                # Wildcard: match genotype prefix, e.g. "g3_*" matches any g3 plant
                if raw_geno == pat[:-2]:
                    return True
            else:
                # Exact match: e.g. "g4_2" matches "g4_2"
                if raw_label == pat:
                    return True
    return False


def write_xls(export_path: Path, hours: list[float], columns: list[str],
              included: list[str], plants: dict[str, dict]) -> None:
    """Write approved distances to Excel 2003 (.xls) format."""
    wb = xlwt.Workbook()
    ws = wb.add_sheet("approved_distances")

    # Header row
    ws.write(0, 0, "time")
    for col_idx, col_name in enumerate(columns, start=1):
        ws.write(0, col_idx, col_name)

    # Data rows
    for i, hour in enumerate(hours):
        ws.write(i + 1, 0, round(hour, 1))
        for col_idx, label in enumerate(included, start=1):
            val = plants[label]["frames"][i]
            if val == val:  # not NaN
                ws.write(i + 1, col_idx, round(val, 3))
            # NaN cells left empty

    wb.save(str(export_path))


def export_single(experiment_dir: Path) -> None:
    """Original single-experiment export."""
    output_dir = experiment_dir / "output"
    csv_path = output_dir / "tip_distances.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No tip_distances.csv found: {csv_path}\nRun 05_measure.py first.")

    plants = load_tip_distances_single(csv_path)
    all_labels = list(plants.keys())
    n_frames = len(plants[all_labels[0]]["frames"]) if all_labels else 0

    # Apply exclude list
    included = []
    excluded = []
    for l in all_labels:
        if is_excluded(l, l, EXPERIMENT_ID, EXCLUDE_PLANTS):
            excluded.append(l)
        else:
            included.append(l)

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
    export_csv = output_dir / "approved_distances.csv"
    with open(export_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + columns)
        for i in range(n_frames):
            row = [f"{hours[i]:.1f}"]
            for label in included:
                val = plants[label]["frames"][i]
                row.append(f"{val:.3f}" if val == val else "")  # nan check
            writer.writerow(row)

    # Write XLS
    export_xls = output_dir / "approved_distances.xls"
    write_xls(export_xls, hours, columns, included, plants)

    print(f"\nExported -> {export_csv}")
    print(f"Exported -> {export_xls}")
    print(f"Columns: time, {', '.join(columns)}")


def export_multi(exp_ids: list[str], experiments_dir: Path) -> None:
    """Multi-experiment export: combine all experiments into one output."""
    all_plants: dict[str, dict] = {}

    for exp_id in exp_ids:
        csv_path = experiments_dir / exp_id / "output" / "tip_distances.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No tip_distances.csv found: {csv_path}\nRun 05_measure.py first."
            )
        exp_plants = load_tip_distances(csv_path, exp_id)
        all_plants.update(exp_plants)

    all_labels = list(all_plants.keys())
    n_frames = len(all_plants[all_labels[0]]["frames"]) if all_labels else 0

    # Apply exclude list
    included = []
    excluded = []
    for l in all_labels:
        raw = all_plants[l].get("raw_label", l)
        exp = all_plants[l].get("experiment_id", "")
        if is_excluded(l, raw, exp, EXCLUDE_PLANTS):
            excluded.append(l)
        else:
            included.append(l)

    print(f"Experiments: {exp_ids}")
    print(f"Plants: {len(all_labels)}, Frames: {n_frames}")
    if excluded:
        print(f"Excluded: {', '.join(excluded)}")
    print(f"Exporting: {len(included)} plants")

    if not included:
        print("No plants to export.")
        return

    # Sort: by experiment, then genotype, then replicate
    included.sort(key=lambda l: (
        all_plants[l]["experiment_id"],
        all_plants[l]["genotype"],
        all_plants[l]["replicate"],
    ))

    # Build column names: ExpID_GenotypeName_N (numbered per experiment+genotype)
    geno_counter: dict[str, int] = defaultdict(int)
    columns = []
    for label in included:
        exp_id = all_plants[label]["experiment_id"]
        geno = all_plants[label]["genotype"]
        key = f"{exp_id}_{geno}"
        geno_counter[key] += 1
        geno_name = GENOTYPE_NAMES.get(geno, f"G{geno}")
        columns.append(f"{exp_id} | {geno_name}_{geno_counter[key]}")

    # Time axis in hours
    hours = [i * INTERVAL_MIN / 60.0 for i in range(n_frames)]

    # Output to first experiment's output/combined/
    combined_dir = experiments_dir / exp_ids[0] / "output" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV
    export_csv = combined_dir / "approved_distances.csv"
    with open(export_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + columns)
        for i in range(n_frames):
            row = [f"{hours[i]:.1f}"]
            for label in included:
                val = all_plants[label]["frames"][i]
                row.append(f"{val:.3f}" if val == val else "")  # nan check
            writer.writerow(row)

    # Write XLS
    export_xls = combined_dir / "approved_distances.xls"
    write_xls(export_xls, hours, columns, included, all_plants)

    print(f"\nExported -> {export_csv}")
    print(f"Exported -> {export_xls}")
    print(f"Columns: time, {', '.join(columns)}")


def main() -> None:
    exp_ids = MEASURE_EXPERIMENTS if MEASURE_EXPERIMENTS else []

    if len(exp_ids) > 1:
        export_multi(exp_ids, Path(EXPERIMENTS_DIR))
    else:
        export_single(Path(EXPERIMENT_DIR))


if __name__ == "__main__":
    main()

# ============================================================
# Helper -- takes approved_distances.csv and removes columns
# matching EXCLUDE_SAMPLES (the same Biodare exclusion list).
# Outputs filtered_distances.csv in the same format.
# ============================================================
_OVERRIDE = {}
# ============================================================

import csv
from pathlib import Path

import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_DIR  = _config.EXPERIMENT_DIR
EXCLUDE_SAMPLES = _config.EXCLUDE_SAMPLES


def main() -> None:
    output_dir = Path(EXPERIMENT_DIR) / "output"
    approved_path = output_dir / "approved_distances.csv"

    if not approved_path.exists():
        raise FileNotFoundError(
            f"No approved_distances.csv found: {approved_path}\n"
            "Run 06_export.py first.")

    with open(approved_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Columns after "time" are samples 1, 2, 3, ...
    # Keep columns whose 1-based sample number is NOT in EXCLUDE_SAMPLES
    exclude_set = set(EXCLUDE_SAMPLES)
    keep = [0]  # always keep the time column
    for i in range(1, len(header)):
        sample_num = i  # 1-based
        if sample_num not in exclude_set:
            keep.append(i)

    kept_names = [header[i] for i in keep[1:]]
    removed = [header[i] for i in range(1, len(header)) if i in exclude_set]

    print(f"Input     : {approved_path.name}")
    print(f"Columns   : {len(header) - 1}")
    if removed:
        print(f"Removed   : {len(removed)} ({', '.join(removed)})")
    print(f"Remaining : {len(kept_names)}")

    # Write filtered CSV
    filtered_path = output_dir / "filtered_distances.csv"
    with open(filtered_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header[i] for i in keep])
        for row in rows:
            writer.writerow([row[i] for i in keep])

    print(f"\nExported -> {filtered_path}")


if __name__ == "__main__":
    main()

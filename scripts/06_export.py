from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import xlwt

import hct_runtime as rt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export approved distances across measured datasets.")
    parser.add_argument("--batch", action="store_true", help="Disable prompts and require CLI values.")
    parser.add_argument("--datasets", type=rt.parse_dataset_ids_arg, help="Comma-separated dataset ids to export.")
    parser.add_argument(
        "--genotype-names",
        type=rt.parse_name_mapping,
        help="Comma-separated genotype mappings like 0=WT,4=F2.",
    )
    parser.add_argument("--exclude-plants", help='Comma-separated exclusions like "g4_2,g3_*".')
    parser.add_argument("--exclude-genotypes", type=rt.parse_int_csv, help="Comma-separated genotype ids to exclude.")
    parser.add_argument("--interval-min", type=int, help="Minutes between frames (default: 30).")
    return parser


def load_tip_distances(csv_path: Path, dataset_id: str) -> dict[str, dict[str, Any]]:
    plants: dict[str, dict[str, Any]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row["label"]
            label = f"{dataset_id} | {raw_label}"
            if label not in plants:
                plants[label] = {
                    "genotype": int(row["genotype"]),
                    "replicate": row["replicate"],
                    "dataset_id": dataset_id,
                    "raw_label": raw_label,
                    "frames": [],
                }
            val = row["tip_distance_px"]
            plants[label]["frames"].append(float(val) if val != "nan" else float("nan"))
    return plants


def is_excluded(raw_label: str, patterns: list[str]) -> bool:
    parts = raw_label.split("_", 1)
    raw_geno = parts[0] if parts else ""
    for pattern in patterns:
        if pattern.endswith("_*"):
            if raw_geno == pattern[:-2]:
                return True
        elif raw_label == pattern:
            return True
    return False


def write_xls(export_path: Path, hours: list[float], columns: list[str], included: list[str], plants: dict[str, dict[str, Any]]) -> None:
    wb = xlwt.Workbook()
    ws = wb.add_sheet("approved_distances")
    ws.write(0, 0, "time")
    for col_idx, col_name in enumerate(columns, start=1):
        ws.write(0, col_idx, col_name)
    for row_idx, hour in enumerate(hours, start=1):
        ws.write(row_idx, 0, round(hour, 1))
        for col_idx, label in enumerate(included, start=1):
            val = plants[label]["frames"][row_idx - 1]
            if val == val:
                ws.write(row_idx, col_idx, round(val, 3))
    wb.save(str(export_path))


def resolve_datasets(args: argparse.Namespace) -> list[rt.DatasetInfo]:
    datasets = [info for info in rt.discover_datasets() if info.has_measurements]
    if args.datasets:
        chosen = []
        by_id = {info.dataset_id: info for info in datasets}
        missing = [ds for ds in args.datasets if ds not in by_id]
        if missing:
            raise SystemExit(f"Datasets not found or missing measurements: {', '.join(missing)}")
        for ds_id in args.datasets:
            chosen.append(by_id[ds_id])
        return chosen
    if args.batch:
        raise SystemExit("--batch requires --datasets")

    def render(info: rt.DatasetInfo) -> str:
        plants, frames = rt.count_tip_distance_rows(info.output_dir / "tip_distances.csv")
        return f"{info.dataset_id} ({plants} plants, {frames} frames)"

    return rt.prompt_select_many("Available datasets (with measurements):", datasets, render)


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()
    datasets = resolve_datasets(args)

    genotype_names = args.genotype_names
    if genotype_names is None:
        stored = rt.get_default("genotype_names", rt.EXPORT_DEFAULTS["genotype_names"])
        genotype_names = {int(k): v for k, v in stored.items()} if isinstance(stored, dict) else rt.EXPORT_DEFAULTS["genotype_names"]

    exclude_text = args.exclude_plants
    if exclude_text is None:
        exclude_text = ""
    exclude_patterns = rt.parse_csv_items(exclude_text)
    exclude_genotypes = args.exclude_genotypes or []

    interval_min = args.interval_min
    if interval_min is None:
        interval_min = rt.get_default("interval_min", rt.EXPORT_DEFAULTS["interval_min"])

    all_plants: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        csv_path = dataset.output_dir / "tip_distances.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No tip_distances.csv found for dataset {dataset.dataset_id}: {csv_path}")
        all_plants.update(load_tip_distances(csv_path, dataset.dataset_id))

    all_labels = list(all_plants.keys())
    if not all_labels:
        raise SystemExit("No measurement rows found to export.")
    n_frames = len(all_plants[all_labels[0]]["frames"])

    included = []
    excluded = []
    for label in all_labels:
        raw_label = all_plants[label]["raw_label"]
        if all_plants[label]["genotype"] in exclude_genotypes or is_excluded(raw_label, exclude_patterns):
            excluded.append(label)
        else:
            included.append(label)

    if not included:
        raise SystemExit("No plants left after applying exclusions.")

    included.sort(key=lambda label: (all_plants[label]["dataset_id"], all_plants[label]["genotype"], all_plants[label]["replicate"]))
    multi_dataset = len({all_plants[label]["dataset_id"] for label in included}) > 1

    counters: dict[str, int] = defaultdict(int)
    columns = []
    for label in included:
        dataset_id = all_plants[label]["dataset_id"]
        genotype = all_plants[label]["genotype"]
        key = f"{dataset_id}_{genotype}" if multi_dataset else str(genotype)
        counters[key] += 1
        genotype_name = genotype_names.get(genotype, f"G{genotype}")
        if multi_dataset:
            columns.append(f"{dataset_id} | {genotype_name}_{counters[key]}")
        else:
            columns.append(f"{genotype_name}_{counters[key]}")

    hours = [idx * interval_min / 60.0 for idx in range(n_frames)]

    export_root = rt.exports_dir()
    export_root.mkdir(parents=True, exist_ok=True)
    export_csv = export_root / "approved_distances.csv"
    export_xls = export_root / "approved_distances.xls"
    with open(export_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + columns)
        for idx in range(n_frames):
            row = [f"{hours[idx]:.1f}"]
            for label in included:
                val = all_plants[label]["frames"][idx]
                row.append(f"{val:.3f}" if val == val else "")
            writer.writerow(row)

    write_xls(export_xls, hours, columns, included, all_plants)

    metadata = {
        "exported": datetime.now().isoformat(timespec="seconds"),
        "datasets": [dataset.dataset_id for dataset in datasets],
        "genotype_names": genotype_names,
        "exclude_patterns": exclude_patterns,
        "exclude_genotypes": exclude_genotypes,
        "interval_min": interval_min,
        "columns": columns,
        "source_files": {dataset.dataset_id: str(dataset.output_dir / "tip_distances.csv") for dataset in datasets},
    }
    rt.write_json(export_root / "export_metadata.json", metadata)

    print(f"Exported: {export_csv}")
    print(f"Exported: {export_xls}")
    print(f"Metadata: {export_root / 'export_metadata.json'}")
    if excluded:
        print(f"Excluded: {', '.join(excluded)}")

    rt.save_defaults_after_success(
        genotype_names=genotype_names,
        interval_min=interval_min,
    )


if __name__ == "__main__":
    main()

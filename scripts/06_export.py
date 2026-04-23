from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import xlwt

import hct_runtime as rt


@dataclass
class PlantRunData:
    dataset_id: str
    raw_label: str
    genotype: int
    replicate: int
    run: rt.MeasurementRunInfo
    frames: list[float]


@dataclass
class PlantSelection:
    dataset_id: str
    raw_label: str
    genotype: int
    replicate: int
    runs: dict[str, PlantRunData]
    selected_run_name: str
    include: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export approved distances across measured datasets.")
    parser.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--datasets", type=rt.parse_dataset_ids_arg, help="Comma-separated dataset ids to export.")
    parser.add_argument(
        "--outputs",
        type=rt.parse_string_mapping,
        help='Optional dataset-to-output mapping like "F2_001=my_model,F2_002=other_model".',
    )
    parser.add_argument(
        "--genotype-names",
        type=rt.parse_name_mapping,
        help="Comma-separated genotype mappings like 0=WT,4=F2.",
    )
    parser.add_argument("--exclude-plants", help='Comma-separated exclusions like "g4_2,g3_*".')
    parser.add_argument("--exclude-genotypes", type=rt.parse_int_csv, help="Comma-separated genotype ids to exclude.")
    parser.add_argument("--interval-min", type=int, help="Minutes between frames (default: 30).")
    parser.add_argument("--max-hours", type=float, help="Optional export cutoff in hours (0 = all).")
    parser.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)
    return parser


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


def write_xls(export_path: Path, hours: list[float], columns: list[str], included: list[PlantRunData], n_frames: int) -> None:
    wb = xlwt.Workbook()
    ws = wb.add_sheet("approved_distances")
    ws.write(0, 0, "time")
    for col_idx, col_name in enumerate(columns, start=1):
        ws.write(0, col_idx, col_name)
    for row_idx, hour in enumerate(hours, start=1):
        ws.write(row_idx, 0, round(hour, 1))
        for col_idx, plant in enumerate(included, start=1):
            val = plant.frames[row_idx - 1]
            if row_idx - 1 < n_frames and val == val:
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
        runs = rt.discover_measurement_runs(info)
        if len(runs) == 1:
            plants, frames = rt.count_tip_distance_rows(runs[0].csv_path)
            return f"{info.dataset_id} ({runs[0].name}: {plants} plants, {frames} frames)"
        return f"{info.dataset_id} ({len(runs)} outputs)"

    return rt.prompt_select_many("Available datasets (with measurements):", datasets, render)


def render_measurement_run(run: rt.MeasurementRunInfo) -> str:
    plants, frames = rt.count_tip_distance_rows(run.csv_path)
    measured_text = f", measured {run.measured}" if run.measured else ""
    if run.legacy:
        return f"legacy ({plants} plants, {frames} frames{measured_text})"
    return f"{run.name} ({plants} plants, {frames} frames{measured_text})"


def resolve_output_runs(
    args: argparse.Namespace,
    datasets: list[rt.DatasetInfo],
) -> list[tuple[rt.DatasetInfo, rt.MeasurementRunInfo]]:
    requested = args.outputs or {}
    selections: list[tuple[rt.DatasetInfo, rt.MeasurementRunInfo]] = []

    for dataset in datasets:
        runs = rt.discover_measurement_runs(dataset)
        if not runs:
            raise FileNotFoundError(f"No measurement outputs found for dataset {dataset.dataset_id}")

        chosen: rt.MeasurementRunInfo | None = None
        requested_name = requested.get(dataset.dataset_id)
        if requested_name is not None:
            chosen = next((run for run in runs if run.name == requested_name), None)
            if chosen is None:
                valid = ", ".join(run.name for run in runs)
                raise SystemExit(
                    f"Output '{requested_name}' not found for {dataset.dataset_id}. Available outputs: {valid}"
                )
        elif len(runs) == 1:
            chosen = runs[0]
            print(f"Using {dataset.dataset_id}: {render_measurement_run(chosen)}")
        elif args.batch:
            chosen = runs[0]
            print(f"Using first output for {dataset.dataset_id}: {render_measurement_run(chosen)}")
        else:
            chosen = runs[0]
            print(f"Defaulting {dataset.dataset_id} to first output for review: {render_measurement_run(chosen)}")
        selections.append((dataset, chosen))
    return selections


def load_tip_distances(run: rt.MeasurementRunInfo) -> dict[str, PlantRunData]:
    plants: dict[str, PlantRunData] = {}
    with open(run.csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = row["label"]
            if raw_label not in plants:
                plants[raw_label] = PlantRunData(
                    dataset_id=run.dataset_id,
                    raw_label=raw_label,
                    genotype=int(row["genotype"]),
                    replicate=int(row["replicate"]),
                    run=run,
                    frames=[],
                )
            val = row["tip_distance_px"]
            plants[raw_label].frames.append(float(val) if val != "nan" else float("nan"))
    return plants


def build_plant_selections(
    selected_runs: list[tuple[rt.DatasetInfo, rt.MeasurementRunInfo]],
    exclude_patterns: list[str],
    exclude_genotypes: list[int],
) -> list[PlantSelection]:
    default_run_by_dataset = {dataset.dataset_id: run.name for dataset, run in selected_runs}
    selected_dataset_ids = {dataset.dataset_id for dataset, _ in selected_runs}
    all_runs_by_dataset: dict[str, list[rt.MeasurementRunInfo]] = {
        dataset_id: rt.discover_measurement_runs(dataset_id) for dataset_id in selected_dataset_ids
    }

    selections: list[PlantSelection] = []
    for dataset_id in sorted(selected_dataset_ids):
        grouped_runs: dict[str, dict[str, PlantRunData]] = {}
        for run in all_runs_by_dataset[dataset_id]:
            for raw_label, plant in load_tip_distances(run).items():
                grouped_runs.setdefault(raw_label, {})[run.name] = plant

        for raw_label, run_map in sorted(grouped_runs.items(), key=lambda item: (next(iter(item[1].values())).genotype, next(iter(item[1].values())).replicate)):
            sample = next(iter(run_map.values()))
            preferred_name = default_run_by_dataset.get(dataset_id)
            selected_run_name = preferred_name if preferred_name in run_map else sorted(run_map.keys())[0]
            include = sample.genotype not in exclude_genotypes and not is_excluded(raw_label, exclude_patterns)
            selections.append(
                PlantSelection(
                    dataset_id=dataset_id,
                    raw_label=raw_label,
                    genotype=sample.genotype,
                    replicate=sample.replicate,
                    runs=run_map,
                    selected_run_name=selected_run_name,
                    include=include,
                )
            )
    return selections


def choose_visual_selections(selections: list[PlantSelection]) -> list[PlantSelection]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The export GUI requires OpenCV (`cv2`) and NumPy. "
            "Please run export from the same GUI-capable environment you use for crop/annotate."
        ) from exc
    window = "HCT Export Review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 900)
    plot_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def normalize_image(image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        return None

    def load_plot(selection: PlantSelection) -> np.ndarray:
        chosen_run = selection.runs[selection.selected_run_name]
        cache_key = (selection.dataset_id, selection.raw_label, chosen_run.run.name)
        cached = plot_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        plot_path = chosen_run.run.plots_dir / f"{selection.raw_label}_tip_distance.png"
        if plot_path.exists():
            image = normalize_image(cv2.imread(str(plot_path), cv2.IMREAD_UNCHANGED))
            if image is not None:
                plot_cache[cache_key] = image
                return image.copy()
        fallback = np.full((720, 960, 3), 245, dtype=np.uint8)
        cv2.putText(
            fallback,
            f"No plot found for {chosen_run.run.name}",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        plot_cache[cache_key] = fallback
        return fallback

    def fit_plot(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return np.full((max_height, max_width, 3), 245, dtype=np.uint8)
        scale = min(max_width / width, max_height / height)
        if scale <= 0:
            scale = 1.0
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        return resized

    def render(selection_index: int) -> np.ndarray:
        selection = selections[selection_index]
        canvas = np.full((900, 1400, 3), 252, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (1399, 899), (225, 225, 225), 2)
        cv2.rectangle(canvas, (20, 20), (420, 880), (240, 240, 240), -1)
        cv2.rectangle(canvas, (440, 20), (1380, 880), (255, 255, 255), -1)

        header_lines = [
            "Export Review",
            f"Individual {selection_index + 1}/{len(selections)}",
            f"Dataset: {selection.dataset_id}",
            f"Label: {selection.raw_label}",
            f"Genotype: {selection.genotype}",
            f"Replicate: {selection.replicate}",
            f"Include: {'YES' if selection.include else 'NO'}",
            "",
            "Model outputs:",
        ]
        y = 60
        for line in header_lines:
            scale = 0.9 if line == "Export Review" else 0.65
            thickness = 2 if line == "Export Review" else 1
            cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thickness, cv2.LINE_AA)
            y += 42 if line == "Export Review" else 34

        run_names = sorted(selection.runs.keys())
        for idx, run_name in enumerate(run_names, start=1):
            active = run_name == selection.selected_run_name
            prefix = "[x]" if active else "[ ]"
            color = (10, 90, 180) if active else (60, 60, 60)
            cv2.putText(
                canvas,
                f"{idx}. {prefix} {run_name}",
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2 if active else 1,
                cv2.LINE_AA,
            )
            y += 30

        controls = [
            "",
            "Controls:",
            "Space = include/exclude",
            "[ / ] = previous/next model",
            "1-9 = choose model",
            "A = include all",
            "X = exclude all",
            "Left/Up/P = previous individual",
            "Right/Down/N = next individual",
            "Enter = finish",
            "Esc / Q = cancel",
        ]
        y += 14
        for line in controls:
            cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
            y += 28

        plot = fit_plot(load_plot(selection), 900, 820)
        plot_h, plot_w = plot.shape[:2]
        x0 = 440 + (940 - plot_w) // 2
        y0 = 40 + (820 - plot_h) // 2
        canvas[y0 : y0 + plot_h, x0 : x0 + plot_w] = plot
        return canvas

    index = 0
    prev_keys = {
        81,
        82,
        2424832,
        2490368,
        ord("h"),
        ord("H"),
        ord("k"),
        ord("K"),
        ord("p"),
        ord("P"),
    }
    next_keys = {
        83,
        84,
        2555904,
        2621440,
        ord("l"),
        ord("L"),
        ord("j"),
        ord("J"),
        ord("n"),
        ord("N"),
    }
    try:
        while True:
            try:
                frame = render(index)
                cv2.imshow(window, frame)
            except Exception as exc:
                raise RuntimeError(f"Failed to render export review window: {exc}") from exc

            key = cv2.waitKeyEx(0)
            try:
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1
            if visible < 1:
                raise SystemExit("Export cancelled.")

            selection = selections[index]
            run_names = sorted(selection.runs.keys())
            current_run_idx = run_names.index(selection.selected_run_name)

            if key in (13, 10):
                break
            if key in (27, ord("q"), ord("Q")):
                raise SystemExit("Export cancelled.")
            if key == ord(" "):
                selection.include = not selection.include
            elif key in prev_keys:
                index = (index - 1) % len(selections)
            elif key in next_keys:
                index = (index + 1) % len(selections)
            elif key == ord("["):
                selection.selected_run_name = run_names[(current_run_idx - 1) % len(run_names)]
            elif key == ord("]"):
                selection.selected_run_name = run_names[(current_run_idx + 1) % len(run_names)]
            elif ord("1") <= key <= ord("9"):
                numeric_idx = key - ord("1")
                if numeric_idx < len(run_names):
                    selection.selected_run_name = run_names[numeric_idx]
            elif key in (ord("a"), ord("A")):
                for item in selections:
                    item.include = True
            elif key in (ord("x"), ord("X")):
                for item in selections:
                    item.include = False
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

    return selections


def finalize_selected_plants(selections: list[PlantSelection]) -> list[PlantRunData]:
    included: list[PlantRunData] = []
    for selection in selections:
        if not selection.include:
            continue
        included.append(selection.runs[selection.selected_run_name])
    return included


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()
    datasets = resolve_datasets(args)
    selected_runs = resolve_output_runs(args, datasets)

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

    selections = build_plant_selections(selected_runs, exclude_patterns, exclude_genotypes)
    if not selections:
        raise SystemExit("No measured individuals found for export.")

    if not args.batch:
        selections = choose_visual_selections(selections)

    included = finalize_selected_plants(selections)
    if not included:
        raise SystemExit("No individuals selected for export.")

    max_hours = args.max_hours
    if max_hours is None and args.max_frames is not None:
        max_hours = args.max_frames * interval_min / 60.0
    if max_hours is None:
        if args.batch:
            max_hours = 0.0
        else:
            max_hours = rt.prompt_with_default(
                "Max hours to export (0 = all)",
                default=0.0,
                parser=float,
            )

    frame_lengths = {f"{plant.dataset_id}|{plant.raw_label}|{plant.run.name}": len(plant.frames) for plant in included}
    n_frames = min(frame_lengths.values())
    if max_hours and max_hours > 0:
        requested_frames = max(1, int(math.floor((max_hours * 60.0) / interval_min)) + 1)
        n_frames = min(n_frames, requested_frames)
    if len(set(frame_lengths.values())) > 1:
        print(f"WARNING: Measurement length mismatch across selected individuals; truncating export to {n_frames} frames")

    included.sort(key=lambda plant: (plant.dataset_id, plant.genotype, plant.replicate, plant.raw_label))
    multi_dataset = len({plant.dataset_id for plant in included}) > 1

    counters: dict[str, int] = defaultdict(int)
    columns = []
    for plant in included:
        key = f"{plant.dataset_id}_{plant.genotype}" if multi_dataset else str(plant.genotype)
        counters[key] += 1
        genotype_name = genotype_names.get(plant.genotype, f"G{plant.genotype}")
        if multi_dataset:
            columns.append(f"{plant.dataset_id} | {genotype_name}_{counters[key]}")
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
            for plant in included:
                val = plant.frames[idx]
                row.append(f"{val:.3f}" if val == val else "")
            writer.writerow(row)

    write_xls(export_xls, hours, columns, included, n_frames)

    metadata = {
        "exported": datetime.now().isoformat(timespec="seconds"),
        "datasets": [dataset.dataset_id for dataset in datasets],
        "genotype_names": genotype_names,
        "exclude_patterns": exclude_patterns,
        "exclude_genotypes": exclude_genotypes,
        "interval_min": interval_min,
        "max_hours": max_hours,
        "frames_exported": n_frames,
        "columns": columns,
        "selected_outputs": {
            plant.dataset_id: sorted({item.run.name for item in included if item.dataset_id == plant.dataset_id})
            for plant in included
        },
        "selected_individuals": [
            {
                "dataset_id": plant.dataset_id,
                "raw_label": plant.raw_label,
                "run_name": plant.run.name,
                "csv_path": str(plant.run.csv_path),
            }
            for plant in included
        ],
        "source_files": {f"{plant.dataset_id}:{plant.raw_label}": str(plant.run.csv_path) for plant in included},
    }
    rt.write_json(export_root / "export_metadata.json", metadata)

    print(f"Exported: {export_csv}")
    print(f"Exported: {export_xls}")
    print(f"Metadata: {export_root / 'export_metadata.json'}")

    rt.save_defaults_after_success(
        genotype_names=genotype_names,
        interval_min=interval_min,
        export_max_hours=max_hours,
    )


if __name__ == "__main__":
    main()

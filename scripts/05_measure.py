from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hct_runtime as rt


PEAK_THRESH = 0.05
MIN_DIST_FLOOR = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure leaf tip distance for one or more datasets.")
    parser.add_argument("--batch", action="store_true", help="Disable prompts and require CLI values.")
    parser.add_argument("--dataset", help="Single dataset id to measure.")
    parser.add_argument("--datasets", type=rt.parse_dataset_ids_arg, help="Comma-separated dataset ids to measure.")
    parser.add_argument("--model", help="Model folder name under data/models/.")
    parser.add_argument("--max-frames", type=int, help="Limit to first N frames (0 = all).")
    parser.add_argument("--num-tips", type=int, help="Number of peaks to detect.")
    parser.add_argument("--min-dist", type=int, help="Minimum distance between detected peaks in model pixels.")
    parser.add_argument("--interval-min", type=int, help="Minutes between frames (default: 30).")
    parser.add_argument("--genotype-filter", type=rt.parse_int_csv, help="Comma-separated genotype ids to keep.")
    return parser


def _find_candidates_at(heatmap: np.ndarray, max_n: int, min_dist: int) -> list[tuple[int, int, float]]:
    hmap = heatmap.copy()
    peaks = []
    for _ in range(max_n):
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        val = float(hmap[y, x])
        if val < PEAK_THRESH:
            break
        peaks.append((int(x), int(y), val))
        y0 = max(0, y - min_dist)
        y1 = min(hmap.shape[0], y + min_dist + 1)
        x0 = max(0, x - min_dist)
        x1 = min(hmap.shape[1], x + min_dist + 1)
        hmap[y0:y1, x0:x1] = 0.0
    return peaks


def find_candidates(heatmap: np.ndarray, max_n: int, min_dist: int) -> list[tuple[int, int, float]]:
    dist = min_dist
    while dist >= MIN_DIST_FLOOR:
        peaks = _find_candidates_at(heatmap, max_n, dist)
        if len(peaks) >= max_n:
            return peaks
        dist //= 2
    return _find_candidates_at(heatmap, max_n, MIN_DIST_FLOOR)


def find_peaks(
    heatmap: np.ndarray,
    n: int,
    min_dist: int,
    prev_peaks: list[tuple[int, int, float]] | None = None,
) -> list[tuple[int, int, float]]:
    candidates = find_candidates(heatmap, n * 2, min_dist)
    if len(candidates) <= n or prev_peaks is None:
        return candidates[:n]

    chosen = []
    remaining = list(candidates)
    for px, py, _ in prev_peaks:
        if not remaining:
            break
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (remaining[idx][0] - px) ** 2 + (remaining[idx][1] - py) ** 2,
        )
        chosen.append(remaining.pop(best_idx))
    return chosen


def resolve_datasets(args: argparse.Namespace) -> list[rt.DatasetInfo]:
    datasets = [info for info in rt.discover_datasets() if info.has_crops]
    requested_ids: list[str] = []
    if args.datasets:
        requested_ids.extend(args.datasets)
    if args.dataset:
        requested_ids.append(args.dataset)

    if requested_ids:
        by_id = {info.dataset_id: info for info in datasets}
        missing = [dataset_id for dataset_id in requested_ids if dataset_id not in by_id]
        if missing:
            raise SystemExit(f"Dataset not found or missing crops: {', '.join(missing)}")
        return [by_id[dataset_id] for dataset_id in requested_ids]

    if args.batch:
        raise SystemExit("--batch requires --dataset or --datasets")

    last_dataset = rt.get_default("last_dataset")
    default_index = next((idx for idx, info in enumerate(datasets) if info.dataset_id == last_dataset), None)
    default_indices = [default_index] if default_index is not None else None

    def render(info: rt.DatasetInfo) -> str:
        return f"{info.dataset_id} ({info.plant_count} plants, {len(info.frames)} frames)"

    return rt.prompt_select_many("Available datasets (with crops):", datasets, render, default_indices=default_indices)


def resolve_model(args: argparse.Namespace) -> rt.ModelInfo:
    models = rt.discover_models()
    if args.model:
        matches = [info for info in models if info.name == args.model]
        if not matches:
            raise SystemExit(f"Model not found: {args.model}")
        return matches[0]
    if args.batch:
        raise SystemExit("--batch requires --model")

    last_model = rt.get_default("last_model")
    default_index = next((idx for idx, info in enumerate(models) if info.name == last_model), None)

    def render(info: rt.ModelInfo) -> str:
        metric = f"val_dice: {info.val_dice:.3f}" if info.val_dice is not None else "val_dice: n/a"
        return f"{info.name} ({metric})"

    return rt.prompt_select_one("Available models:", models, render, default_index=default_index)


def measure_dataset(
    dataset: rt.DatasetInfo,
    model: tf.keras.Model,
    model_name: str,
    *,
    img_size: int,
    max_frames: int,
    num_tips: int,
    min_dist: int,
    interval_min: int,
    genotype_filter: list[int],
) -> None:
    crop_log = rt.load_json(dataset.crop_path, {})
    raw_dir = dataset.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    frames = list(crop_log.get("frames", []))
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]

    plants = list(crop_log.get("plants", []))
    if genotype_filter:
        plants = [plant for plant in plants if int(plant.get("genotype", -1)) in genotype_filter]

    if not frames:
        raise ValueError(f"No frames in crops.json for {dataset.dataset_id}.")
    if not plants:
        raise ValueError(f"No plants in crops.json for {dataset.dataset_id} after filtering.")

    output_dir = dataset.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[list[Any]] = []
    print(f"Running inference on {dataset.dataset_id} with model {model_name}...")
    print(f"Frames: {len(frames)}, Plants: {len(plants)}, Interval: {interval_min} min")
    if genotype_filter:
        print(f"Genotype filter: {genotype_filter}")

    for plant in plants:
        plant_id = plant["id"]
        genotype = plant["genotype"]
        replicate = plant["replicate"]
        x0, y0, x1, y1 = plant["bbox"]
        label = f"g{genotype}_{replicate}"

        print(f"\n  {label} ({plant_id})...")
        distances = []
        prev_peaks: list[tuple[int, int, float]] | None = None

        for frame_idx, frame_name in enumerate(frames):
            frame_path = raw_dir / frame_name
            if not frame_path.exists():
                distances.append(float("nan"))
                all_rows.append([dataset.dataset_id, plant_id, label, genotype, replicate, frame_idx, frame_name, "nan"])
                continue

            full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if full_img is None:
                distances.append(float("nan"))
                all_rows.append([dataset.dataset_id, plant_id, label, genotype, replicate, frame_idx, frame_name, "nan"])
                continue

            crop = full_img[y0:y1, x0:x1]
            h0, w0 = crop.shape[:2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            crop_norm = (crop_resized.astype(np.float32) / 255.0)[np.newaxis, ...]

            pred = model.predict(crop_norm, verbose=0)
            heatmap = pred[0, :, :, 0].astype(np.float32)
            peaks = find_peaks(heatmap, num_tips, min_dist, prev_peaks)
            prev_peaks = peaks

            sx = (w0 - 1) / (img_size - 1) if img_size > 1 else 1.0
            sy = (h0 - 1) / (img_size - 1) if img_size > 1 else 1.0
            pts = []
            for px, py, _ in peaks:
                ox = int(round(px * sx))
                oy = int(round(py * sy))
                ox = max(0, min(w0 - 1, ox))
                oy = max(0, min(h0 - 1, oy))
                pts.append((ox, oy))

            if len(pts) < 2:
                dist = float("nan")
            else:
                dx = pts[0][0] - pts[1][0]
                dy = pts[0][1] - pts[1][1]
                dist = float(np.hypot(dx, dy))

            distances.append(dist)
            all_rows.append([dataset.dataset_id, plant_id, label, genotype, replicate, frame_idx, frame_name, f"{dist:.3f}"])

        times_min = np.arange(len(distances)) * interval_min
        plt.figure(figsize=(10, 4))
        plt.plot(times_min, distances, marker=".", markersize=3, linewidth=1)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Tip distance (px)")
        plt.title(f"Leaf tip distance - {dataset.dataset_id} / {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = plots_dir / f"{label}_tip_distance.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"    {len(distances)} frames, plot -> {plot_path.name}")

    csv_path = output_dir / "tip_distances.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_id", "plant_id", "label", "genotype", "replicate", "frame_index", "frame_filename", "tip_distance_px"])
        writer.writerows(all_rows)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plots_dir}")


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()

    datasets = resolve_datasets(args)
    model_info = resolve_model(args)

    max_frames = args.max_frames
    if max_frames is None:
        max_frames = rt.get_default("measure_max_frames", rt.MEASURE_DEFAULTS["max_frames"])

    num_tips = args.num_tips
    if num_tips is None:
        num_tips = rt.get_default("num_tips", rt.MEASURE_DEFAULTS["num_tips"])

    min_dist = args.min_dist
    if min_dist is None:
        min_dist = rt.MEASURE_DEFAULTS["min_dist"]

    interval_min = args.interval_min
    if interval_min is None:
        interval_min = rt.get_default("interval_min", rt.MEASURE_DEFAULTS["interval_min"])

    genotype_filter = args.genotype_filter or []

    model = tf.keras.models.load_model(str(model_info.model_path), compile=False)
    img_size = int(model.input_shape[1] if model.input_shape[1] is not None else rt.TRAIN_DEFAULTS["img_size"])

    for dataset in datasets:
        measure_dataset(
            dataset,
            model,
            model_info.name,
            img_size=img_size,
            max_frames=max_frames,
            num_tips=num_tips,
            min_dist=min_dist,
            interval_min=interval_min,
            genotype_filter=genotype_filter,
        )

    rt.save_defaults_after_success(
        last_dataset=datasets[-1].dataset_id,
        last_model=model_info.name,
        interval_min=interval_min,
        num_tips=num_tips,
        measure_max_frames=max_frames,
    )


if __name__ == "__main__":
    main()

# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# ============================================================
_OVERRIDE = {}
# ============================================================

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_ID  = _config.EXPERIMENT_ID
EXPERIMENT_DIR = _config.EXPERIMENT_DIR
EXPERIMENTS_DIR = _config.EXPERIMENTS_DIR
MODEL_PATH     = _config.MODEL_PATH
IMG_SIZE       = _config.IMG_SIZE
NUM_TIPS       = _config.NUM_TIPS
MIN_DIST       = _config.MIN_DIST
TRACK_RADIUS   = _config.TRACK_RADIUS
INTERVAL_MIN   = _config.INTERVAL_MIN


# ---- Peak detection ----


def find_peaks_global(heatmap: np.ndarray, n: int, min_dist: int) -> list[tuple[int, int, float]]:
    """Find the n strongest peaks in the heatmap, each separated by at least min_dist."""
    hmap = heatmap.copy()
    peaks = []
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        val = float(hmap[y, x])
        peaks.append((int(x), int(y), val))
        # Zero out area around found peak
        y0 = max(0, y - min_dist)
        y1 = min(hmap.shape[0], y + min_dist + 1)
        x0 = max(0, x - min_dist)
        x1 = min(hmap.shape[1], x + min_dist + 1)
        hmap[y0:y1, x0:x1] = 0.0
    return peaks


def find_peak_near(heatmap: np.ndarray, prev_x: int, prev_y: int,
                   radius: int) -> tuple[int, int, float]:
    """Find the strongest peak within radius of (prev_x, prev_y)."""
    h, w = heatmap.shape
    r = radius if radius > 0 else max(h, w) // 2
    y0 = max(0, prev_y - r)
    y1 = min(h, prev_y + r + 1)
    x0 = max(0, prev_x - r)
    x1 = min(w, prev_x + r + 1)
    region = heatmap[y0:y1, x0:x1]
    ry, rx = np.unravel_index(np.argmax(region), region.shape)
    val = float(region[ry, rx])
    return (int(rx + x0), int(ry + y0), val)


def find_peaks_tracked(heatmap: np.ndarray, prev_peaks: list[tuple[int, int, float]],
                       radius: int, min_dist: int) -> list[tuple[int, int, float]]:
    """Find peaks near previous frame's peak locations.

    Searches locally around each previous peak. Zeros out found peaks
    to prevent two tracks from collapsing onto the same peak.
    """
    hmap = heatmap.copy()
    peaks = []
    for px, py, _ in prev_peaks:
        x, y, val = find_peak_near(hmap, px, py, radius)
        peaks.append((x, y, val))
        # Zero out so next track can't land on same peak
        zy0 = max(0, y - min_dist)
        zy1 = min(hmap.shape[0], y + min_dist + 1)
        zx0 = max(0, x - min_dist)
        zx1 = min(hmap.shape[1], x + min_dist + 1)
        hmap[zy0:zy1, zx0:zx1] = 0.0
    return peaks


# ---- Main ----

def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)

    # Load crop log for this experiment
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop log not found: {crop_path}\nRun 01_crop.py first.")
    with open(crop_path) as f:
        crop_log = json.load(f)

    # Validate experiment ID
    log_eid = crop_log.get("experiment_id", "")
    if log_eid and log_eid != EXPERIMENT_ID:
        raise RuntimeError(
            f"Experiment ID mismatch! Config: {EXPERIMENT_ID}, JSON: {log_eid}"
        )

    # Find model -- use MODEL_PATH if set, otherwise find latest run's best.keras
    if MODEL_PATH:
        model_file = Path(MODEL_PATH)
    else:
        models_root = experiment_dir / "models"
        if not models_root.exists():
            raise FileNotFoundError(f"No models folder: {models_root}\nRun 04_train.py first.")
        runs = sorted([d for d in models_root.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No training runs in {models_root}\nRun 04_train.py first.")
        model_file = runs[-1] / "best.keras"  # latest run
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}\nRun 04_train.py first.")

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Model: {model_file}")

    model = tf.keras.models.load_model(str(model_file), compile=False)

    # Determine model input size
    shp = model.input_shape
    img_size = shp[1] if shp[1] is not None else IMG_SIZE
    print(f"Model input size: {img_size}")

    # Get raw dir and frames from crop log
    raw_dir = Path(crop_log["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    frames = crop_log.get("frames", [])
    plants = crop_log.get("plants", [])
    if not frames:
        raise ValueError("No frames in crop log.")
    if not plants:
        raise ValueError("No plants in crop log.")

    print(f"Frames: {len(frames)}, Plants: {len(plants)}")

    # Output directory
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each plant across all frames
    all_rows = []
    n_tips = NUM_TIPS

    for plant in plants:
        plant_id = plant["id"]
        genotype = plant["genotype"]
        replicate = plant["replicate"]
        bbox = plant["bbox"]
        x0, y0, x1, y1 = bbox
        label = f"g{genotype}_{replicate}"

        print(f"\n  {label} ({plant_id})...")
        distances = []
        prev_peaks = None  # for tracking across frames

        for frame_idx, frame_name in enumerate(frames):
            frame_path = raw_dir / frame_name
            if not frame_path.exists():
                distances.append(float("nan"))
                all_rows.append([EXPERIMENT_ID, plant_id, label, genotype, replicate, frame_idx, frame_name, "nan"])
                continue

            full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if full_img is None:
                distances.append(float("nan"))
                all_rows.append([EXPERIMENT_ID, plant_id, label, genotype, replicate, frame_idx, frame_name, "nan"])
                continue

            # Virtual crop
            crop = full_img[y0:y1, x0:x1]
            h0, w0 = crop.shape[:2]

            # Resize for model
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            crop_norm = (crop_resized.astype(np.float32) / 255.0)[np.newaxis, ...]

            # Inference
            pred = model.predict(crop_norm, verbose=0)
            heatmap = pred[0, :, :, 0].astype(np.float32)

            # Find peaks -- first frame uses global search, subsequent frames track
            if prev_peaks is None:
                peaks = find_peaks_global(heatmap, n_tips, min_dist=MIN_DIST)
            else:
                peaks = find_peaks_tracked(heatmap, prev_peaks, radius=TRACK_RADIUS, min_dist=MIN_DIST)
            prev_peaks = peaks

            # Scale back to original crop dimensions
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
            all_rows.append([EXPERIMENT_ID, plant_id, label, genotype, replicate, frame_idx, frame_name, f"{dist:.3f}"])

        # Plot this plant
        times_min = np.arange(len(distances)) * INTERVAL_MIN
        plt.figure(figsize=(10, 4))
        plt.plot(times_min, distances, marker=".", markersize=3, linewidth=1)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Tip distance (px)")
        plt.title(f"Leaf tip distance - {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / f"{label}_tip_distance.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"    {len(distances)} frames, plot -> {plot_path.name}")

    # Save CSV
    csv_path = output_dir / "tip_distances.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "plant_id", "label", "genotype", "replicate", "frame_index", "frame_filename", "tip_distance_px"])
        writer.writerows(all_rows)

    print(f"\n{'='*50}")
    print(f"CSV -> {csv_path}")
    print(f"Plots -> {output_dir}")
    print(f"Plants measured: {len(plants)}, Frames per plant: {len(frames)}")


if __name__ == "__main__":
    main()

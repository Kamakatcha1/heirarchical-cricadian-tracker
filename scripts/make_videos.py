# ============================================================
# Helper -- generate crop journey + overlay diagnostic videos
# for every replicate.  Not part of the numbered pipeline.
# Usage: just run it.  Outputs to {EXPERIMENT_DIR}/output/videos/
# ============================================================
_OVERRIDE = {}
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

EXPERIMENT_ID  = _config.EXPERIMENT_ID
EXPERIMENT_DIR = _config.EXPERIMENT_DIR
MODEL_PATH     = _config.MODEL_PATH
IMG_SIZE       = _config.IMG_SIZE
NUM_TIPS       = _config.NUM_TIPS
MIN_DIST       = _config.MIN_DIST

PEAK_THRESH    = 0.05
MIN_DIST_FLOOR = 5
VIDEO_FPS      = 15
DISPLAY_SCALE  = 3


# -- peak detection (same as check_model.py / 05_measure.py) --

def _find_candidates_at(heatmap, max_n, min_dist):
    hmap = heatmap.copy()
    peaks = []
    for _ in range(max_n):
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        val = float(hmap[y, x])
        if val < PEAK_THRESH:
            break
        peaks.append((int(x), int(y), val))
        y0, y1 = max(0, y - min_dist), min(hmap.shape[0], y + min_dist + 1)
        x0, x1 = max(0, x - min_dist), min(hmap.shape[1], x + min_dist + 1)
        hmap[y0:y1, x0:x1] = 0.0
    return peaks


def find_candidates(heatmap, max_n, min_dist):
    dist = min_dist
    while dist >= MIN_DIST_FLOOR:
        peaks = _find_candidates_at(heatmap, max_n, dist)
        if len(peaks) >= max_n:
            return peaks
        dist = dist // 2
    return _find_candidates_at(heatmap, max_n, MIN_DIST_FLOOR)


def find_peaks(heatmap, n, min_dist, prev_peaks=None):
    candidates = find_candidates(heatmap, n * 2, min_dist)
    if len(candidates) <= n or prev_peaks is None:
        return candidates[:n]
    chosen = []
    remaining = list(candidates)
    for px, py, _ in prev_peaks:
        if not remaining:
            break
        best_i = min(range(len(remaining)),
                     key=lambda i: (remaining[i][0] - px) ** 2 + (remaining[i][1] - py) ** 2)
        chosen.append(remaining.pop(best_i))
    return chosen


# -- main --

def main():

    
    experiment_dir = Path(EXPERIMENT_DIR)

    # Load crop log
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop log not found: {crop_path}\nRun 01_crop.py first.")
    with open(crop_path) as f:
        crop_log = json.load(f)

    # Find model
    if MODEL_PATH:
        model_file = Path(MODEL_PATH)
    else:
        models_root = experiment_dir / "models"
        runs = sorted([d for d in models_root.iterdir() if d.is_dir()])
        model_file = runs[-1] / "best.keras"
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    print(f"Experiment : {EXPERIMENT_ID}")
    print(f"Model      : {model_file}")

    model = tf.keras.models.load_model(str(model_file), compile=False)
    shp = model.input_shape
    img_size = shp[1] if shp[1] is not None else IMG_SIZE

    raw_dir = Path(crop_log["raw_dir"])
    frames = crop_log["frames"]
    plants = crop_log["plants"]
    print(f"Frames     : {len(frames)}")
    print(f"Plants     : {len(plants)}")

    video_dir = experiment_dir / "output" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for pi, plant in enumerate(plants):
        genotype = plant["genotype"]
        replicate = plant["replicate"]
        bbox = plant["bbox"]
        x0, y0, x1, y1 = bbox
        label = f"g{genotype}_{replicate}"
        crop_h, crop_w = y1 - y0, x1 - x0

        print(f"\n[{pi + 1}/{len(plants)}] {label}")

        # Video 1: raw crop journey
        crop_out = video_dir / f"{label}_crop.mp4"
        crop_writer = cv2.VideoWriter(str(crop_out), fourcc, VIDEO_FPS,
                                      (crop_w, crop_h))

        # Video 2: overlay diagnostic (side-by-side, scaled up)
        canvas_w = crop_w * 2
        canvas_h = crop_h
        disp_w = canvas_w * DISPLAY_SCALE
        disp_h = canvas_h * DISPLAY_SCALE
        overlay_out = video_dir / f"{label}_overlay.mp4"
        overlay_writer = cv2.VideoWriter(str(overlay_out), fourcc, VIDEO_FPS,
                                         (disp_w, disp_h))

        if not crop_writer.isOpened() or not overlay_writer.isOpened():
            print(f"  WARNING: could not open video writers, skipping {label}")
            crop_writer.release()
            overlay_writer.release()
            continue

        prev_peaks = None

        for fi, frame_name in enumerate(frames):
            frame_path = raw_dir / frame_name
            full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if full_img is None:
                print(f"  Warning: could not read {frame_name}, skipping")
                continue

            crop = full_img[y0:y1, x0:x1]

            # === Video 1: write raw crop ===
            crop_writer.write(crop)

            # === Video 2: model inference + overlay ===
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (img_size, img_size),
                                      interpolation=cv2.INTER_AREA)
            crop_norm = (crop_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
            pred = model.predict(crop_norm, verbose=0)
            heatmap = pred[0, :, :, 0].astype(np.float32)

            peaks = find_peaks(heatmap, NUM_TIPS, MIN_DIST, prev_peaks)
            prev_peaks = peaks

            # Left panel: crop with peak markers + line
            overlay_crop = crop.copy()
            h0, w0 = overlay_crop.shape[:2]
            sx = (w0 - 1) / (img_size - 1) if img_size > 1 else 1.0
            sy = (h0 - 1) / (img_size - 1) if img_size > 1 else 1.0
            pts = []
            for i, (px, py, val) in enumerate(peaks):
                ox = int(round(px * sx))
                oy = int(round(py * sy))
                ox = max(0, min(w0 - 1, ox))
                oy = max(0, min(h0 - 1, oy))
                pts.append((ox, oy))
                cv2.circle(overlay_crop, (ox, oy), 3, (0, 255, 0), 1)
                cv2.putText(overlay_crop, str(i + 1), (ox + 5, oy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            dist_text = ""
            if len(pts) >= 2:
                cv2.line(overlay_crop, pts[0], pts[1], (255, 255, 255), 1)
                dx = pts[0][0] - pts[1][0]
                dy = pts[0][1] - pts[1][1]
                dist = float(np.hypot(dx, dy))
                dist_text = f"  dist={dist:.1f}px"

            # Right panel: crop with low-opacity heatmap overlay
            hmap_big = cv2.resize(heatmap, (w0, h0),
                                  interpolation=cv2.INTER_LINEAR)
            hmap_color = cv2.applyColorMap(
                (hmap_big * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hmap_color = cv2.addWeighted(crop, 0.3, hmap_color, 0.3, 0)

            # Stitch side by side + scale up
            canvas = np.hstack([overlay_crop, hmap_color])
            disp = cv2.resize(canvas,
                              (canvas.shape[1] * DISPLAY_SCALE,
                               canvas.shape[0] * DISPLAY_SCALE),
                              interpolation=cv2.INTER_NEAREST)

            # Title bar
            title = (f"{label}  frame {fi}/{len(frames) - 1}"
                     f"  [{frame_name}]{dist_text}")
            cv2.putText(disp, title, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            overlay_writer.write(disp)

            if (fi + 1) % 50 == 0 or fi == len(frames) - 1:
                print(f"  {fi + 1}/{len(frames)} frames")

        crop_writer.release()
        overlay_writer.release()
        print(f"  -> {crop_out.name}")
        print(f"  -> {overlay_out.name}")

    print(f"\n{'=' * 50}")
    print(f"Videos -> {video_dir}")
    print(f"Plants: {len(plants)}, Frames per plant: {len(frames)}")
    print(f"FPS: {VIDEO_FPS}")


if __name__ == "__main__":
    main()

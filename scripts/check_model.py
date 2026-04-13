# ============================================================
# Visual sanity check -- shows crop + heatmap overlay + peaks
# Navigate: arrow keys (left/right = frame, up/down = plant)
#           q = quit
# Uses same peak detection as 05_measure.py
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
DISPLAY_SCALE  = 3

PEAK_THRESH = 0.05
MIN_DIST_FLOOR = 5


# -- reuse peak detection from 05_measure --

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
    with open(experiment_dir / "logs" / "01_crop.json") as f:
        crop_log = json.load(f)

    # Find model
    if MODEL_PATH:
        model_file = Path(MODEL_PATH)
    else:
        models_root = experiment_dir / "models"
        runs = sorted([d for d in models_root.iterdir() if d.is_dir()])
        model_file = runs[-1] / "best.keras"
    print(f"Model: {model_file}")

    model = tf.keras.models.load_model(str(model_file), compile=False)
    shp = model.input_shape
    img_size = shp[1] if shp[1] is not None else IMG_SIZE

    raw_dir = Path(crop_log["raw_dir"])
    frames = crop_log["frames"]
    all_plants = crop_log["plants"]

    # Build label lookup
    def label_for(p):
        return f"g{p['genotype']}_{p['replicate']}"

    # Let user pick which plants to review
    print(f"\nPlants ({len(all_plants)}):")
    genotypes = sorted(set(p["genotype"] for p in all_plants))
    for p in all_plants:
        print(f"  {label_for(p)}")
    print(f"\nGenotypes: {', '.join(f'g{g}' for g in genotypes)}")
    print(f"\nType a label (g1_r01), genotype (g1), comma-separated (g1_r01,g2_r03), or Enter for all:")
    choice = input("> ").strip().lower()

    if choice == "":
        plants = all_plants
    else:
        parts = [c.strip() for c in choice.split(",")]
        matched = []
        for part in parts:
            # Try exact label match first (e.g. g1_r01)
            exact = [p for p in all_plants if label_for(p).lower() == part]
            if exact:
                matched.extend(exact)
            else:
                # Try genotype match (e.g. g1)
                try:
                    geno = int(part.lstrip("g"))
                    matched.extend(p for p in all_plants if p["genotype"] == geno)
                except ValueError:
                    print(f"  Skipping unknown '{part}'")
        # Deduplicate preserving order
        seen = set()
        plants = []
        for p in matched:
            pid = p.get("crop_uid", p["id"])
            if pid not in seen:
                seen.add(pid)
                plants.append(p)
        if not plants:
            print("No matches, using all.")
            plants = all_plants

    print(f"Reviewing {len(plants)} plant(s): {', '.join(label_for(p) for p in plants)}")

    # State
    plant_idx = 0
    frame_idx = 0
    # Cache tracked peaks per plant: plant_idx -> {frame_idx -> peaks}
    peak_cache: dict[int, dict[int, list]] = {}

    playing = False
    window = "check_model | n/b=frame | w/s=plant | space=play/pause | q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def get_peaks_for(pi, fi):
        """Get peaks for plant pi at frame fi, running tracking from frame 0."""
        if pi not in peak_cache:
            peak_cache[pi] = {}
        if fi in peak_cache[pi]:
            return peak_cache[pi][fi]
        # Find the latest cached frame before fi
        cached = sorted(peak_cache[pi].keys())
        start = 0
        prev = None
        for c in cached:
            if c < fi:
                start = c + 1
                prev = peak_cache[pi][c]
        # Run forward from start to fi
        plant = plants[pi]
        bbox = plant["bbox"]
        x0, y0, x1, y1 = bbox
        for f in range(start, fi + 1):
            if f in peak_cache[pi]:
                prev = peak_cache[pi][f]
                continue
            frame_path = raw_dir / frames[f]
            full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if full_img is None:
                peak_cache[pi][f] = prev if prev else []
                continue
            crop = full_img[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            crop_norm = (crop_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
            pred = model.predict(crop_norm, verbose=0)
            heatmap = pred[0, :, :, 0].astype(np.float32)
            peaks = find_peaks(heatmap, NUM_TIPS, MIN_DIST, prev)
            peak_cache[pi][f] = peaks
            prev = peaks
        return peak_cache[pi].get(fi, [])

    def show():
        plant = plants[plant_idx]
        bbox = plant["bbox"]
        x0, y0, x1, y1 = bbox
        label = f"g{plant['genotype']}_{plant['replicate']}"

        frame_path = raw_dir / frames[frame_idx]
        full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if full_img is None:
            print(f"  Could not read {frame_path}")
            return
        crop = full_img[y0:y1, x0:x1]
        h0, w0 = crop.shape[:2]

        # Get model heatmap
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        crop_norm = (crop_resized.astype(np.float32) / 255.0)[np.newaxis, ...]
        pred = model.predict(crop_norm, verbose=0)
        heatmap = pred[0, :, :, 0].astype(np.float32)

        # Get tracked peaks
        peaks = get_peaks_for(plant_idx, frame_idx)

        # Left side: crop with peaks + line overlay
        # Right side: raw heatmap
        hmap_big = cv2.resize(heatmap, (w0, h0), interpolation=cv2.INTER_LINEAR)
        hmap_color = cv2.applyColorMap((hmap_big * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Scale peaks to crop coords and draw on crop
        sx = (w0 - 1) / (img_size - 1) if img_size > 1 else 1.0
        sy = (h0 - 1) / (img_size - 1) if img_size > 1 else 1.0
        pts = []
        for i, (px, py, val) in enumerate(peaks):
            ox = int(round(px * sx))
            oy = int(round(py * sy))
            ox = max(0, min(w0 - 1, ox))
            oy = max(0, min(h0 - 1, oy))
            pts.append((ox, oy))
            cv2.circle(crop, (ox, oy), 3, (0, 255, 0), 1)
            cv2.putText(crop, str(i+1), (ox+5, oy-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Draw line + distance between first two peaks
        dist_text = ""
        if len(pts) >= 2:
            cv2.line(crop, pts[0], pts[1], (255, 255, 255), 1)
            dx = pts[0][0] - pts[1][0]
            dy = pts[0][1] - pts[1][1]
            dist = float(np.hypot(dx, dy))
            dist_text = f"  dist={dist:.1f}px"

        # Stitch side by side: crop overlay | heatmap
        canvas = np.hstack([crop, hmap_color])

        # Scale up for display
        dh, dw = canvas.shape[:2]
        disp = cv2.resize(canvas, (dw * DISPLAY_SCALE, dh * DISPLAY_SCALE),
                          interpolation=cv2.INTER_NEAREST)

        # Title bar
        title = f"{label}  frame {frame_idx}/{len(frames)-1}  [{frames[frame_idx]}]{dist_text}"
        cv2.putText(disp, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window, disp)

    show()
    while True:
        wait = 16 if playing else 0  # ~60fps when playing
        key = cv2.waitKey(wait) & 0xFF
        if playing and key == 255:  # no key pressed, advance
            if frame_idx < len(frames) - 1:
                frame_idx += 1
                show()
            else:
                playing = False
            continue
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("n"):
            playing = False
            frame_idx = min(frame_idx + 1, len(frames) - 1)
            show()
        elif key == ord("b"):
            playing = False
            frame_idx = max(frame_idx - 1, 0)
            show()
        elif key == ord("w"):
            playing = False
            plant_idx = (plant_idx - 1) % len(plants)
            frame_idx = 0
            show()
        elif key == ord("s"):
            playing = False
            plant_idx = (plant_idx + 1) % len(plants)
            frame_idx = 0
            show()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

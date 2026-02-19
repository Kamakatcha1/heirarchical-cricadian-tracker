# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
# ============================================================
_OVERRIDE = {}
# ============================================================


# TODO Crop.py should remember the crops you made and allow you to alter them after the fact if referencing the same experiments through a deletion and redraw tool or add more drawings even if you like

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)
if "EXPERIMENT_ID" in _OVERRIDE or "BASE_DIR" in _OVERRIDE:
    _config.EXPERIMENTS_DIR = rf"{_config.BASE_DIR}\data\experiments"
    _config.EXPERIMENT_DIR  = rf"{_config.EXPERIMENTS_DIR}\{_config.EXPERIMENT_ID}"
if "DATASET_ID" in _OVERRIDE or "BASE_DIR" in _OVERRIDE:
    _config.RAW_DIR = rf"{_config.BASE_DIR}\data\raw\{_config.DATASET_ID}"

EXPERIMENT_ID   = _config.EXPERIMENT_ID
EXPERIMENT_DIR  = _config.EXPERIMENT_DIR
RAW_DIR         = _config.RAW_DIR
DATASET_ID      = _config.DATASET_ID
BLEND_METHOD    = _config.BLEND_METHOD
EXPECTED_PLANTS = _config.EXPECTED_PLANTS
MAX_DISPLAY     = _config.MAX_DISPLAY


def collect_raw_frames(raw_dir: Path) -> list[Path]:
    """Gather and sort all image files in the raw dataset folder."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No images found in raw dataset: {raw_dir}")
    return paths


def blend_composite(image_paths: list[Path], method: str) -> np.ndarray:
    """Blend raw frames into a single composite in memory.

    Exact logic from original grid.py blend_images():
    - 'max': per-pixel maximum across all frames
    - 'mean': per-pixel average across all frames
    Frames that differ in size from the first are resized to match.
    """
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image: {image_paths[0]}")
    h, w = first.shape[:2]

    if method == "max":
        blended = first.copy()
        for path in image_paths[1:]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            blended = np.maximum(blended, img)
        return blended

    if method == "mean":
        acc = first.astype(np.float32)
        count = 1
        for path in image_paths[1:]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            acc += img.astype(np.float32)
            count += 1
        return np.clip(acc / max(count, 1), 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown blend method: {method}")


def select_rectangles(
    img: np.ndarray,
    max_display: int,
    expected: int,
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Interactive rectangle selection on a downscaled preview.

    Exact logic from original grid.py select_rectangles():
    - Draw rectangles by dragging. Coordinates map back to full-res.
    - Press 1/2/3/4 to assign genotype to the last drawn rectangle.
    - u = undo last, c = clear all, s/Enter = save, q/Esc = cancel.
    - Won't save until all rectangles have a genotype assigned.
    """
    h, w = img.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > max_display:
        scale = max_display / float(max_side)
    preview = img if scale == 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)))

    window = "Draw rectangles (drag). u=undo, c=clear, 1-4=genotype, s=save, q=quit"
    rects: List[Tuple[Tuple[int, int, int, int], int]] = []
    drawing = False
    start = (0, 0)
    current = (0, 0)

    def redraw():
        canvas = preview.copy()
        for (x0, y0, x1, y1), geno in rects:
            cv2.rectangle(
                canvas,
                (int(x0 * scale), int(y0 * scale)),
                (int(x1 * scale), int(y1 * scale)),
                (0, 255, 0),
                2,
            )
            label = f"G{geno}"
            cv2.putText(
                canvas,
                label,
                (int(x0 * scale) + 4, int(y0 * scale) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if drawing:
            cv2.rectangle(canvas, start, current, (0, 0, 255), 2)
        cv2.imshow(window, canvas)

    def _cb(event, x, y, flags, _param):
        nonlocal drawing, start, current
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = start
            x1, y1 = current
            if x0 == x1 or y0 == y1:
                return
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            ox0 = int(round(x0 / scale))
            oy0 = int(round(y0 / scale))
            ox1 = int(round(x1 / scale))
            oy1 = int(round(y1 / scale))
            rects.append(((ox0, oy0, ox1, oy1), 0))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, preview)
    cv2.setMouseCallback(window, _cb)

    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            rects = []
            break
        if key in (ord("u"), ord("U")) and rects:
            rects.pop()
        if key in (ord("c"), ord("C")):
            rects.clear()
        if key in (ord("1"), ord("2"), ord("3"), ord("4")) and rects:
            geno = int(chr(key))
            rect, _ = rects[-1]
            rects[-1] = (rect, geno)
        if key in (ord("s"), ord("S"), 13):
            if expected and len(rects) != expected:
                print(f"Expected {expected} rectangles, got {len(rects)}. Keep drawing.")
                continue
            missing = [i for i, (_, g) in enumerate(rects, 1) if g == 0]
            if missing:
                print(f"Rectangles missing genotype (press 1/2/3/4): {missing}")
                continue
            break

    cv2.destroyAllWindows()
    if not rects:
        raise RuntimeError("Rectangle selection cancelled or empty")
    return rects


def assign_replicate_numbers(
    rects: List[Tuple[Tuple[int, int, int, int], int]],
) -> list[dict]:
    """Assign replicate numbers per genotype in draw order.

    Returns list of plant dicts ready for JSON serialization.
    """
    genotype_counter: Counter = Counter()
    plants = []
    for (x0, y0, x1, y1), geno in rects:
        genotype_counter[geno] += 1
        rep = genotype_counter[geno]
        plants.append({
            "id": f"g{geno}_r{rep:02d}",
            "genotype": geno,
            "replicate": rep,
            "bbox": [x0, y0, x1, y1],
        })
    return plants


def write_log(log_path: Path, log_data: dict) -> None:
    """Write a JSON log for this step."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def sanity_check(plants: list[dict], frame_count: int) -> dict:
    """Verify the crop definitions are valid."""
    checks = {}

    checks["plant_count"] = len(plants)
    checks["has_plants"] = len(plants) > 0

    # All genotypes valid (1-4)
    genotypes = [p["genotype"] for p in plants]
    checks["all_genotypes_valid"] = all(1 <= g <= 4 for g in genotypes)

    # No zero-area rectangles
    areas = [(p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) for p in plants]
    checks["min_crop_area_px"] = min(areas) if areas else 0
    checks["no_zero_area"] = all(a > 0 for a in areas)

    # No duplicate plant IDs
    ids = [p["id"] for p in plants]
    checks["no_duplicate_ids"] = len(ids) == len(set(ids))

    # Frame count is reasonable
    checks["frame_count"] = frame_count
    checks["has_frames"] = frame_count >= 1

    checks["all_passed"] = all([
        checks["has_plants"],
        checks["all_genotypes_valid"],
        checks["no_zero_area"],
        checks["no_duplicate_ids"],
        checks["has_frames"],
    ])
    return checks


def main() -> None:
    raw_dir = Path(RAW_DIR)
    experiment_dir = Path(EXPERIMENT_DIR)
    log_path = experiment_dir / "logs" / "01_crop.json"

    # Collect raw frames
    raw_frames = collect_raw_frames(raw_dir)
    print(f"Dataset: {DATASET_ID}")
    print(f"Found {len(raw_frames)} raw frames in {raw_dir}")

    # Blend composite in memory
    print(f"Blending composite in memory (method: {BLEND_METHOD})...")
    composite = blend_composite(raw_frames, BLEND_METHOD)
    print(f"Composite: {composite.shape[1]}x{composite.shape[0]} (WxH)")

    # Interactive rectangle selection
    print("\nDraw bounding rectangles on the composite.")
    print("  Drag to draw, 1-4 to assign genotype, u=undo, c=clear")
    print("  s or Enter to save, q or Esc to cancel")
    rects = select_rectangles(composite, MAX_DISPLAY, EXPECTED_PLANTS)

    # Assign replicate numbers per genotype
    plants = assign_replicate_numbers(rects)

    print(f"\n{len(plants)} plants defined:")
    for p in plants:
        x0, y0, x1, y1 = p["bbox"]
        print(f"  {p['id']}: ({x0},{y0})-({x1},{y1})  {x1-x0}x{y1-y0}px")

    # Build frame index (filename -> index mapping)
    frames = [
        {"index": i, "filename": p.name}
        for i, p in enumerate(raw_frames)
    ]

    # Sanity check
    checks = sanity_check(plants, len(raw_frames))
    if not checks["all_passed"]:
        print("\n*** SANITY CHECK FAILED ***")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        print("Aborting.")
        sys.exit(1)
    print("Sanity checks passed.")

    # Write crop definitions JSON
    log_data = {
        "step": "01_crop",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "timestamp": datetime.now().isoformat(),
        "raw_dir": str(raw_dir),
        "blend_method": BLEND_METHOD,
        "frame_count": len(raw_frames),
        "frames": frames,
        "plants": plants,
        "sanity": checks,
    }
    write_log(log_path, log_data)
    print(f"\nCrop definitions saved to: {log_path}")
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

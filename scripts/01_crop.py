# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
# ============================================================
_OVERRIDE = {}
# ============================================================


import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

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
MAX_FRAMES      = _config.MAX_FRAMES


def collect_raw_frames(raw_dir: Path, max_frames: int = 0) -> list[Path]:
    """Gather and sort all image files in the raw dataset folder.

    If max_frames > 0, keep only the first max_frames frames.
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No images found in raw dataset: {raw_dir}")

    if max_frames > 0 and len(paths) > max_frames:
        paths = paths[:max_frames]

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


def load_existing_crops(experiment_dir: Path) -> list[dict] | None:
    """Load previously saved crop definitions if they exist.

    Returns the plants list from the existing 01_crop.json, or None
    if no previous crop log exists.
    """
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        return None
    with open(crop_path) as f:
        data = json.load(f)
    return data.get("plants", None)


def plants_to_rects(
    plants: list[dict],
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Convert plant dicts back to the (bbox, genotype) tuple format used by
    the rectangle selection GUI."""
    rects = []
    for p in plants:
        x0, y0, x1, y1 = p["bbox"]
        rects.append(((x0, y0, x1, y1), p["genotype"]))
    return rects


def select_rectangles(
    img: np.ndarray,
    max_display: int,
    expected: int,
    existing_rects: List[Tuple[Tuple[int, int, int, int], int]] | None = None,
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Interactive rectangle selection on a downscaled preview.

    Controls:
    - Drag to draw a new rectangle.  Coordinates map back to full-res.
    - Press 1/2/3/4 to assign genotype to the last drawn rectangle.
    - Right-click on a rectangle to select it (highlighted in yellow).
    - d = delete the selected (yellow) rectangle.
    - u = undo last drawn rectangle.
    - c = clear ALL rectangles.
    - w = wipe ALL crops (same as clear, explicit reset hotkey).
    - s / Enter = save (won't save until all rects have a genotype).
    - q / Esc = cancel (discards everything).

    If existing_rects is provided, they are pre-loaded onto the canvas so you
    can add to, modify, or delete previous crops.
    """
    h, w = img.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > max_display:
        scale = max_display / float(max_side)
    preview = img if scale == 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)))

    window = "Draw rects | 1-4=genotype | d=delete | u=undo | c=clear | w=wipe all | s=save | q=quit"
    rects: List[Tuple[Tuple[int, int, int, int], int]] = []
    if existing_rects:
        rects.extend(existing_rects)

    drawing = False
    start = (0, 0)
    current = (0, 0)
    selected_idx = -1  # index of currently selected (yellow) rectangle

    def _point_in_rect(px: int, py: int, rx0: int, ry0: int, rx1: int, ry1: int) -> bool:
        """Check if a display-space point is inside a rectangle (display-space)."""
        return rx0 <= px <= rx1 and ry0 <= py <= ry1

    def redraw():
        canvas = preview.copy()
        for i, ((x0, y0, x1, y1), geno) in enumerate(rects):
            color = (0, 255, 255) if i == selected_idx else (0, 255, 0)
            thickness = 3 if i == selected_idx else 2
            cv2.rectangle(
                canvas,
                (int(x0 * scale), int(y0 * scale)),
                (int(x1 * scale), int(y1 * scale)),
                color,
                thickness,
            )
            label = f"G{geno}" if geno > 0 else "G?"
            cv2.putText(
                canvas,
                label,
                (int(x0 * scale) + 4, int(y0 * scale) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        if drawing:
            cv2.rectangle(canvas, start, current, (0, 0, 255), 2)
        cv2.imshow(window, canvas)

    def _cb(event, x, y, flags, _param):
        nonlocal drawing, start, current, selected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            current = (x, y)
            selected_idx = -1  # deselect on new draw
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = start
            x1, y1 = current
            if abs(x0 - x1) < 3 and abs(y0 - y1) < 3:
                # Tiny drag = treat as a click to select existing rect
                for i, ((rx0, ry0, rx1, ry1), _geno) in enumerate(rects):
                    sx0 = int(rx0 * scale)
                    sy0 = int(ry0 * scale)
                    sx1 = int(rx1 * scale)
                    sy1 = int(ry1 * scale)
                    if _point_in_rect(x, y, sx0, sy0, sx1, sy1):
                        selected_idx = i
                        return
                selected_idx = -1
                return
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            ox0 = int(round(x0 / scale))
            oy0 = int(round(y0 / scale))
            ox1 = int(round(x1 / scale))
            oy1 = int(round(y1 / scale))
            rects.append(((ox0, oy0, ox1, oy1), 0))
            selected_idx = len(rects) - 1  # auto-select the new rect
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to select an existing rectangle
            for i, ((rx0, ry0, rx1, ry1), _geno) in enumerate(rects):
                sx0 = int(rx0 * scale)
                sy0 = int(ry0 * scale)
                sx1 = int(rx1 * scale)
                sy1 = int(ry1 * scale)
                if _point_in_rect(x, y, sx0, sy0, sx1, sy1):
                    selected_idx = i
                    return
            selected_idx = -1

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, preview)
    cv2.setMouseCallback(window, _cb)

    cancelled = False
    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cancelled = True
            break
        if key in (ord("d"), ord("D")):
            # Delete the selected rectangle
            if 0 <= selected_idx < len(rects):
                rects.pop(selected_idx)
                selected_idx = -1
        if key in (ord("u"), ord("U")) and rects:
            rects.pop()
            selected_idx = -1
        if key in (ord("c"), ord("C")):
            rects.clear()
            selected_idx = -1
        if key in (ord("w"), ord("W")):
            rects.clear()
            selected_idx = -1
            print("All crops wiped. Draw new rectangles, then save.")
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            geno = int(chr(key))
            # Assign genotype to selected rect, or last rect if none selected
            target = selected_idx if 0 <= selected_idx < len(rects) else (len(rects) - 1 if rects else -1)
            if target >= 0:
                rect, _ = rects[target]
                rects[target] = (rect, geno)
        if key in (ord("s"), ord("S"), 13):
            # Allow an explicit empty save (e.g., after wipe-all), regardless of expected count.
            if expected and rects and len(rects) != expected:
                print(f"Expected {expected} rectangles, got {len(rects)}. Keep drawing.")
                continue
            missing = [i for i, (_, g) in enumerate(rects, 1) if g == 0]
            if missing:
                print(f"Rectangles missing genotype (press 1/2/3/4): {missing}")
                continue
            break

    cv2.destroyAllWindows()
    if cancelled:
        raise RuntimeError("Rectangle selection cancelled")
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


def assign_stable_crop_uids(
    plants: list[dict],
    existing_plants: list[dict] | None,
) -> list[dict]:
    """Attach stable crop_uids, preserving IDs for unchanged existing crops.

    Matching is by exact (genotype, bbox). This keeps identity stable when
    crops are deleted/reordered, and only assigns new IDs to genuinely new crops.
    """
    existing_key_to_uids: dict[tuple[int, tuple[int, int, int, int]], list[str]] = {}
    if existing_plants:
        for p in existing_plants:
            crop_uid = p.get("crop_uid")
            if not crop_uid:
                continue
            key = (int(p["genotype"]), tuple(p["bbox"]))
            existing_key_to_uids.setdefault(key, []).append(crop_uid)

    for p in plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        if key in existing_key_to_uids and existing_key_to_uids[key]:
            p["crop_uid"] = existing_key_to_uids[key].pop(0)
        else:
            p["crop_uid"] = uuid4().hex
    return plants


def write_log(log_path: Path, log_data: dict) -> None:
    """Write a JSON log for this step."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def sync_annotation_log_with_crops(
    experiment_dir: Path,
    plants: list[dict],
) -> dict | None:
    """Prune/relabel existing annotations so they match current crops.

    This keeps 02_annotate.json free of dead annotations immediately after
    crop edits, even before running 02_annotate.py again.
    """
    ann_path = experiment_dir / "logs" / "02_annotate.json"
    if not ann_path.exists():
        return None

    with open(ann_path) as f:
        ann_log = json.load(f)

    annotations = ann_log.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    by_uid = {p.get("crop_uid"): p for p in plants if p.get("crop_uid")}
    key_to_plants: dict[tuple[int, tuple[int, int, int, int]], list[dict]] = {}
    for p in plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        key_to_plants.setdefault(key, []).append(p)

    kept: list[dict] = []
    dropped = 0
    for ann in annotations:
        if not isinstance(ann, dict):
            dropped += 1
            continue

        ann2 = dict(ann)
        matched = None

        uid = ann2.get("crop_uid")
        if uid and uid in by_uid:
            matched = by_uid[uid]
        elif not uid:
            bbox = ann2.get("crop_bbox")
            genotype = ann2.get("genotype")
            if bbox is not None and genotype is not None:
                key = (int(genotype), tuple(bbox))
                candidates = key_to_plants.get(key, [])
                if len(candidates) == 1:
                    matched = candidates[0]

        if matched is None:
            dropped += 1
            continue

        ann2["crop_uid"] = matched.get("crop_uid", matched["id"])
        ann2["plant_id"] = matched["id"]
        ann2["replicate"] = matched["replicate"]
        ann2["genotype"] = matched["genotype"]
        kept.append(ann2)

    dedup: dict[tuple[str, int], dict] = {}
    for ann in kept:
        frame_index = ann.get("frame_index")
        if not isinstance(frame_index, int):
            dropped += 1
            continue
        dedup[(ann["crop_uid"], frame_index)] = ann

    kept = list(dedup.values())

    ann_log["annotations"] = kept
    ann_log["timestamp"] = datetime.now().isoformat()
    ann_log["total_annotations"] = len(kept)
    ann_log["plants_total"] = len(plants)
    ann_log["plants_annotated"] = len({a.get("crop_uid", a.get("plant_id")) for a in kept})

    with open(ann_path, "w") as f:
        json.dump(ann_log, f, indent=2)

    return {
        "path": str(ann_path),
        "kept": len(kept),
        "dropped": dropped,
    }


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

    # Stable crop_uids are required and unique
    crop_uids = [p.get("crop_uid") for p in plants]
    checks["all_have_crop_uid"] = all(isinstance(uid, str) and len(uid) > 0 for uid in crop_uids)
    checks["no_duplicate_crop_uid"] = len(crop_uids) == len(set(crop_uids))

    # Frame count is reasonable
    checks["frame_count"] = frame_count
    checks["has_frames"] = frame_count >= 1

    checks["all_passed"] = all([
        checks["all_genotypes_valid"],
        checks["no_zero_area"],
        checks["no_duplicate_ids"],
        checks["all_have_crop_uid"],
        checks["no_duplicate_crop_uid"],
        checks["has_frames"],
    ])
    return checks


def main() -> None:
    raw_dir = Path(RAW_DIR)
    experiment_dir = Path(EXPERIMENT_DIR)
    log_path = experiment_dir / "logs" / "01_crop.json"

    # Collect raw frames (with optional frame limiting)
    raw_frames = collect_raw_frames(raw_dir, MAX_FRAMES)
    print(f"Dataset: {DATASET_ID}")
    if MAX_FRAMES > 0:
        print(f"Frame limit: first {MAX_FRAMES} frames")
    print(f"Using {len(raw_frames)} raw frames from {raw_dir}")

    # Check for existing crop definitions
    existing_plants = load_existing_crops(experiment_dir)
    existing_rects = None
    if existing_plants:
        print(f"\nFound {len(existing_plants)} existing crop definitions:")
        for p in existing_plants:
            x0, y0, x1, y1 = p["bbox"]
            print(f"  {p['id']}: ({x0},{y0})-({x1},{y1})  {x1-x0}x{y1-y0}px")
        print("These will be pre-loaded. You can add, delete (d), or redraw.")
        existing_rects = plants_to_rects(existing_plants)

    # Blend composite in memory
    print(f"\nBlending composite in memory (method: {BLEND_METHOD})...")
    composite = blend_composite(raw_frames, BLEND_METHOD)
    print(f"Composite: {composite.shape[1]}x{composite.shape[0]} (WxH)")

    # Interactive rectangle selection (with existing rects pre-loaded)
    print("\nDraw bounding rectangles on the composite.")
    print("  Drag to draw, right-click or tiny-drag to select existing")
    print("  1-4 = assign genotype (to selected or last drawn)")
    print("  d = delete selected, u = undo last, c = clear all, w = wipe all crops")
    print("  s or Enter to save, q or Esc to cancel")
    rects = select_rectangles(composite, MAX_DISPLAY, EXPECTED_PLANTS, existing_rects)

    # Assign replicate numbers per genotype
    plants = assign_replicate_numbers(rects)
    plants = assign_stable_crop_uids(plants, existing_plants)

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
        "max_frames": MAX_FRAMES,
        "frame_count": len(raw_frames),
        "frames": frames,
        "plants": plants,
        "sanity": checks,
    }
    write_log(log_path, log_data)
    print(f"\nCrop definitions saved to: {log_path}")
    sync_result = sync_annotation_log_with_crops(experiment_dir, plants)
    if sync_result is not None:
        print(
            f"Synced annotations: kept={sync_result['kept']}, dropped={sync_result['dropped']}"
            f" ({sync_result['path']})"
        )
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

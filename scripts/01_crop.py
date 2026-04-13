# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
#
# For EXISTING experiments, settings are restored from the
# saved 01_crop.json automatically.  Only use _OVERRIDE if
# you want to CHANGE a saved setting (e.g. switch dataset).
# ============================================================
_OVERRIDE = {}
# ============================================================


import json
import sys
from collections import Counter
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
    """Blend raw frames into a single composite in memory."""
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


def load_existing_crop_log(experiment_dir: Path) -> dict | None:
    """Load the full existing crop JSON if it exists."""
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        return None
    with open(crop_path) as f:
        return json.load(f)


def plants_to_rects(
    plants: list[dict],
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """Convert plant dicts back to the (bbox, genotype) tuple format."""
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
    - s / Enter = save (won't save until all rects have a genotype).
    - q / Esc = cancel (discards everything).

    If existing_rects is provided, they are pre-loaded onto the canvas.
    """
    h, w = img.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > max_display:
        scale = max_display / float(max_side)
    preview = img if scale == 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)))

    window = "Draw rects | 1-4=genotype | d=delete | u=undo | c=clear | s=save | q=quit"
    rects: List[Tuple[Tuple[int, int, int, int], int]] = []
    if existing_rects:
        rects.extend(existing_rects)

    drawing = False
    start = (0, 0)
    current = (0, 0)
    selected_idx = -1

    def _point_in_rect(px, py, rx0, ry0, rx1, ry1):
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
                color, thickness,
            )
            if geno > 0:
                rep = sum(1 for _, g in rects[:i+1] if g == geno)
                label = f"g{geno}_r{rep:02d}"
            else:
                label = "G?"
            cv2.putText(
                canvas, label,
                (int(x0 * scale) + 4, int(y0 * scale) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
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
            selected_idx = -1
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = start
            x1, y1 = current
            if abs(x0 - x1) < 3 and abs(y0 - y1) < 3:
                for i, ((rx0, ry0, rx1, ry1), _g) in enumerate(rects):
                    if _point_in_rect(x, y, int(rx0*scale), int(ry0*scale), int(rx1*scale), int(ry1*scale)):
                        selected_idx = i
                        return
                selected_idx = -1
                return
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            rects.append(((int(round(x0/scale)), int(round(y0/scale)),
                           int(round(x1/scale)), int(round(y1/scale))), 0))
            selected_idx = len(rects) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, ((rx0, ry0, rx1, ry1), _g) in enumerate(rects):
                if _point_in_rect(x, y, int(rx0*scale), int(ry0*scale), int(rx1*scale), int(ry1*scale)):
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
        if key in (ord("d"), ord("D")) and 0 <= selected_idx < len(rects):
            rects.pop(selected_idx)
            selected_idx = -1
        if key in (ord("u"), ord("U")) and rects:
            rects.pop()
            selected_idx = -1
        if key in (ord("c"), ord("C")):
            rects.clear()
            selected_idx = -1
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            geno = int(chr(key))
            target = selected_idx if 0 <= selected_idx < len(rects) else (len(rects) - 1 if rects else -1)
            if target >= 0:
                rect, _ = rects[target]
                rects[target] = (rect, geno)
        if key in (ord("s"), ord("S"), 13):
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
    """Assign replicate numbers per genotype in draw order."""
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
    """Attach stable crop_uids, preserving IDs for unchanged existing crops."""
    existing_key_to_uids: dict[tuple, list[str]] = {}
    if existing_plants:
        for p in existing_plants:
            uid = p.get("crop_uid")
            if not uid:
                continue
            key = (int(p["genotype"]), tuple(p["bbox"]))
            existing_key_to_uids.setdefault(key, []).append(uid)

    for p in plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        if key in existing_key_to_uids and existing_key_to_uids[key]:
            p["crop_uid"] = existing_key_to_uids[key].pop(0)
        else:
            p["crop_uid"] = uuid4().hex
    return plants


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def sync_annotation_log(experiment_id: str, experiment_dir: Path, plants: list[dict]) -> dict | None:
    """Prune/relabel existing annotations so they match current crops."""
    ann_path = experiment_dir / "logs" / "02_annotate.json"
    if not ann_path.exists():
        return None

    with open(ann_path) as f:
        ann_log = json.load(f)

    annotations = ann_log.get("annotations", [])
    by_uid = {p.get("crop_uid"): p for p in plants if p.get("crop_uid")}
    key_to_plants: dict[tuple, list[dict]] = {}
    for p in plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        key_to_plants.setdefault(key, []).append(p)

    kept, dropped = [], 0
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
            geno = ann2.get("genotype")
            if bbox is not None and geno is not None:
                candidates = key_to_plants.get((int(geno), tuple(bbox)), [])
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

    # Deduplicate same crop/frame pairs
    dedup: dict[tuple, dict] = {}
    for ann in kept:
        fi = ann.get("frame_index")
        if isinstance(fi, int):
            dedup[(ann["crop_uid"], fi)] = ann
        else:
            dropped += 1
    kept = list(dedup.values())

    # Preserve any extra keys (images_per_folder, display_scale) from the existing log
    out = {k: v for k, v in ann_log.items() if k != "annotations"}
    out["experiment_id"] = experiment_id
    out["annotations"] = kept
    write_json(ann_path, out)
    return {"kept": len(kept), "dropped": dropped}


def main() -> None:
    global DATASET_ID, RAW_DIR, MAX_FRAMES, BLEND_METHOD

    experiment_dir = Path(EXPERIMENT_DIR)
    log_path = experiment_dir / "logs" / "01_crop.json"

    # --- Restore settings from existing experiment ---
    existing_log = load_existing_crop_log(experiment_dir)
    existing_plants = None

    if existing_log is not None:
        existing_plants = existing_log.get("plants")
        restored = []

        # Restore dataset_id + raw_dir (unless explicitly overridden)
        if "DATASET_ID" not in _OVERRIDE:
            saved_ds = existing_log.get("dataset_id")
            if saved_ds and saved_ds != DATASET_ID:
                DATASET_ID = saved_ds
                RAW_DIR = existing_log.get("raw_dir", RAW_DIR)
                restored.append(f"dataset_id={DATASET_ID}")

        # Restore max_frames
        if "MAX_FRAMES" not in _OVERRIDE:
            saved_mf = existing_log.get("max_frames", 0)
            if saved_mf != MAX_FRAMES:
                MAX_FRAMES = saved_mf
                restored.append(f"max_frames={MAX_FRAMES}")

        # Restore blend_method
        if "BLEND_METHOD" not in _OVERRIDE:
            saved_bm = existing_log.get("blend_method")
            if saved_bm and saved_bm != BLEND_METHOD:
                BLEND_METHOD = saved_bm
                restored.append(f"blend_method={BLEND_METHOD}")

        if restored:
            print(f"Restored from existing experiment: {', '.join(restored)}")
            print("  (use _OVERRIDE to change these)")

    raw_dir = Path(RAW_DIR)

    # Collect raw frames
    raw_frames = collect_raw_frames(raw_dir, MAX_FRAMES)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Dataset: {DATASET_ID}")
    if MAX_FRAMES > 0:
        print(f"Frame limit: first {MAX_FRAMES} frames")
    print(f"Using {len(raw_frames)} raw frames from {raw_dir}")

    # Show existing crops if any
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

    # Interactive rectangle selection
    print("\nDraw bounding rectangles on the composite.")
    print("  Drag to draw, right-click or tiny-drag to select existing")
    print("  1-4 = assign genotype (to selected or last drawn)")
    print("  d = delete selected, u = undo last, c = clear all")
    print("  s or Enter to save, q or Esc to cancel")
    rects = select_rectangles(composite, MAX_DISPLAY, EXPECTED_PLANTS, existing_rects)

    # Assign replicate numbers + stable UIDs
    plants = assign_replicate_numbers(rects)
    plants = assign_stable_crop_uids(plants, existing_plants)

    print(f"\n{len(plants)} plants defined:")
    for p in plants:
        x0, y0, x1, y1 = p["bbox"]
        print(f"  {p['id']}: ({x0},{y0})-({x1},{y1})  {x1-x0}x{y1-y0}px")

    # Validate
    genotypes = [p["genotype"] for p in plants]
    areas = [(p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) for p in plants]
    ids = [p["id"] for p in plants]
    if not all(1 <= g <= 4 for g in genotypes):
        print("ERROR: Invalid genotype found."); sys.exit(1)
    if any(a <= 0 for a in areas):
        print("ERROR: Zero-area rectangle found."); sys.exit(1)
    if len(ids) != len(set(ids)):
        print("ERROR: Duplicate plant IDs."); sys.exit(1)
    print("Validation passed.")

    # Write JSON: data + the settings that produced it
    log_data = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "raw_dir": str(raw_dir),
        "max_frames": MAX_FRAMES,
        "blend_method": BLEND_METHOD,
        "frames": [p.name for p in raw_frames],
        "plants": plants,
    }
    write_json(log_path, log_data)
    print(f"\nCrop definitions saved to: {log_path}")

    # Sync annotation log if it exists
    sync_result = sync_annotation_log(EXPERIMENT_ID, experiment_dir, plants)
    if sync_result is not None:
        print(f"Synced annotations: kept={sync_result['kept']}, dropped={sync_result['dropped']}")

    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

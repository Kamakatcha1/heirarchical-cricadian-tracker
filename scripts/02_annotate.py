# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
# ============================================================
_OVERRIDE = {}
# ============================================================

# TODO Annotate.py should tell you if when changes are made as above ^^ you're missing annotations for any new of these boxes. If you are, it should allow you to annotate just those, as well as override old annotations if you need it.


import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)
if "EXPERIMENT_ID" in _OVERRIDE or "BASE_DIR" in _OVERRIDE:
    _config.EXPERIMENTS_DIR = rf"{_config.BASE_DIR}\data\experiments"
    _config.EXPERIMENT_DIR  = rf"{_config.EXPERIMENTS_DIR}\{_config.EXPERIMENT_ID}"
if "DATASET_ID" in _OVERRIDE or "BASE_DIR" in _OVERRIDE:
    _config.RAW_DIR = rf"{_config.BASE_DIR}\data\raw\{_config.DATASET_ID}"

EXPERIMENT_ID     = _config.EXPERIMENT_ID
EXPERIMENT_DIR    = _config.EXPERIMENT_DIR
RAW_DIR           = _config.RAW_DIR
IMAGES_PER_FOLDER = _config.IMAGES_PER_FOLDER
DISPLAY_SCALE     = _config.DISPLAY_SCALE


# --- Globals used by the mouse callback (same pattern as original) ---
points = []
display_img = None
orig_img = None


def draw_point(img, point_xy, idx):
    """Draw a circle on the display image.

    Exact color convention from original annotate.py:
    - idx 1 (center_stem) = blue (255, 0, 0) in BGR
    - idx 0, 2 (tips) = red (0, 0, 255) in BGR
    """
    color = (255, 0, 0) if idx == 1 else (0, 0, 255)
    cv2.circle(img, tuple(point_xy), 6, color, -1)


def mouse_callback(event, x, y, flags, param):
    """Record clicks in original coords, draw in display coords.

    Exact logic from original annotate.py mouse_callback().
    """
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
        x0 = int(round(x / DISPLAY_SCALE))
        y0 = int(round(y / DISPLAY_SCALE))

        h0, w0 = orig_img.shape[:2]
        x0 = max(0, min(w0 - 1, x0))
        y0 = max(0, min(h0 - 1, y0))

        points.append([x0, y0])

        draw_point(display_img, [x, y], len(points) - 1)
        cv2.imshow("annotator", display_img)


def load_crop_log(experiment_dir: Path) -> dict:
    """Load the crop definitions from 01_crop."""
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        raise FileNotFoundError(
            f"Crop log not found at {crop_path}. Run 01_crop.py first."
        )
    with open(crop_path) as f:
        return json.load(f)


def load_annotation_log(experiment_dir: Path) -> dict:
    """Load existing annotation log if it exists, otherwise return empty structure."""
    ann_path = experiment_dir / "logs" / "02_annotate.json"
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)
    return {"annotations": []}


def virtual_crop(raw_frame_path: Path, bbox: list[int]) -> "np.ndarray | None":
    """Read a raw frame and crop it using bbox coordinates. No image saved."""
    img = cv2.imread(str(raw_frame_path))
    if img is None:
        return None
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


def select_frames(
    frame_filenames: list[str],
    target_count: int,
    already_annotated: set[int],
) -> tuple[list[int], list[int]]:
    """Select frame indices with even temporal spacing.

    Exact selection logic from original annotate.py:
    Divides unannotated frames into target_count segments,
    picks one random frame from each.
    Returns (selected_indices, remaining_indices).
    """
    available = [i for i in range(len(frame_filenames)) if i not in already_annotated]
    if not available:
        return [], []

    target_count = min(target_count, len(available))
    if target_count == 0:
        return [], available

    n = len(available)
    selected = []
    for i in range(target_count):
        start = int(i * n / target_count)
        end = int((i + 1) * n / target_count)
        segment = available[start:end]
        if segment:
            selected.append(random.choice(segment))

    if len(selected) < target_count:
        remaining_pool = [i for i in available if i not in selected]
        needed = target_count - len(selected)
        selected.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))

    remaining = [i for i in available if i not in selected]
    return selected, remaining


def annotate_plant(
    plant: dict,
    raw_dir: Path,
    frame_filenames: list[str],
    existing_annotations: list[dict],
    images_per_folder: int,
    display_scale: int,
) -> list[dict]:
    """Run the annotation loop for one plant.

    Returns list of new annotation dicts created this session.
    """
    global points, display_img, orig_img

    plant_id = plant["id"]
    bbox = plant["bbox"]

    # Find which frames are already annotated for this plant
    already_annotated = set()
    for ann in existing_annotations:
        if ann["plant_id"] == plant_id:
            already_annotated.add(ann["frame_index"])

    available_count = len(frame_filenames) - len(already_annotated)
    if available_count == 0:
        print(f"  {plant_id}: all frames annotated, skipping.")
        return []

    target = max(0, images_per_folder - len(already_annotated))
    if target == 0:
        print(f"  {plant_id}: target met ({len(already_annotated)} >= {images_per_folder}), skipping.")
        return []

    if already_annotated:
        print(f"  {plant_id}: {len(already_annotated)} existing, {available_count} available, need {target} more.")

    selected, remaining = select_frames(frame_filenames, target, already_annotated)
    new_annotations = []
    idx = 0

    while idx < len(selected):
        frame_index = selected[idx]
        idx += 1
        points = []

        frame_path = raw_dir / frame_filenames[frame_index]
        orig_img = virtual_crop(frame_path, bbox)
        if orig_img is None:
            print(f"  Could not read/crop {frame_path}", file=sys.stderr)
            continue

        h0, w0 = orig_img.shape[:2]
        disp_w, disp_h = int(w0 * display_scale), int(h0 * display_scale)
        display_img = cv2.resize(orig_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("annotator", display_img)
        cv2.resizeWindow("annotator", disp_w, disp_h)

        print(f"\n  Annotating {plant_id} / frame {frame_index:03d} ({frame_filenames[frame_index]})")
        print("  Click order: leaf_tip_1, center_stem, leaf_tip_2")
        print("  s=save  r=reset  n=skip+replace  x=skip  q=quit")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                points = []
                display_img = cv2.resize(orig_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("annotator", display_img)

            if key == ord("n"):
                if remaining:
                    replacement = random.choice(remaining)
                    remaining.remove(replacement)
                    selected.append(replacement)
                break

            if key == ord("x"):
                break

            if key == ord("s") and len(points) == 3:
                ann = {
                    "plant_id": plant_id,
                    "genotype": plant["genotype"],
                    "replicate": plant["replicate"],
                    "frame_index": frame_index,
                    "frame_filename": frame_filenames[frame_index],
                    "crop_bbox": bbox,
                    "crop_size": [w0, h0],
                    "points": {
                        "leaf_tip_1": points[0],
                        "center_stem": points[1],
                        "leaf_tip_2": points[2],
                    },
                }
                new_annotations.append(ann)
                break

            if key == ord("q"):
                cv2.destroyAllWindows()
                raise SystemExit

    return new_annotations


def write_annotation_log(log_path: Path, log_data: dict) -> None:
    """Write the annotation log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def sanity_check(all_annotations: list[dict]) -> dict:
    """Verify annotations are valid."""
    checks = {}
    checks["total_annotations"] = len(all_annotations)

    malformed = []
    for i, ann in enumerate(all_annotations):
        pts = ann.get("points", {})
        required = {"leaf_tip_1", "center_stem", "leaf_tip_2"}
        if not required.issubset(pts.keys()):
            malformed.append(i)
            continue
        for key in required:
            coord = pts[key]
            if not (isinstance(coord, list) and len(coord) == 2):
                malformed.append(i)
                break

    checks["malformed_indices"] = malformed
    checks["all_valid"] = len(malformed) == 0

    # Coverage
    plant_ids = set(ann["plant_id"] for ann in all_annotations)
    checks["plants_annotated"] = len(plant_ids)

    checks["all_passed"] = checks["all_valid"]
    return checks


def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)
    raw_dir = Path(RAW_DIR)
    log_path = experiment_dir / "logs" / "02_annotate.json"

    # Load crop definitions
    crop_log = load_crop_log(experiment_dir)
    plants = crop_log["plants"]
    frame_filenames = [f["filename"] for f in crop_log["frames"]]
    dataset_id = crop_log["dataset_id"]

    # Resolve raw dir from crop log (in case dataset changed)
    raw_dir = Path(crop_log["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Dataset: {dataset_id} ({len(frame_filenames)} frames)")
    print(f"Plants: {len(plants)}")
    print(f"Target: {IMAGES_PER_FOLDER} annotations per plant")
    print(f"Display scale: {DISPLAY_SCALE}x")

    # Load existing annotations (annotations accumulate, never deleted)
    existing_log = load_annotation_log(experiment_dir)
    existing_annotations = existing_log.get("annotations", [])
    if existing_annotations:
        print(f"Existing annotations: {len(existing_annotations)}")

    # Set up window
    cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotator", mouse_callback)

    all_new = []
    for plant in plants:
        print(f"\n--- {plant['id']} (genotype {plant['genotype']}) ---")
        new = annotate_plant(
            plant, raw_dir, frame_filenames,
            existing_annotations, IMAGES_PER_FOLDER, DISPLAY_SCALE,
        )
        all_new.extend(new)
        # Add to existing so next plant loop sees them
        existing_annotations.extend(new)

    cv2.destroyAllWindows()

    print(f"\nSession complete: {len(all_new)} new annotations")

    # Sanity check on full annotation set
    checks = sanity_check(existing_annotations)
    if not checks["all_passed"]:
        print("\n*** SANITY CHECK FAILED ***")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        print("WARNING: Some annotations are malformed.")
    else:
        print("Sanity checks passed.")

    # Write full annotation log (replaces file -- contains ALL annotations)
    log_data = {
        "step": "02_annotate",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": dataset_id,
        "timestamp": datetime.now().isoformat(),
        "images_per_folder": IMAGES_PER_FOLDER,
        "display_scale": DISPLAY_SCALE,
        "new_this_session": len(all_new),
        "total_annotations": len(existing_annotations),
        "plants_annotated": checks["plants_annotated"],
        "plants_total": len(plants),
        "annotations": existing_annotations,
        "sanity": checks,
    }
    write_annotation_log(log_path, log_data)
    print(f"Annotation log saved to: {log_path}")
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

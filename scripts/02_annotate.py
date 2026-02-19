# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
# ============================================================
_OVERRIDE = {}
# ============================================================


import json
import random
import sys
from datetime import datetime
from pathlib import Path

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


def detect_crop_changes(
    current_plants: list[dict],
    existing_annotations: list[dict],
) -> dict:
    """Compare current crop definitions against existing annotations.

    Detects:
    - new_plants: plant IDs in crops but with zero annotations
    - changed_plants: plant IDs whose bbox changed since their annotations
    - removed_plants: plant IDs with annotations but no longer in crops
    - unchanged_plants: plant IDs with matching bbox and existing annotations

    Returns a dict describing the state of each plant.
    """
    current_ids = {}
    for p in current_plants:
        pid = p.get("crop_uid", p["id"])
        current_ids[pid] = p

    # Group existing annotations by stable crop identity
    ann_by_plant: dict[str, list[dict]] = {}
    for ann in existing_annotations:
        pid = ann.get("crop_uid", ann["plant_id"])
        ann_by_plant.setdefault(pid, []).append(ann)

    annotated_ids = set(ann_by_plant.keys())
    crop_ids = set(current_ids.keys())

    new_plants = []
    changed_plants = []
    removed_plants = []
    unchanged_plants = []

    for pid in crop_ids:
        if pid not in annotated_ids:
            new_plants.append(pid)
        else:
            # Check if bbox changed
            current_bbox = current_ids[pid]["bbox"]
            # All annotations for this plant should have the same crop_bbox
            old_bbox = ann_by_plant[pid][0].get("crop_bbox", None)
            if old_bbox is not None and old_bbox != current_bbox:
                changed_plants.append(pid)
            else:
                unchanged_plants.append(pid)

    for pid in annotated_ids:
        if pid not in crop_ids:
            removed_plants.append(pid)

    return {
        "new_plants": new_plants,
        "changed_plants": changed_plants,
        "removed_plants": removed_plants,
        "unchanged_plants": unchanged_plants,
    }


def remap_existing_annotations_to_current_plants(
    current_plants: list[dict],
    existing_annotations: list[dict],
) -> tuple[list[dict], int]:
    """Backfill stable crop IDs and refresh plant labels on old annotations.

    Existing annotation files may only have plant_id (gX_rYY), which can drift
    when crops are deleted/reordered. We reattach by exact (genotype, crop_bbox).
    Annotations that cannot be tied to a current crop are dropped.
    """
    key_to_plants: dict[tuple[int, tuple[int, int, int, int]], list[dict]] = {}
    for p in current_plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        key_to_plants.setdefault(key, []).append(p)

    by_uid = {p.get("crop_uid"): p for p in current_plants if p.get("crop_uid")}

    remapped: list[dict] = []
    dropped = 0
    for ann in existing_annotations:
        ann2 = dict(ann)
        matched_plant = None

        uid = ann2.get("crop_uid")
        if uid and uid in by_uid:
            matched_plant = by_uid[uid]
        elif not uid:
            bbox = ann2.get("crop_bbox")
            genotype = ann2.get("genotype")
            if bbox is not None and genotype is not None:
                key = (int(genotype), tuple(bbox))
                candidates = key_to_plants.get(key, [])
                if len(candidates) == 1:
                    matched_plant = candidates[0]

        if matched_plant is None:
            dropped += 1
            continue

        ann2["crop_uid"] = matched_plant.get("crop_uid", matched_plant["id"])
        ann2["plant_id"] = matched_plant["id"]
        ann2["replicate"] = matched_plant["replicate"]
        ann2["genotype"] = matched_plant["genotype"]
        remapped.append(ann2)

    # Deduplicate same crop/frame pairs (keep last encountered).
    dedup: dict[tuple[str, int], dict] = {}
    for ann in remapped:
        key = (ann["crop_uid"], int(ann["frame_index"]))
        dedup[key] = ann

    return list(dedup.values()), dropped


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
    force_new: bool = False,
) -> list[dict]:
    """Run the annotation loop for one plant.

    If force_new is True, ignores existing annotation count and always
    annotates images_per_folder new frames (used for changed/new plants).

    Returns list of new annotation dicts created this session.
    """
    global points, display_img, orig_img

    plant_id = plant["id"]
    crop_uid = plant.get("crop_uid", plant_id)
    bbox = plant["bbox"]

    # Find which frames are already annotated for this plant
    already_annotated = set()
    for ann in existing_annotations:
        ann_uid = ann.get("crop_uid", ann["plant_id"])
        if ann_uid == crop_uid:
            already_annotated.add(ann["frame_index"])

    available_count = len(frame_filenames) - len(already_annotated)
    if available_count == 0:
        print(f"  {plant_id}: all frames annotated, skipping.")
        return []

    if force_new:
        target = min(images_per_folder, available_count)
        print(f"  {plant_id}: force-annotating {target} frames (bbox changed or new plant).")
    else:
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
                    "crop_uid": crop_uid,
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

    # Load existing annotations
    existing_log = load_annotation_log(experiment_dir)
    existing_annotations = existing_log.get("annotations", [])
    existing_annotations, dropped_stale = remap_existing_annotations_to_current_plants(
        plants, existing_annotations
    )
    if existing_annotations:
        print(f"Existing annotations: {len(existing_annotations)}")
    if dropped_stale:
        print(f"Dropped stale annotations not tied to current crops: {dropped_stale}")

    # --- Change detection ---
    changes = detect_crop_changes(plants, existing_annotations)

    if changes["removed_plants"]:
        removed = changes["removed_plants"]
        print(f"\n  REMOVED plants (no longer in crops): {removed}")
        print(
            f"  Dropping {sum(1 for a in existing_annotations if a.get('crop_uid', a['plant_id']) in removed)} orphaned annotations."
        )
        existing_annotations = [
            a for a in existing_annotations
            if a.get("crop_uid", a["plant_id"]) not in set(removed)
        ]

    if changes["changed_plants"]:
        changed = changes["changed_plants"]
        print(f"\n  CHANGED bbox for plants: {changed}")
        print(f"  Old annotations for these plants are INVALID (bbox moved).")
        print(f"  Dropping old annotations and re-annotating.")
        existing_annotations = [
            a for a in existing_annotations
            if a.get("crop_uid", a["plant_id"]) not in set(changed)
        ]

    if changes["new_plants"]:
        print(f"\n  NEW plants needing annotation: {changes['new_plants']}")

    if changes["unchanged_plants"]:
        print(f"  Unchanged plants: {changes['unchanged_plants']}")

    needs_force = set(changes["new_plants"]) | set(changes["changed_plants"])
    if not needs_force and not any(
        max(
            0,
            IMAGES_PER_FOLDER
            - sum(
                1
                for a in existing_annotations
                if a.get("crop_uid", a["plant_id"]) == p.get("crop_uid", p["id"])
            ),
        )
        for p in plants
    ):
        print("\nAll plants fully annotated. Nothing to do.")
        # Still write log to update timestamp
        checks = sanity_check(existing_annotations)
        log_data = {
            "step": "02_annotate",
            "experiment_id": EXPERIMENT_ID,
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "images_per_folder": IMAGES_PER_FOLDER,
            "display_scale": DISPLAY_SCALE,
            "new_this_session": 0,
            "total_annotations": len(existing_annotations),
            "plants_annotated": checks["plants_annotated"],
            "plants_total": len(plants),
            "annotations": existing_annotations,
            "sanity": checks,
        }
        write_annotation_log(log_path, log_data)
        print(f"Annotation log saved to: {log_path}")
        return

    # Set up window
    cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotator", mouse_callback)

    all_new = []
    for plant in plants:
        plant_id = plant["id"]
        crop_uid = plant.get("crop_uid", plant_id)
        force = crop_uid in needs_force

        print(f"\n--- {plant_id} (genotype {plant['genotype']}) ---")
        new = annotate_plant(
            plant, raw_dir, frame_filenames,
            existing_annotations, IMAGES_PER_FOLDER, DISPLAY_SCALE,
            force_new=force,
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

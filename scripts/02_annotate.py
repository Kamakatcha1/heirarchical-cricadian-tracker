# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"EXPERIMENT_ID": "exp_002_0301"}
#
# For EXISTING experiments, settings are restored from the
# saved 02_annotate.json automatically.  Only use _OVERRIDE
# to CHANGE a saved setting.
# ============================================================
_OVERRIDE = {}
# ============================================================


import json
import random
import sys
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


# --- Globals used by the mouse callback ---
points = []
display_img = None
orig_img = None


def draw_point(img, point_xy, idx):
    """Draw a numbered circle on the display image."""
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    color = colors[idx % len(colors)]
    cv2.circle(img, tuple(point_xy), 6, color, -1)
    cv2.putText(img, str(idx + 1), (point_xy[0] + 8, point_xy[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def mouse_callback(event, x, y, flags, param):
    """Record clicks in original coords, draw in display coords."""
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        x0 = int(round(x / DISPLAY_SCALE))
        y0 = int(round(y / DISPLAY_SCALE))
        h0, w0 = orig_img.shape[:2]
        x0 = max(0, min(w0 - 1, x0))
        y0 = max(0, min(h0 - 1, y0))
        points.append([x0, y0])
        draw_point(display_img, [x, y], len(points) - 1)
        cv2.imshow("annotator", display_img)


def load_crop_log(experiment_dir: Path, expected_experiment_id: str) -> dict:
    crop_path = experiment_dir / "logs" / "01_crop.json"
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop log not found at {crop_path}. Run 01_crop.py first.")
    with open(crop_path) as f:
        data = json.load(f)
    stored_id = data.get("experiment_id", "")
    if stored_id != expected_experiment_id:
        raise RuntimeError(
            f"Experiment ID mismatch!\n"
            f"  Config says: {expected_experiment_id}\n"
            f"  01_crop.json says: {stored_id}\n"
            f"  Check _config.py EXPERIMENT_ID or re-run 01_crop.py."
        )
    return data


def load_annotation_log(experiment_dir: Path, expected_experiment_id: str) -> tuple[list[dict], dict]:
    """Load existing annotations + stored settings.

    Returns (annotations_list, settings_dict).
    Validates experiment_id matches if the file exists.
    """
    ann_path = experiment_dir / "logs" / "02_annotate.json"
    if ann_path.exists():
        with open(ann_path) as f:
            data = json.load(f)
        stored_id = data.get("experiment_id", "")
        if stored_id and stored_id != expected_experiment_id:
            raise RuntimeError(
                f"Experiment ID mismatch!\n"
                f"  Config says: {expected_experiment_id}\n"
                f"  02_annotate.json says: {stored_id}\n"
                f"  Check _config.py EXPERIMENT_ID."
            )
        settings = {}
        if "images_per_folder" in data:
            settings["images_per_folder"] = data["images_per_folder"]
        if "display_scale" in data:
            settings["display_scale"] = data["display_scale"]
        return data.get("annotations", []), settings
    return [], {}


def remap_annotations_to_plants(
    plants: list[dict],
    annotations: list[dict],
) -> tuple[list[dict], int]:
    """Reattach annotations to current plants by crop_uid or (genotype, bbox).

    Drops annotations that can't be matched. Updates plant_id/replicate
    in case replicate numbering changed.
    """
    by_uid = {p.get("crop_uid"): p for p in plants if p.get("crop_uid")}
    key_to_plants: dict[tuple, list[dict]] = {}
    for p in plants:
        key = (int(p["genotype"]), tuple(p["bbox"]))
        key_to_plants.setdefault(key, []).append(p)

    remapped, dropped = [], 0
    for ann in annotations:
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
        remapped.append(ann2)

    # Deduplicate same crop/frame pairs (keep last)
    dedup: dict[tuple, dict] = {}
    for ann in remapped:
        fi = ann.get("frame_index")
        if isinstance(fi, int):
            dedup[(ann["crop_uid"], fi)] = ann
        else:
            dropped += 1
    return list(dedup.values()), dropped


def detect_crop_changes(
    plants: list[dict],
    annotations: list[dict],
) -> dict:
    """Compare current crops against existing annotations.

    Returns dict with new_plants, changed_plants, removed_plants,
    unchanged_plants (lists of crop_uid strings).
    """
    current = {p.get("crop_uid", p["id"]): p for p in plants}

    ann_by_uid: dict[str, list[dict]] = {}
    for ann in annotations:
        uid = ann.get("crop_uid", ann["plant_id"])
        ann_by_uid.setdefault(uid, []).append(ann)

    crop_ids = set(current.keys())
    ann_ids = set(ann_by_uid.keys())

    new, changed, removed, unchanged = [], [], [], []
    for uid in crop_ids:
        if uid not in ann_ids:
            new.append(uid)
        else:
            old_bbox = ann_by_uid[uid][0].get("crop_bbox")
            if old_bbox is not None and old_bbox != current[uid]["bbox"]:
                changed.append(uid)
            else:
                unchanged.append(uid)
    for uid in ann_ids:
        if uid not in crop_ids:
            removed.append(uid)

    return {"new_plants": new, "changed_plants": changed,
            "removed_plants": removed, "unchanged_plants": unchanged}


def virtual_crop(raw_frame_path: Path, bbox: list[int]) -> "np.ndarray | None":
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
    """Select frame indices with even temporal spacing."""
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
        pool = [i for i in available if i not in selected]
        selected.extend(random.sample(pool, min(target_count - len(selected), len(pool))))

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

    If force_new, ignores existing count and annotates fresh frames.
    """
    global points, display_img, orig_img

    plant_id = plant["id"]
    crop_uid = plant.get("crop_uid", plant_id)
    bbox = plant["bbox"]

    already_annotated = set()
    for ann in existing_annotations:
        if ann.get("crop_uid", ann["plant_id"]) == crop_uid:
            already_annotated.add(ann["frame_index"])

    available_count = len(frame_filenames) - len(already_annotated)
    if available_count == 0:
        print(f"  {plant_id}: all frames annotated, skipping.")
        return []

    if force_new:
        target = min(images_per_folder, available_count)
        print(f"  {plant_id}: force-annotating {target} frames (new/changed plant).")
    else:
        target = max(0, images_per_folder - len(already_annotated))
        if target == 0:
            print(f"  {plant_id}: target met ({len(already_annotated)} >= {images_per_folder}), skipping.")
            return []
        if already_annotated:
            print(f"  {plant_id}: {len(already_annotated)} existing, need {target} more.")

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
        print("  Click leaf tips (any number). s=save  r=reset  n=skip+replace  x=skip  q=quit")

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
            if key == ord("s"):
                new_annotations.append({
                    "plant_id": plant_id,
                    "crop_uid": crop_uid,
                    "genotype": plant["genotype"],
                    "replicate": plant["replicate"],
                    "frame_index": frame_index,
                    "frame_filename": frame_filenames[frame_index],
                    "crop_bbox": bbox,
                    "crop_size": [w0, h0],
                    "tips": [pt for pt in points],
                })
                break
            if key == ord("q"):
                cv2.destroyAllWindows()
                raise SystemExit

    return new_annotations


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    global IMAGES_PER_FOLDER, DISPLAY_SCALE

    experiment_dir = Path(EXPERIMENT_DIR)
    raw_dir = Path(RAW_DIR)
    log_path = experiment_dir / "logs" / "02_annotate.json"

    # Load crop definitions (validates experiment_id matches config)
    crop_log = load_crop_log(experiment_dir, EXPERIMENT_ID)
    plants = crop_log["plants"]
    frame_filenames = crop_log["frames"]  # already a flat list of filenames
    dataset_id = crop_log["dataset_id"]

    raw_dir = Path(crop_log["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")

    # Load and remap existing annotations (validates experiment_id)
    existing, saved_settings = load_annotation_log(experiment_dir, EXPERIMENT_ID)

    # Restore settings from existing experiment (unless overridden)
    restored = []
    if "IMAGES_PER_FOLDER" not in _OVERRIDE and "images_per_folder" in saved_settings:
        saved_ipf = saved_settings["images_per_folder"]
        if saved_ipf != IMAGES_PER_FOLDER:
            IMAGES_PER_FOLDER = saved_ipf
            restored.append(f"images_per_folder={IMAGES_PER_FOLDER}")
    if "DISPLAY_SCALE" not in _OVERRIDE and "display_scale" in saved_settings:
        saved_ds = saved_settings["display_scale"]
        if saved_ds != DISPLAY_SCALE:
            DISPLAY_SCALE = saved_ds
            restored.append(f"display_scale={DISPLAY_SCALE}")
    if restored:
        print(f"Restored from existing experiment: {', '.join(restored)}")
        print("  (use _OVERRIDE to change these)")

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Dataset: {dataset_id} ({len(frame_filenames)} frames)")
    print(f"Plants: {len(plants)}")
    print(f"Target: {IMAGES_PER_FOLDER} annotations per plant")
    print(f"Display scale: {DISPLAY_SCALE}x")

    existing, dropped_stale = remap_annotations_to_plants(plants, existing)
    if existing:
        print(f"Existing annotations: {len(existing)}")
    if dropped_stale:
        print(f"Dropped stale annotations: {dropped_stale}")

    # Change detection
    changes = detect_crop_changes(plants, existing)

    if changes["removed_plants"]:
        removed = set(changes["removed_plants"])
        n = sum(1 for a in existing if a.get("crop_uid", a["plant_id"]) in removed)
        print(f"\n  REMOVED plants: {changes['removed_plants']} ({n} annotations dropped)")
        existing = [a for a in existing if a.get("crop_uid", a["plant_id"]) not in removed]

    if changes["changed_plants"]:
        changed = set(changes["changed_plants"])
        print(f"\n  CHANGED bbox: {changes['changed_plants']} (old annotations dropped, re-annotating)")
        existing = [a for a in existing if a.get("crop_uid", a["plant_id"]) not in changed]

    if changes["new_plants"]:
        print(f"\n  NEW plants needing annotation: {changes['new_plants']}")

    needs_force = set(changes["new_plants"]) | set(changes["changed_plants"])

    # Check if there's any work to do
    needs_any = bool(needs_force)
    if not needs_any:
        for p in plants:
            uid = p.get("crop_uid", p["id"])
            count = sum(1 for a in existing if a.get("crop_uid", a["plant_id"]) == uid)
            if count < IMAGES_PER_FOLDER:
                needs_any = True
                break

    if not needs_any:
        print("\nAll plants fully annotated. Nothing to do.")
        write_json(log_path, {
            "experiment_id": EXPERIMENT_ID,
            "images_per_folder": IMAGES_PER_FOLDER,
            "display_scale": DISPLAY_SCALE,
            "annotations": existing,
        })
        print(f"Annotation log saved to: {log_path}")
        return

    # Annotate
    cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotator", mouse_callback)

    all_new = []
    for plant in plants:
        uid = plant.get("crop_uid", plant["id"])
        force = uid in needs_force
        print(f"\n--- {plant['id']} (genotype {plant['genotype']}) ---")
        new = annotate_plant(
            plant, raw_dir, frame_filenames,
            existing, IMAGES_PER_FOLDER, DISPLAY_SCALE,
            force_new=force,
        )
        all_new.extend(new)
        existing.extend(new)

    cv2.destroyAllWindows()
    print(f"\nSession complete: {len(all_new)} new annotations")

    # Validate
    malformed = []
    for i, ann in enumerate(existing):
        tips = ann.get("tips", [])
        if not isinstance(tips, list):
            malformed.append(i)
            continue
        for tip in tips:
            if not (isinstance(tip, list) and len(tip) == 2):
                malformed.append(i)
                break
    if malformed:
        print(f"WARNING: {len(malformed)} malformed annotations at indices {malformed}")
    else:
        print("Validation passed.")

    # Write JSON with settings + annotations
    write_json(log_path, {
        "experiment_id": EXPERIMENT_ID,
        "images_per_folder": IMAGES_PER_FOLDER,
        "display_scale": DISPLAY_SCALE,
        "annotations": existing,
    })
    print(f"Annotation log saved to: {log_path}")
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

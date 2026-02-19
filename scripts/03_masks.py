# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"TRAINING_RUN_ID": "run_002"}
# ============================================================
_OVERRIDE = {}
# ============================================================

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

BASE_DIR              = _config.BASE_DIR
EXPERIMENTS_DIR       = _config.EXPERIMENTS_DIR
TRAINING_DIR          = _config.TRAINING_DIR
TRAINING_RUN_ID       = _config.TRAINING_RUN_ID
TRAINING_EXPERIMENTS  = _config.TRAINING_EXPERIMENTS
SIGMA                 = _config.SIGMA
AUGMENT               = _config.AUGMENT
AUGS_PER_IMAGE        = _config.AUGS_PER_IMAGE


def gaussian(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian heatmap.

    Exact logic from original make_masks.py gaussian().
    """
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def load_experiment_data(experiment_dir: Path) -> tuple[dict, dict]:
    """Load crop and annotation logs for one experiment."""
    crop_path = experiment_dir / "logs" / "01_crop.json"
    ann_path = experiment_dir / "logs" / "02_annotate.json"

    if not crop_path.exists():
        raise FileNotFoundError(f"Crop log not found: {crop_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation log not found: {ann_path}")

    with open(crop_path) as f:
        crop_log = json.load(f)
    with open(ann_path) as f:
        ann_log = json.load(f)

    return crop_log, ann_log


def collect_all_experiments(
    experiments_dir: Path,
    experiment_ids: list[str],
) -> list[tuple[str, dict, dict]]:
    """Load (experiment_id, crop_log, annotation_log) for every requested experiment."""
    sources = []
    for exp_id in experiment_ids:
        exp_dir = experiments_dir / exp_id
        try:
            crop_log, ann_log = load_experiment_data(exp_dir)
            ann_count = len(ann_log.get("annotations", []))
            sources.append((exp_id, crop_log, ann_log))
            print(f"  {exp_id}: {ann_count} annotations")
        except FileNotFoundError as e:
            print(f"  WARNING: {exp_id} -- {e}")
    return sources


def generate_mask_for_annotation(
    ann: dict,
    raw_dir: Path,
    out_img_dir: Path,
    out_mask_dir: Path,
    sigma: float,
    filename_prefix: str,
) -> bool:
    """Generate a cropped image + Gaussian tip mask for one annotation.

    Reads the raw frame, crops it using the stored bbox, generates a
    Gaussian heatmap at each tip position, and writes both the cropped
    image and the mask to disk.

    Exact mask logic from original make_masks.py.
    filename_prefix is prepended to avoid collisions across experiments.
    Returns True if successful, False if skipped.
    """
    plant_id = ann["plant_id"]
    frame_index = ann["frame_index"]
    frame_filename = ann["frame_filename"]
    bbox = ann["crop_bbox"]
    crop_size = ann["crop_size"]  # [w, h] at annotation time

    frame_path = raw_dir / frame_filename
    if not frame_path.exists():
        print(f"    Missing raw frame: {frame_path}")
        return False

    full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if full_img is None:
        print(f"    Could not read: {frame_path}")
        return False

    x0, y0, x1, y1 = bbox
    crop = full_img[y0:y1, x0:x1]

    h, w = crop.shape[:2]
    ann_w, ann_h = crop_size

    # Scale factor: annotation coords -> actual crop coords
    # (safety logic from original -- in practice these are identical)
    sx = w / float(ann_w)
    sy = h / float(ann_h)

    tips = [
        ann["points"]["leaf_tip_1"],
        ann["points"]["leaf_tip_2"],
    ]

    tip_map = np.zeros((h, w), dtype=np.float32)
    for tip in tips:
        tx = float(max(0, min(w - 1, tip[0] * sx)))
        ty = float(max(0, min(h - 1, tip[1] * sy)))
        tip_map += gaussian(h, w, tx, ty, sigma)

    tip_map = np.clip(tip_map, 0.0, 1.0)
    mask = (tip_map * 255).astype(np.uint8)

    # Filename: {exp_id}_{plant_id}_frame_{NNN}.png
    out_base = f"{filename_prefix}_{plant_id}_frame_{frame_index:03d}"
    cv2.imwrite(str(out_img_dir / f"{out_base}.png"), crop)
    cv2.imwrite(str(out_mask_dir / f"{out_base}.png"), mask)
    return True


def write_log(log_path: Path, log_data: dict) -> None:
    """Write a JSON log for this training run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def sanity_check(out_img_dir: Path, out_mask_dir: Path) -> dict:
    """Verify mask output is valid."""
    checks = {}

    images = sorted(out_img_dir.glob("*.png"))
    masks = sorted(out_mask_dir.glob("*.png"))
    checks["image_count"] = len(images)
    checks["mask_count"] = len(masks)
    checks["counts_match"] = len(images) == len(masks)

    img_stems = {p.stem for p in images}
    mask_stems = {p.stem for p in masks}
    unmatched_images = sorted(img_stems - mask_stems)
    unmatched_masks = sorted(mask_stems - img_stems)
    checks["unmatched_images"] = unmatched_images
    checks["unmatched_masks"] = unmatched_masks
    checks["all_paired"] = len(unmatched_images) == 0 and len(unmatched_masks) == 0

    checks["spot_check_passed"] = True
    if masks:
        sample = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        if sample is None:
            checks["spot_check_passed"] = False
            checks["spot_check_detail"] = "Could not read sample mask"
        elif sample.max() == 0:
            checks["spot_check_passed"] = False
            checks["spot_check_detail"] = "Sample mask is all zeros"
        else:
            checks["spot_check_detail"] = f"Sample max={sample.max()}, mean={sample.mean():.1f}"

    checks["at_least_one_pair"] = len(images) >= 1

    checks["all_passed"] = all([
        checks["counts_match"],
        checks["all_paired"],
        checks["spot_check_passed"],
        checks["at_least_one_pair"],
    ])
    return checks


def main() -> None:
    experiments_dir = Path(EXPERIMENTS_DIR)
    training_dir = Path(TRAINING_DIR)

    run_dir = training_dir / TRAINING_RUN_ID
    out_img_dir = run_dir / "images"
    out_mask_dir = run_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {TRAINING_RUN_ID}")
    print(f"Output: {run_dir}")
    print(f"Sigma: {SIGMA}")
    if AUGMENT:
        # NOTE: Augmentation is not yet implemented.
        print(f"Augmentation: ENABLED ({AUGS_PER_IMAGE} per image) -- NOT YET IMPLEMENTED")
    else:
        print("Augmentation: disabled")
    print(f"\nExperiments to combine ({len(TRAINING_EXPERIMENTS)}):")

    # Load all requested experiments
    sources = collect_all_experiments(experiments_dir, TRAINING_EXPERIMENTS)
    if not sources:
        raise SystemExit(
            "No valid experiments found. Check TRAINING_EXPERIMENTS in _config.py.\n"
            "Each experiment needs logs/01_crop.json and logs/02_annotate.json."
        )

    generated = 0
    skipped = 0
    per_experiment = {}

    for exp_id, crop_log, ann_log in sources:
        raw_dir = Path(crop_log["raw_dir"])
        if not raw_dir.exists():
            print(f"\n  WARNING: Raw dir missing for {exp_id}: {raw_dir}")
            continue

        annotations = ann_log.get("annotations", [])
        if not annotations:
            print(f"\n  {exp_id}: no annotations, skipping")
            continue

        print(f"\n  Processing {exp_id} ({len(annotations)} annotations, raw: {raw_dir.name})...")
        count = 0
        for ann in annotations:
            if generate_mask_for_annotation(ann, raw_dir, out_img_dir, out_mask_dir, SIGMA, exp_id):
                generated += 1
                count += 1
            else:
                skipped += 1

        per_experiment[exp_id] = count

    print(f"\n{'='*50}")
    print(f"Total: {generated} image+mask pairs, {skipped} skipped")
    print(f"Images -> {out_img_dir}")
    print(f"Masks  -> {out_mask_dir}")

    # Sanity check
    checks = sanity_check(out_img_dir, out_mask_dir)
    if not checks["all_passed"]:
        print("\n*** SANITY CHECK FAILED ***")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        print("Aborting.")
        sys.exit(1)
    print("Sanity checks passed.")

    # Write log into the training run folder (not an experiment folder)
    log_data = {
        "training_run": TRAINING_RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "experiments": TRAINING_EXPERIMENTS,
        "sigma": SIGMA,
        "augment": AUGMENT,
        "augs_per_image": AUGS_PER_IMAGE if AUGMENT else 0,
        "generated": generated,
        "skipped": skipped,
        "per_experiment": per_experiment,
        "out_img_dir": str(out_img_dir),
        "out_mask_dir": str(out_mask_dir),
        "sanity": checks,
    }
    write_log(run_dir / "run_log.json", log_data)
    print(f"Log written to: {run_dir / 'run_log.json'}")


if __name__ == "__main__":
    main()

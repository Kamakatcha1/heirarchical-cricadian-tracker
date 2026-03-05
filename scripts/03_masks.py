# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# Set keys here to override specific config values for this
# script only.  e.g. {"TRAINING_EXPERIMENTS": ["exp_001_0218", "exp_002_0301"]}
# ============================================================
_OVERRIDE = {}
# ============================================================

import json
import random
import shutil
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
TRAINING_EXPERIMENTS  = _config.TRAINING_EXPERIMENTS
GENOTYPE_FILTER       = _config.GENOTYPE_FILTER
SIGMA                 = _config.SIGMA
AUGMENT               = _config.AUGMENT
AUGS_PER_IMAGE        = _config.AUGS_PER_IMAGE
AUG_MAX_ROTATE        = _config.AUG_MAX_ROTATE
AUG_MIN_SCALE         = _config.AUG_MIN_SCALE
AUG_MAX_SCALE         = _config.AUG_MAX_SCALE
AUG_MAX_SHIFT         = _config.AUG_MAX_SHIFT
AUG_HFLIP_PROB        = _config.AUG_HFLIP_PROB
AUG_VFLIP_PROB        = _config.AUG_VFLIP_PROB
AUG_BRIGHTNESS_ALPHA  = _config.AUG_BRIGHTNESS_ALPHA
AUG_BRIGHTNESS_BETA   = _config.AUG_BRIGHTNESS_BETA
AUG_SEED              = _config.AUG_SEED


# ---- Gaussian heatmap ----

def gaussian(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


# ---- Augmentation (cv2-based, matching original augment_dataset.py) ----

def random_affine_matrix(h: int, w: int, max_rotate: float, min_scale: float,
                         max_scale: float, max_shift: float) -> np.ndarray:
    """Build a random 2x3 affine matrix (rotation + scale + translation)."""
    angle = random.uniform(-max_rotate, max_rotate)
    scale = random.uniform(min_scale, max_scale)
    dx = random.uniform(-max_shift, max_shift) * w
    dy = random.uniform(-max_shift, max_shift) * h
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def augment_pair(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random augmentation to an image+mask pair.

    Augmentations (matching original augment_dataset.py):
    - Horizontal flip (prob AUG_HFLIP_PROB)
    - Vertical flip (prob AUG_VFLIP_PROB)
    - Random affine (rotation, scale, translation)
    - Brightness/contrast jitter (image only, not mask)
    """
    h, w = img.shape[:2]
    aug_img = img.copy()
    aug_mask = mask.copy()

    # Horizontal flip
    if random.random() < AUG_HFLIP_PROB:
        aug_img = cv2.flip(aug_img, 1)
        aug_mask = cv2.flip(aug_mask, 1)

    # Vertical flip
    if random.random() < AUG_VFLIP_PROB:
        aug_img = cv2.flip(aug_img, 0)
        aug_mask = cv2.flip(aug_mask, 0)

    # Random affine
    M = random_affine_matrix(h, w, AUG_MAX_ROTATE, AUG_MIN_SCALE,
                             AUG_MAX_SCALE, AUG_MAX_SHIFT)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    aug_mask = cv2.warpAffine(aug_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Brightness/contrast jitter (image only)
    alpha = random.uniform(AUG_BRIGHTNESS_ALPHA[0], AUG_BRIGHTNESS_ALPHA[1])
    beta = random.uniform(AUG_BRIGHTNESS_BETA[0], AUG_BRIGHTNESS_BETA[1])
    aug_img = np.clip(aug_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return aug_img, aug_mask


# ---- Experiment loading ----

def load_experiment_data(experiment_dir: Path, expected_id: str) -> tuple[dict, dict]:
    """Load crop + annotation JSONs for one experiment.

    Validates that experiment_id inside each JSON matches the folder name.
    """
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

    # Validate experiment IDs match the folder we're reading from
    crop_eid = crop_log.get("experiment_id", "")
    ann_eid = ann_log.get("experiment_id", "")
    if crop_eid and crop_eid != expected_id:
        raise RuntimeError(
            f"Experiment ID mismatch in {crop_path}!\n"
            f"  Folder: {expected_id}, JSON says: {crop_eid}"
        )
    if ann_eid and ann_eid != expected_id:
        raise RuntimeError(
            f"Experiment ID mismatch in {ann_path}!\n"
            f"  Folder: {expected_id}, JSON says: {ann_eid}"
        )

    return crop_log, ann_log


def collect_all_experiments(
    experiments_dir: Path,
    experiment_ids: list[str],
) -> list[tuple[str, dict, dict]]:
    sources = []
    for exp_id in experiment_ids:
        exp_dir = experiments_dir / exp_id
        try:
            crop_log, ann_log = load_experiment_data(exp_dir, exp_id)
            n = len(ann_log.get("annotations", []))
            sources.append((exp_id, crop_log, ann_log))
            print(f"  {exp_id}: {n} annotations")
        except FileNotFoundError as e:
            print(f"  WARNING: {exp_id} -- {e}")
    return sources


# ---- Generation ----

def generate_pair(ann: dict, raw_dir: Path, sigma: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate a cropped image + Gaussian tip mask for one annotation.

    Returns (crop_image, mask_image) or None on failure.
    """
    bbox = ann["crop_bbox"]
    crop_size = ann["crop_size"]  # [w, h] at annotation time
    frame_path = raw_dir / ann["frame_filename"]

    if not frame_path.exists():
        print(f"    Missing raw frame: {frame_path}")
        return None

    full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if full_img is None:
        print(f"    Could not read: {frame_path}")
        return None

    x0, y0, x1, y1 = bbox
    crop = full_img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    ann_w, ann_h = crop_size

    sx = w / float(ann_w)
    sy = h / float(ann_h)

    tips = ann.get("tips", [])
    if not tips:
        print(f"    No tips annotated for {ann.get('plant_id', '?')}, skipping")
        return None
    tip_map = np.zeros((h, w), dtype=np.float32)
    for tip in tips:
        tx = float(max(0, min(w - 1, tip[0] * sx)))
        ty = float(max(0, min(h - 1, tip[1] * sy)))
        tip_map += gaussian(h, w, tx, ty, sigma)

    tip_map = np.clip(tip_map, 0.0, 1.0)
    mask = (tip_map * 255).astype(np.uint8)
    return crop, mask


def write_pair(img: np.ndarray, mask: np.ndarray,
               out_img_dir: Path, out_mask_dir: Path, basename: str) -> None:
    cv2.imwrite(str(out_img_dir / f"{basename}.png"), img)
    cv2.imwrite(str(out_mask_dir / f"{basename}.png"), mask)


# ---- Main ----

def main() -> None:
    experiments_dir = Path(EXPERIMENTS_DIR)
    training_dir = Path(TRAINING_DIR)

    # Flat training folder -- wipe and recreate
    out_img_dir = training_dir / "images"
    out_mask_dir = training_dir / "masks"
    if training_dir.exists():
        # Only remove images/ and masks/ subfolders, leave 03_masks.json intact until we overwrite
        if out_img_dir.exists():
            shutil.rmtree(out_img_dir)
        if out_mask_dir.exists():
            shutil.rmtree(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {training_dir}")
    print(f"Sigma: {SIGMA}")
    if GENOTYPE_FILTER:
        print(f"Genotype filter: {GENOTYPE_FILTER}")
    else:
        print("Genotype filter: all genotypes")
    if AUGMENT:
        print(f"Augmentation: ENABLED ({AUGS_PER_IMAGE} per image)")
    else:
        print("Augmentation: disabled")

    # Seed RNG
    if AUG_SEED > 0:
        random.seed(AUG_SEED)
        np.random.seed(AUG_SEED)

    print(f"\nExperiments to combine ({len(TRAINING_EXPERIMENTS)}):")
    sources = collect_all_experiments(experiments_dir, TRAINING_EXPERIMENTS)
    if not sources:
        raise SystemExit(
            "No valid experiments found. Check TRAINING_EXPERIMENTS in _config.py.\n"
            "Each experiment needs logs/01_crop.json and logs/02_annotate.json."
        )

    generated = 0
    augmented = 0
    skipped = 0
    per_experiment = {}

    for exp_id, crop_log, ann_log in sources:
        raw_dir = Path(crop_log["raw_dir"])
        if not raw_dir.exists():
            print(f"\n  WARNING: Raw dir missing for {exp_id}: {raw_dir}")
            continue

        annotations = ann_log.get("annotations", [])
        if GENOTYPE_FILTER:
            annotations = [a for a in annotations if a.get("genotype") in GENOTYPE_FILTER]
        if not annotations:
            print(f"\n  {exp_id}: no annotations (after filtering), skipping")
            continue

        print(f"\n  Processing {exp_id} ({len(annotations)} annotations)...")
        count = 0
        genotypes_used = set()
        for ann in annotations:
            genotypes_used.add(ann.get("genotype"))
            result = generate_pair(ann, raw_dir, SIGMA)
            if result is None:
                skipped += 1
                continue

            crop_img, mask_img = result
            basename = f"{exp_id}_{ann['plant_id']}_frame_{ann['frame_index']:03d}"

            # Write original pair
            write_pair(crop_img, mask_img, out_img_dir, out_mask_dir, basename)
            generated += 1
            count += 1

            # Write augmented copies
            if AUGMENT:
                for aug_i in range(AUGS_PER_IMAGE):
                    aug_img, aug_mask = augment_pair(crop_img, mask_img)
                    aug_name = f"{basename}_aug{aug_i:02d}"
                    write_pair(aug_img, aug_mask, out_img_dir, out_mask_dir, aug_name)
                    augmented += 1

        per_experiment[exp_id] = {
            "dataset_id": crop_log.get("dataset_id", ""),
            "annotations": count,
            "genotypes": sorted(genotypes_used),
        }

    total_files = generated + augmented
    print(f"\n{'='*50}")
    print(f"Originals: {generated}, Augmented: {augmented}, Total: {total_files}, Skipped: {skipped}")
    print(f"Images -> {out_img_dir}")
    print(f"Masks  -> {out_mask_dir}")

    # Sanity check
    images = sorted(out_img_dir.glob("*.png"))
    masks = sorted(out_mask_dir.glob("*.png"))
    img_stems = {p.stem for p in images}
    mask_stems = {p.stem for p in masks}
    ok = True
    if len(images) != len(masks):
        print(f"ERROR: image count ({len(images)}) != mask count ({len(masks)})"); ok = False
    if img_stems != mask_stems:
        print(f"ERROR: unmatched files"); ok = False
    if len(images) == 0:
        print("ERROR: no output files"); ok = False
    if ok and masks:
        sample = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        if sample is not None and sample.max() == 0:
            print("WARNING: sample mask is all zeros")
    if not ok:
        sys.exit(1)
    print("Validation passed.")

    # Write training manifest (provenance for what's in the training folder)
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "genotype_filter": GENOTYPE_FILTER if GENOTYPE_FILTER else "all",
        "sigma": SIGMA,
        "augment": AUGMENT,
        "augs_per_image": AUGS_PER_IMAGE if AUGMENT else 0,
        "total_pairs": total_files,
        "experiments": per_experiment,
    }
    manifest_path = training_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

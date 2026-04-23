from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import hct_runtime as rt

MASK_SIGMA = rt.MASK_DEFAULTS["sigma"]
MASK_AUGMENTATION = {
    "aug_max_rotate": rt.MASK_DEFAULTS["aug_max_rotate"],
    "aug_min_scale": rt.MASK_DEFAULTS["aug_min_scale"],
    "aug_max_scale": rt.MASK_DEFAULTS["aug_max_scale"],
    "aug_max_shift": rt.MASK_DEFAULTS["aug_max_shift"],
    "aug_hflip_prob": rt.MASK_DEFAULTS["aug_hflip_prob"],
    "aug_vflip_prob": rt.MASK_DEFAULTS["aug_vflip_prob"],
    "aug_brightness_alpha_min": rt.MASK_DEFAULTS["aug_brightness_alpha_min"],
    "aug_brightness_alpha_max": rt.MASK_DEFAULTS["aug_brightness_alpha_max"],
    "aug_brightness_beta_min": rt.MASK_DEFAULTS["aug_brightness_beta_min"],
    "aug_brightness_beta_max": rt.MASK_DEFAULTS["aug_brightness_beta_max"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate training masks from annotated datasets.",
        epilog=(
            "Examples:\n"
            "  ./hct masks --augment false\n"
            "  ./hct masks --datasets F2_001,F2_002 --augment true --augmentations-per-image 4\n"
            "  ./hct masks --datasets F2_001,F2_002 --genotype-filter 0,4,7"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--datasets", type=rt.parse_dataset_ids_arg, help="Comma-separated dataset ids to include.")
    parser.add_argument("--augment", type=rt.parse_bool_flag, help="Enable or disable augmentation.")
    parser.add_argument("--augmentations-per-image", type=int, help="Augmented copies per base image.")
    parser.add_argument("--genotype-filter", type=rt.parse_int_csv, help="Comma-separated genotype ids to keep.")
    parser.add_argument("--aug-seed", type=int, help="Optional random seed for deterministic augmentation.")
    return parser


def gaussian(height: int, width: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


def random_affine_matrix(height: int, width: int, settings: dict[str, Any]) -> np.ndarray:
    angle = random.uniform(-settings["aug_max_rotate"], settings["aug_max_rotate"])
    scale = random.uniform(settings["aug_min_scale"], settings["aug_max_scale"])
    dx = random.uniform(-settings["aug_max_shift"], settings["aug_max_shift"]) * width
    dy = random.uniform(-settings["aug_max_shift"], settings["aug_max_shift"]) * height
    cx, cy = width / 2.0, height / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix


def augment_pair(img: np.ndarray, mask: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    aug_img = img.copy()
    aug_mask = mask.copy()

    if random.random() < settings["aug_hflip_prob"]:
        aug_img = cv2.flip(aug_img, 1)
        aug_mask = cv2.flip(aug_mask, 1)
    if random.random() < settings["aug_vflip_prob"]:
        aug_img = cv2.flip(aug_img, 0)
        aug_mask = cv2.flip(aug_mask, 0)

    matrix = random_affine_matrix(height, width, settings)
    aug_img = cv2.warpAffine(aug_img, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    aug_mask = cv2.warpAffine(aug_mask, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    alpha = random.uniform(settings["aug_brightness_alpha_min"], settings["aug_brightness_alpha_max"])
    beta = random.uniform(settings["aug_brightness_beta_min"], settings["aug_brightness_beta_max"])
    aug_img = np.clip(aug_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return aug_img, aug_mask


def generate_pair(annotation: dict[str, Any], raw_dir: Path, sigma: float, bbox: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    crop_size = annotation["crop_size"]
    frame_path = raw_dir / annotation["frame_filename"]
    if not frame_path.exists():
        print(f"    Missing raw frame: {frame_path}")
        return None

    full_img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if full_img is None:
        print(f"    Could not read: {frame_path}")
        return None

    x0, y0, x1, y1 = bbox
    crop = full_img[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    ann_w, ann_h = crop_size
    sx = width / float(ann_w)
    sy = height / float(ann_h)

    tips = annotation.get("tips", [])
    if not tips:
        print(f"    No tips annotated for {annotation.get('plant_id', '?')}, skipping")
        return None

    tip_map = np.zeros((height, width), dtype=np.float32)
    for tip in tips:
        tx = float(max(0, min(width - 1, tip[0] * sx)))
        ty = float(max(0, min(height - 1, tip[1] * sy)))
        tip_map += gaussian(height, width, tx, ty, sigma)

    tip_map = np.clip(tip_map, 0.0, 1.0)
    mask = (tip_map * 255).astype(np.uint8)
    return crop, mask


def write_pair(img: np.ndarray, mask: np.ndarray, out_img_dir: Path, out_mask_dir: Path, basename: str) -> None:
    cv2.imwrite(str(out_img_dir / f"{basename}.png"), img)
    cv2.imwrite(str(out_mask_dir / f"{basename}.png"), mask)


def resolve_datasets(args: argparse.Namespace) -> list[rt.DatasetInfo]:
    datasets = [info for info in rt.discover_datasets() if info.has_annotations]
    if args.datasets:
        chosen = []
        by_id = {info.dataset_id: info for info in datasets}
        missing = [ds for ds in args.datasets if ds not in by_id]
        if missing:
            raise SystemExit(f"Datasets not found or missing annotations: {', '.join(missing)}")
        for ds_id in args.datasets:
            chosen.append(by_id[ds_id])
        return chosen

    if args.batch:
        raise SystemExit("--batch requires --datasets")

    def render(info: rt.DatasetInfo) -> str:
        return f"{info.dataset_id} ({info.plant_count} plants, {info.annotation_count} annotations)"

    return rt.prompt_select_many("Available datasets (with annotations):", datasets, render)


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()
    datasets = resolve_datasets(args)

    augment = args.augment if args.augment is not None else rt.MASK_DEFAULTS["augment"]
    augmentations_per_image = (
        args.augmentations_per_image
        if args.augmentations_per_image is not None
        else rt.MASK_DEFAULTS["augmentations_per_image"]
    )

    settings = {
        "sigma": MASK_SIGMA,
        "augment": augment,
        "augmentations_per_image": augmentations_per_image,
        "genotype_filter": args.genotype_filter or [],
        **MASK_AUGMENTATION,
        "aug_seed": args.aug_seed if args.aug_seed is not None else rt.MASK_DEFAULTS["aug_seed"],
    }

    if settings["aug_seed"] is not None:
        random.seed(settings["aug_seed"])
        np.random.seed(settings["aug_seed"])

    training_root = rt.training_dir()
    out_img_dir = training_root / "images"
    out_mask_dir = training_root / "masks"
    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)
    if out_mask_dir.exists():
        shutil.rmtree(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training data from {[info.dataset_id for info in datasets]}")
    print(f"Output: {training_root}")
    print(f"Sigma: {settings['sigma']}")
    print(f"Genotype filter: {settings['genotype_filter'] or 'all'}")
    if settings["augment"]:
        print(f"Augmentation: ENABLED ({settings['augmentations_per_image']} per image)")
    else:
        print("Augmentation: disabled")

    generated = 0
    augmented = 0
    skipped = 0
    manifest_datasets: dict[str, Any] = {}

    for dataset in datasets:
        crop_log = rt.load_json(dataset.crop_path, {})
        ann_log = rt.load_json(dataset.annotation_path, {})
        raw_dir = dataset.raw_dir
        if not raw_dir.exists():
            print(f"\n  WARNING: Raw dir missing for {dataset.dataset_id}: {raw_dir}")
            continue

        annotations = ann_log.get("annotations", [])
        plant_by_uid = {
            str(plant.get("crop_uid")): plant
            for plant in crop_log.get("plants", [])
            if isinstance(plant, dict) and plant.get("crop_uid")
        }
        if settings["genotype_filter"]:
            annotations = [ann for ann in annotations if ann.get("genotype") in settings["genotype_filter"]]
        if not annotations:
            print(f"\n  {dataset.dataset_id}: no annotations after filtering, skipping")
            continue

        count = 0
        genotypes_used: set[int] = set()
        print(f"\n  Processing {dataset.dataset_id} ({len(annotations)} annotations)...")
        for ann in annotations:
            genotypes_used.add(int(ann.get("genotype", 0)))
            bbox = ann.get("crop_bbox")
            crop_uid = str(ann.get("crop_uid", ""))
            current_plant = plant_by_uid.get(crop_uid)
            if current_plant is not None:
                current_bbox = list(current_plant.get("bbox", []))
                if bbox is not None and current_bbox and list(bbox) != current_bbox:
                    print(f"    WARNING: crop changed since annotation for {ann.get('plant_id', '?')} ({crop_uid}); skipping")
                    skipped += 1
                    continue
                bbox = current_bbox
            if not isinstance(bbox, list) or len(bbox) != 4:
                print(f"    WARNING: missing or invalid crop bbox for {ann.get('plant_id', '?')}, skipping")
                skipped += 1
                continue

            result = generate_pair(ann, raw_dir, settings["sigma"], bbox)
            if result is None:
                skipped += 1
                continue
            crop_img, mask_img = result
            basename = f"{dataset.dataset_id}_{ann['plant_id']}_frame_{ann['frame_index']:03d}"
            write_pair(crop_img, mask_img, out_img_dir, out_mask_dir, basename)
            generated += 1
            count += 1

            if settings["augment"]:
                for aug_idx in range(settings["augmentations_per_image"]):
                    aug_img, aug_mask = augment_pair(crop_img, mask_img, settings)
                    aug_name = f"{basename}_aug{aug_idx:02d}"
                    write_pair(aug_img, aug_mask, out_img_dir, out_mask_dir, aug_name)
                    augmented += 1

        manifest_datasets[dataset.dataset_id] = {
            "raw_dir": str(raw_dir),
            "annotations": count,
            "genotypes": sorted(genotypes_used),
            "frame_count": len(crop_log.get("frames", [])),
            "plant_count": len(crop_log.get("plants", [])),
        }

    images = sorted(out_img_dir.glob("*.png"))
    masks = sorted(out_mask_dir.glob("*.png"))
    if len(images) != len(masks):
        raise SystemExit(f"image count ({len(images)}) != mask count ({len(masks)})")
    if not images:
        raise SystemExit("No output files were generated.")
    if {p.stem for p in images} != {p.stem for p in masks}:
        raise SystemExit("Image/mask filename mismatch.")

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "datasets": [info.dataset_id for info in datasets],
        "sigma": settings["sigma"],
        "augment": settings["augment"],
        "augmentations_per_image": settings["augmentations_per_image"],
        "genotype_filter": settings["genotype_filter"],
        "total_pairs": generated + augmented,
        "original_pairs": generated,
        "augmented_pairs": augmented,
        "datasets_info": manifest_datasets,
        "settings": settings,
    }
    rt.write_json(training_root / "manifest.json", manifest)

    print(f"\n{'=' * 50}")
    print(f"Originals: {generated}, Augmented: {augmented}, Total: {generated + augmented}, Skipped: {skipped}")
    print(f"Images -> {out_img_dir}")
    print(f"Masks  -> {out_mask_dir}")
    print(f"Manifest -> {training_root / 'manifest.json'}")


if __name__ == "__main__":
    main()

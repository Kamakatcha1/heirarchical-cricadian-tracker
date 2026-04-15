from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import hct_runtime as rt


points: list[list[int]] = []
display_img: np.ndarray | None = None
orig_img: np.ndarray | None = None
display_scale_value = rt.ANNOTATE_DEFAULTS["display_scale"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate leaf tips for a dataset.")
    parser.add_argument("--batch", action="store_true", help="Disable prompts and require CLI values.")
    parser.add_argument("--dataset", help="Dataset id under data/datasets/.")
    parser.add_argument("--images-per-plant", type=int, help="Target annotation count per plant.")
    parser.add_argument("--display-scale", type=int, help="Zoom factor for the annotation window.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for frame selection.")
    return parser


def draw_point(img: np.ndarray, point_xy: list[int], idx: int) -> None:
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    color = colors[idx % len(colors)]
    cv2.circle(img, tuple(point_xy), 6, color, -1)
    cv2.putText(img, str(idx + 1), (point_xy[0] + 8, point_xy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def mouse_callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
    global points, display_img, orig_img
    if event != cv2.EVENT_LBUTTONDOWN or orig_img is None or display_img is None:
        return
    x0 = int(round(x / display_scale_value))
    y0 = int(round(y / display_scale_value))
    h0, w0 = orig_img.shape[:2]
    x0 = max(0, min(w0 - 1, x0))
    y0 = max(0, min(h0 - 1, y0))
    points.append([x0, y0])
    draw_point(display_img, [x, y], len(points) - 1)
    cv2.imshow("annotator", display_img)


def load_crop_log(path: Path, dataset_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Crop log not found at {path}. Run 01_crop.py first.")
    data = rt.load_json(path, {})
    stored_id = data.get("dataset_id", "")
    if stored_id and stored_id != dataset_id:
        raise RuntimeError(f"Dataset id mismatch: selected {dataset_id}, crops.json says {stored_id}")
    return data


def load_annotation_log(path: Path, dataset_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {}
    data = rt.load_json(path, {})
    stored_id = data.get("dataset_id", "")
    if stored_id and stored_id != dataset_id:
        raise RuntimeError(f"Dataset id mismatch: selected {dataset_id}, annotations.json says {stored_id}")
    settings = {}
    if "images_per_folder" in data:
        settings["images_per_folder"] = data["images_per_folder"]
    if "display_scale" in data:
        settings["display_scale"] = data["display_scale"]
    return data.get("annotations", []), settings


def remap_annotations_to_plants(plants: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_uid = {plant.get("crop_uid"): plant for plant in plants if plant.get("crop_uid")}
    key_to_plants: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    for plant in plants:
        key = (int(plant["genotype"]), tuple(plant["bbox"]))
        key_to_plants.setdefault(key, []).append(plant)

    remapped: list[dict[str, Any]] = []
    dropped = 0
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
                matches = key_to_plants.get((int(geno), tuple(bbox)), [])
                if len(matches) == 1:
                    matched = matches[0]
        if matched is None:
            dropped += 1
            continue
        ann2["crop_uid"] = matched.get("crop_uid", matched["id"])
        ann2["plant_id"] = matched["id"]
        ann2["replicate"] = matched["replicate"]
        ann2["genotype"] = matched["genotype"]
        remapped.append(ann2)

    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for ann in remapped:
        frame_index = ann.get("frame_index")
        if isinstance(frame_index, int):
            dedup[(ann["crop_uid"], frame_index)] = ann
        else:
            dropped += 1
    return list(dedup.values()), dropped


def detect_crop_changes(plants: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, list[str]]:
    current = {str(plant.get("crop_uid", plant["id"])): plant for plant in plants}
    ann_by_uid: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        uid = str(ann.get("crop_uid", ann["plant_id"]))
        ann_by_uid.setdefault(uid, []).append(ann)

    crop_ids = set(current.keys())
    ann_ids = set(ann_by_uid.keys())
    new: list[str] = []
    changed: list[str] = []
    removed: list[str] = []
    unchanged: list[str] = []

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

    return {
        "new_plants": new,
        "changed_plants": changed,
        "removed_plants": removed,
        "unchanged_plants": unchanged,
    }


def virtual_crop(raw_frame_path: Path, bbox: list[int]) -> np.ndarray | None:
    img = cv2.imread(str(raw_frame_path))
    if img is None:
        return None
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


def select_frames(frame_filenames: list[str], target_count: int, already_annotated: set[int]) -> tuple[list[int], list[int]]:
    available = [idx for idx in range(len(frame_filenames)) if idx not in already_annotated]
    if not available:
        return [], []
    target_count = min(target_count, len(available))
    if target_count == 0:
        return [], available

    n = len(available)
    selected: list[int] = []
    for idx in range(target_count):
        start = int(idx * n / target_count)
        end = int((idx + 1) * n / target_count)
        segment = available[start:end]
        if segment:
            selected.append(random.choice(segment))

    if len(selected) < target_count:
        pool = [idx for idx in available if idx not in selected]
        selected.extend(random.sample(pool, min(target_count - len(selected), len(pool))))

    remaining = [idx for idx in available if idx not in selected]
    return selected, remaining


def annotate_plant(
    plant: dict[str, Any],
    raw_dir: Path,
    frame_filenames: list[str],
    existing_annotations: list[dict[str, Any]],
    images_per_plant: int,
    force_new: bool = False,
) -> list[dict[str, Any]]:
    global points, display_img, orig_img

    plant_id = plant["id"]
    crop_uid = plant.get("crop_uid", plant_id)
    bbox = plant["bbox"]

    already_annotated: set[int] = set()
    for ann in existing_annotations:
        if ann.get("crop_uid", ann["plant_id"]) == crop_uid:
            already_annotated.add(int(ann["frame_index"]))

    available_count = len(frame_filenames) - len(already_annotated)
    if available_count == 0:
        print(f"  {plant_id}: all frames annotated, skipping.")
        return []

    if force_new:
        target = min(images_per_plant, available_count)
        print(f"  {plant_id}: force-annotating {target} frames (new/changed plant).")
    else:
        target = max(0, images_per_plant - len(already_annotated))
        if target == 0:
            print(f"  {plant_id}: target met ({len(already_annotated)} >= {images_per_plant}), skipping.")
            return []
        if already_annotated:
            print(f"  {plant_id}: {len(already_annotated)} existing, need {target} more.")

    selected, remaining = select_frames(frame_filenames, target, already_annotated)
    new_annotations: list[dict[str, Any]] = []
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
        disp_w, disp_h = int(w0 * display_scale_value), int(h0 * display_scale_value)
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
                new_annotations.append(
                    {
                        "plant_id": plant_id,
                        "crop_uid": crop_uid,
                        "genotype": plant["genotype"],
                        "replicate": plant["replicate"],
                        "frame_index": frame_index,
                        "frame_filename": frame_filenames[frame_index],
                        "crop_bbox": bbox,
                        "crop_size": [w0, h0],
                        "tips": [pt for pt in points],
                    }
                )
                break
            if key == ord("q"):
                cv2.destroyAllWindows()
                raise SystemExit

    return new_annotations


def select_dataset(args: argparse.Namespace) -> rt.DatasetInfo:
    datasets = [info for info in rt.discover_datasets() if info.has_crops]
    if args.dataset:
        matches = [info for info in datasets if info.dataset_id == args.dataset]
        if not matches:
            raise SystemExit(f"Dataset not found or missing crops: {args.dataset}")
        return matches[0]

    if args.batch:
        raise SystemExit("--batch requires --dataset")

    last_dataset = rt.get_default("last_dataset")
    default_index = next((idx for idx, info in enumerate(datasets) if info.dataset_id == last_dataset), None)

    def render(info: rt.DatasetInfo) -> str:
        return f"{info.dataset_id} ({info.plant_count} plants, {info.fully_annotated_plants}/{info.plant_count} fully annotated)"

    return rt.prompt_select_one("Available datasets (with crops):", datasets, render, default_index=default_index)


def main() -> None:
    global display_scale_value

    args = build_parser().parse_args()
    rt.ensure_layout()
    if args.seed is not None:
        random.seed(args.seed)

    dataset = select_dataset(args)
    crop_log = load_crop_log(dataset.crop_path, dataset.dataset_id)
    plants = crop_log["plants"]
    frame_filenames = crop_log["frames"]

    if not dataset.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {dataset.raw_dir}")

    existing, saved_settings = load_annotation_log(dataset.annotation_path, dataset.dataset_id)
    images_per_plant = args.images_per_plant
    if images_per_plant is None:
        images_per_plant = saved_settings.get("images_per_folder", rt.ANNOTATE_DEFAULTS["images_per_plant"])

    display_scale = args.display_scale
    if display_scale is None:
        display_scale = saved_settings.get("display_scale", rt.ANNOTATE_DEFAULTS["display_scale"])
    display_scale_value = display_scale

    print(f"Dataset: {dataset.dataset_id} ({len(frame_filenames)} frames)")
    print(f"Plants: {len(plants)}")
    print(f"Target: {images_per_plant} annotations per plant")
    print(f"Display scale: {display_scale}x")

    existing, dropped_stale = remap_annotations_to_plants(plants, existing)
    if existing:
        print(f"Existing annotations: {len(existing)}")
    if dropped_stale:
        print(f"Dropped stale annotations: {dropped_stale}")

    changes = detect_crop_changes(plants, existing)
    if changes["removed_plants"]:
        removed = set(changes["removed_plants"])
        n_removed = sum(1 for ann in existing if ann.get("crop_uid", ann["plant_id"]) in removed)
        print(f"\n  REMOVED plants: {changes['removed_plants']} ({n_removed} annotations dropped)")
        existing = [ann for ann in existing if ann.get("crop_uid", ann["plant_id"]) not in removed]

    if changes["changed_plants"]:
        changed = set(changes["changed_plants"])
        print(f"\n  CHANGED bbox: {changes['changed_plants']} (old annotations dropped, re-annotating)")
        existing = [ann for ann in existing if ann.get("crop_uid", ann["plant_id"]) not in changed]

    if changes["new_plants"]:
        print(f"\n  NEW plants needing annotation: {changes['new_plants']}")

    needs_force = set(changes["new_plants"]) | set(changes["changed_plants"])
    needs_any = bool(needs_force)
    if not needs_any:
        for plant in plants:
            uid = plant.get("crop_uid", plant["id"])
            count = sum(1 for ann in existing if ann.get("crop_uid", ann["plant_id"]) == uid)
            if count < images_per_plant:
                needs_any = True
                break

    if not needs_any:
        print("\nAll plants fully annotated. Nothing to do.")
        rt.write_json(
            dataset.annotation_path,
            {
                "dataset_id": dataset.dataset_id,
                "images_per_folder": images_per_plant,
                "display_scale": display_scale,
                "annotations": existing,
            },
        )
        rt.save_defaults_after_success(last_dataset=dataset.dataset_id)
        print(f"Annotation log saved to: {dataset.annotation_path}")
        return

    cv2.namedWindow("annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("annotator", mouse_callback)

    all_new: list[dict[str, Any]] = []
    for plant in plants:
        uid = plant.get("crop_uid", plant["id"])
        force = uid in needs_force
        print(f"\n--- {plant['id']} (genotype {plant['genotype']}) ---")
        new_annotations = annotate_plant(
            plant,
            dataset.raw_dir,
            frame_filenames,
            existing,
            images_per_plant,
            force_new=force,
        )
        all_new.extend(new_annotations)
        existing.extend(new_annotations)

    cv2.destroyAllWindows()
    print(f"\nSession complete: {len(all_new)} new annotations")

    malformed: list[int] = []
    for idx, ann in enumerate(existing):
        tips = ann.get("tips", [])
        if not isinstance(tips, list):
            malformed.append(idx)
            continue
        for tip in tips:
            if not (isinstance(tip, list) and len(tip) == 2):
                malformed.append(idx)
                break
    if malformed:
        print(f"WARNING: {len(malformed)} malformed annotations at indices {malformed}")
    else:
        print("Validation passed.")

    rt.write_json(
        dataset.annotation_path,
        {
            "dataset_id": dataset.dataset_id,
            "images_per_folder": images_per_plant,
            "display_scale": display_scale,
            "annotations": existing,
        },
    )
    rt.save_defaults_after_success(last_dataset=dataset.dataset_id)
    print(f"Annotation log saved to: {dataset.annotation_path}")
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

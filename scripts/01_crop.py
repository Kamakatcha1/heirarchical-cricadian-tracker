from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

import hct_runtime as rt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop plants for a dataset.")
    parser.add_argument("--batch", action="store_true", help="Disable prompts and require CLI values.")
    parser.add_argument("--dataset", help="Dataset id under data/datasets/.")
    parser.add_argument("--max-frames", type=int, help="Use only the first N frames (0 = all).")
    return parser


def collect_raw_frames(raw_dir: Path, max_frames: int = 0) -> list[Path]:
    paths = rt.image_files(raw_dir, max_frames=max_frames)
    if not paths:
        raise FileNotFoundError(f"No images found in raw dataset: {raw_dir}")
    return paths


def blend_composite(image_paths: list[Path], method: str) -> np.ndarray:
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


def load_existing_crop_log(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plants_to_rects(plants: list[dict[str, Any]]) -> list[tuple[tuple[int, int, int, int], int]]:
    rects = []
    for plant in plants:
        x0, y0, x1, y1 = plant["bbox"]
        rects.append(((x0, y0, x1, y1), int(plant["genotype"])))
    return rects


def select_rectangles(
    img: np.ndarray,
    max_display: int,
    expected: int,
    existing_rects: list[tuple[tuple[int, int, int, int], int]] | None = None,
) -> list[tuple[tuple[int, int, int, int], int]]:
    h, w = img.shape[:2]
    scale = 1.0
    max_side = max(h, w)
    if max_side > max_display:
        scale = max_display / float(max_side)
    preview = img if scale == 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)))

    window = "Draw rects | 0-9=genotype | d=delete | u=undo | c=clear | s=save | q=quit"
    rects: list[tuple[tuple[int, int, int, int], int]] = list(existing_rects or [])
    drawing = False
    start = (0, 0)
    current = (0, 0)
    selected_idx = -1

    def point_in_rect(px: int, py: int, rect: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = rect
        return x0 <= px <= x1 and y0 <= py <= y1

    def redraw() -> None:
        canvas = preview.copy()
        for idx, ((x0, y0, x1, y1), geno) in enumerate(rects):
            color = (0, 255, 255) if idx == selected_idx else (0, 255, 0)
            thickness = 3 if idx == selected_idx else 2
            cv2.rectangle(
                canvas,
                (int(x0 * scale), int(y0 * scale)),
                (int(x1 * scale), int(y1 * scale)),
                color,
                thickness,
            )
            label = "G?"
            if geno >= 0:
                rep = sum(1 for _, g in rects[: idx + 1] if g == geno)
                label = f"g{geno}_r{rep:02d}"
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

    def callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
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
                for idx, (rect, _geno) in enumerate(rects):
                    scaled = tuple(int(v * scale) for v in rect)
                    if point_in_rect(x, y, scaled):  # type: ignore[arg-type]
                        selected_idx = idx
                        return
                selected_idx = -1
                return
            x0, x1 = sorted([x0, x1])
            y0, y1 = sorted([y0, y1])
            rects.append(
                (
                    (
                        int(round(x0 / scale)),
                        int(round(y0 / scale)),
                        int(round(x1 / scale)),
                        int(round(y1 / scale)),
                    ),
                    -1,
                )
            )
            selected_idx = len(rects) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            for idx, (rect, _geno) in enumerate(rects):
                scaled = tuple(int(v * scale) for v in rect)
                if point_in_rect(x, y, scaled):  # type: ignore[arg-type]
                    selected_idx = idx
                    return
            selected_idx = -1

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, preview)
    cv2.setMouseCallback(window, callback)

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
        if ord("0") <= key <= ord("9"):
            geno = int(chr(key))
            target = selected_idx if 0 <= selected_idx < len(rects) else (len(rects) - 1 if rects else -1)
            if target >= 0:
                rect, _ = rects[target]
                rects[target] = (rect, geno)
        if key in (ord("s"), ord("S"), 13):
            if expected and rects and len(rects) != expected:
                print(f"Expected {expected} rectangles, got {len(rects)}. Keep drawing.")
                continue
            missing = [idx for idx, (_, geno) in enumerate(rects, start=1) if geno < 0]
            if missing:
                print(f"Rectangles missing genotype (press 0-9): {missing}")
                continue
            break

    cv2.destroyAllWindows()
    if cancelled:
        raise RuntimeError("Rectangle selection cancelled")
    return rects


def assign_replicate_numbers(rects: list[tuple[tuple[int, int, int, int], int]]) -> list[dict[str, Any]]:
    counts: Counter[int] = Counter()
    plants = []
    for (x0, y0, x1, y1), geno in rects:
        counts[geno] += 1
        rep = counts[geno]
        plants.append(
            {
                "id": f"g{geno}_r{rep:02d}",
                "genotype": geno,
                "replicate": rep,
                "bbox": [x0, y0, x1, y1],
            }
        )
    return plants


def assign_stable_crop_uids(plants: list[dict[str, Any]], existing_plants: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    existing: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for plant in existing_plants or []:
        uid = plant.get("crop_uid")
        if not uid:
            continue
        key = (int(plant["genotype"]), tuple(plant["bbox"]))
        existing.setdefault(key, []).append(uid)

    for plant in plants:
        key = (int(plant["genotype"]), tuple(plant["bbox"]))
        if existing.get(key):
            plant["crop_uid"] = existing[key].pop(0)
        else:
            plant["crop_uid"] = uuid4().hex
    return plants


def sync_annotations(annotation_path: Path, dataset_id: str, plants: list[dict[str, Any]]) -> dict[str, int] | None:
    if not annotation_path.exists():
        return None

    ann_log = rt.load_json(annotation_path, {})
    annotations = ann_log.get("annotations", [])
    by_uid = {plant.get("crop_uid"): plant for plant in plants if plant.get("crop_uid")}
    key_to_plants: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    for plant in plants:
        key = (int(plant["genotype"]), tuple(plant["bbox"]))
        key_to_plants.setdefault(key, []).append(plant)

    kept: list[dict[str, Any]] = []
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
        kept.append(ann2)

    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for ann in kept:
        frame_index = ann.get("frame_index")
        if isinstance(frame_index, int):
            dedup[(ann["crop_uid"], frame_index)] = ann
        else:
            dropped += 1

    out = {key: value for key, value in ann_log.items() if key != "annotations"}
    out["dataset_id"] = dataset_id
    out["annotations"] = list(dedup.values())
    rt.write_json(annotation_path, out)
    return {"kept": len(dedup), "dropped": dropped}


def select_dataset(args: argparse.Namespace) -> rt.DatasetInfo:
    datasets = rt.discover_datasets()
    if args.dataset:
        matches = [info for info in datasets if info.dataset_id == args.dataset]
        if not matches:
            raise SystemExit(f"Dataset not found: {args.dataset}")
        return matches[0]

    if args.batch:
        raise SystemExit("--batch requires --dataset")

    last_dataset = rt.get_default("last_dataset")
    default_index = next((idx for idx, info in enumerate(datasets) if info.dataset_id == last_dataset), None)

    def render(info: rt.DatasetInfo) -> str:
        crop_status = "has crops" if info.has_crops else "no crops yet"
        return f"{info.dataset_id} ({info.raw_frame_count} frames, {crop_status})"

    return rt.prompt_select_one("Available datasets:", datasets, render, default_index=default_index)


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()

    dataset = select_dataset(args)
    crop_path = dataset.crop_path
    annotation_path = dataset.annotation_path
    existing_log = load_existing_crop_log(crop_path)
    existing_plants = existing_log.get("plants") if existing_log else None

    default_max_frames = args.max_frames
    if default_max_frames is None:
        default_max_frames = (existing_log or {}).get("max_frames", rt.CROP_DEFAULTS["max_frames"])
    max_frames = args.max_frames if args.max_frames is not None else default_max_frames
    blend_method = rt.CROP_DEFAULTS["blend_method"]
    expected_plants = rt.CROP_DEFAULTS["expected_plants"]
    max_display = rt.CROP_DEFAULTS["max_display"]

    raw_frames = collect_raw_frames(dataset.raw_dir, max_frames)

    print(f"Dataset: {dataset.dataset_id}")
    if max_frames > 0:
        print(f"Frame limit: first {max_frames} frames")
    print(f"Using {len(raw_frames)} raw frames from {dataset.raw_dir}")

    existing_rects = None
    if existing_plants:
        print(f"\nFound {len(existing_plants)} existing crop definitions:")
        for plant in existing_plants:
            x0, y0, x1, y1 = plant["bbox"]
            print(f"  {plant['id']}: ({x0},{y0})-({x1},{y1})  {x1 - x0}x{y1 - y0}px")
        print("These will be pre-loaded. You can add, delete (d), or redraw.")
        existing_rects = plants_to_rects(existing_plants)

    print(f"\nBlending composite in memory (method: {blend_method})...")
    composite = blend_composite(raw_frames, blend_method)
    print(f"Composite: {composite.shape[1]}x{composite.shape[0]} (WxH)")

    print("\nDraw bounding rectangles on the composite.")
    print("  Drag to draw, right-click or tiny-drag to select existing")
    print("  0-9 = assign genotype (to selected or last drawn)")
    print("  d = delete selected, u = undo last, c = clear all")
    print("  s or Enter to save, q or Esc to cancel")
    rects = select_rectangles(composite, max_display, expected_plants, existing_rects)

    plants = assign_replicate_numbers(rects)
    plants = assign_stable_crop_uids(plants, existing_plants)

    print(f"\n{len(plants)} plants defined:")
    for plant in plants:
        x0, y0, x1, y1 = plant["bbox"]
        print(f"  {plant['id']}: ({x0},{y0})-({x1},{y1})  {x1 - x0}x{y1 - y0}px")

    genotypes = [plant["genotype"] for plant in plants]
    areas = [(p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) for p in plants]
    ids = [plant["id"] for plant in plants]
    if not all(0 <= int(geno) <= 9 for geno in genotypes):
        print("ERROR: Invalid genotype found.")
        sys.exit(1)
    if any(area <= 0 for area in areas):
        print("ERROR: Zero-area rectangle found.")
        sys.exit(1)
    if len(ids) != len(set(ids)):
        print("ERROR: Duplicate plant IDs.")
        sys.exit(1)
    print("Validation passed.")

    log_data = {
        "dataset_id": dataset.dataset_id,
        "raw_dir": str(dataset.raw_dir),
        "max_frames": max_frames,
        "blend_method": blend_method,
        "frames": [path.name for path in raw_frames],
        "plants": plants,
    }
    rt.write_json(crop_path, log_data)
    print(f"\nCrop definitions saved to: {crop_path}")

    sync_result = sync_annotations(annotation_path, dataset.dataset_id, plants)
    if sync_result is not None:
        print(f"Synced annotations: kept={sync_result['kept']}, dropped={sync_result['dropped']}")

    rt.save_defaults_after_success(last_dataset=dataset.dataset_id)
    print("No images were written -- coordinates only.")


if __name__ == "__main__":
    main()

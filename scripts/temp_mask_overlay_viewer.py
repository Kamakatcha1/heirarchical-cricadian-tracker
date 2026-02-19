import argparse
from pathlib import Path

import cv2
import numpy as np


def list_pngs(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.glob("*.png"))}


def resolve_dirs(run_dir: Path | None, images_dir: Path | None, masks_dir: Path | None) -> tuple[Path, Path]:
    if images_dir is not None and masks_dir is not None:
        return images_dir, masks_dir

    if run_dir is None:
        run_dir = Path("data/training/run_001")

    return run_dir / "images", run_dir / "masks"


def make_overlay(image_bgr: np.ndarray, mask_gray: np.ndarray, alpha: float) -> np.ndarray:
    if mask_gray.shape[:2] != image_bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(image_bgr, dtype=np.uint8)
    color_mask[:, :, 2] = mask_gray  # red channel

    return cv2.addWeighted(image_bgr, 1.0, color_mask, alpha, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary sanity check viewer for image/mask overlays.")
    parser.add_argument("--run-dir", type=Path, default=Path("data/training/run_001"), help="Training run dir containing images/ and masks/")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional explicit images dir")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Optional explicit masks dir")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity (0.0-1.0)")
    args = parser.parse_args()

    images_dir, masks_dir = resolve_dirs(args.run_dir, args.images_dir, args.masks_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_dir}")

    images = list_pngs(images_dir)
    masks = list_pngs(masks_dir)
    common = sorted(set(images.keys()) & set(masks.keys()))

    if not common:
        raise SystemExit("No matching image/mask .png pairs found by filename stem.")

    only_images = sorted(set(images.keys()) - set(masks.keys()))
    only_masks = sorted(set(masks.keys()) - set(images.keys()))
    if only_images:
        print(f"Warning: {len(only_images)} images without masks")
    if only_masks:
        print(f"Warning: {len(only_masks)} masks without images")

    window = "mask overlay sanity check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    idx = 0
    total = len(common)
    print("Controls: n=next, b=back, q or esc=quit")

    while True:
        stem = common[idx]
        img = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None:
            print(f"Skipping unreadable pair: {stem}")
            idx = (idx + 1) % total
            continue

        overlay = make_overlay(img, msk, float(np.clip(args.alpha, 0.0, 1.0)))
        header = f"{idx + 1}/{total}  {stem}"
        cv2.putText(overlay, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, overlay)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            idx = (idx + 1) % total
        elif key == ord("b"):
            idx = (idx - 1) % total
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

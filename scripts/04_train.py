from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import hct_runtime as rt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model from generated training data.")
    parser.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-name", help="Optional model folder name under data/models/.")
    parser.add_argument("--epochs", type=int, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate.")
    parser.add_argument("--val-split", type=float, help="Validation split fraction.")
    parser.add_argument("--patience", type=int, help="Early stopping patience.")
    parser.add_argument("--img-size", type=int, help="Square image size.")
    parser.add_argument("--wmse-alpha", type=float, help="Weighted MSE alpha.")
    parser.add_argument("--dice-weight", type=float, help="Soft dice loss weight.")
    parser.add_argument("--fn-weight", type=float, help="False negative penalty weight.")
    parser.add_argument("--train-seed", type=int, help="Train/validation split seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing model folder for today.")
    return parser


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def tensorflow_device_summary() -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    details = []
    for gpu in gpus:
        detail = {"name": gpu.name, "device_type": gpu.device_type}
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
        except Exception:
            gpu_details = {}
        device_name = gpu_details.get("device_name")
        compute_capability = gpu_details.get("compute_capability")
        if device_name:
            detail["device_name"] = device_name
        if compute_capability:
            detail["compute_capability"] = str(compute_capability)
        details.append(detail)
    return {
        "visible_gpu_count": len(gpus),
        "visible_gpus": details,
        "visible_cpu_count": len(cpus),
        "python_cpu_count": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def print_device_summary(summary: dict[str, Any]) -> None:
    print("\nTensorFlow device check:")
    print(f"  Visible GPUs: {summary['visible_gpu_count']}")
    if summary["visible_gpus"]:
        for idx, gpu in enumerate(summary["visible_gpus"], start=1):
            device_name = gpu.get("device_name") or gpu["name"]
            compute_capability = gpu.get("compute_capability")
            suffix = f", compute capability {compute_capability}" if compute_capability else ""
            print(f"  GPU {idx}: {device_name}{suffix}")
    else:
        print("  WARNING: TensorFlow sees no GPUs. Training will run on CPU.")
    print(f"  Visible CPU devices: {summary['visible_cpu_count']}")
    print(f"  Python CPU count: {summary['python_cpu_count']}")
    print(f"  SLURM_CPUS_PER_TASK: {summary['slurm_cpus_per_task'] or 'not set'}")
    print(f"  CUDA_VISIBLE_DEVICES: {summary['cuda_visible_devices'] or 'not set'}")


def load_sample_np(img_path_str: str, img_dir: str, mask_dir: str, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    base = os.path.basename(img_path_str).replace(".png", "")
    mask_path = os.path.join(mask_dir, f"{base}.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    img_bgr = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path_str)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask = mask.astype(np.float32) / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    mask = np.expand_dims(mask, axis=-1)
    if mask.max() < 0.05:
        raise ValueError(f"Mask too faint after resize for {base}: max={mask.max()}")
    return img, mask


def tf_load_sample(path: tf.Tensor, img_dir: str, mask_dir: str, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    img, mask = tf.numpy_function(
        func=lambda p: load_sample_np(p.decode("utf-8"), img_dir, mask_dir, img_size),
        inp=[path],
        Tout=[tf.float32, tf.float32],
    )
    img.set_shape([img_size, img_size, 3])
    mask.set_shape([img_size, img_size, 1])
    return img, mask


def make_dataset(paths: list[str], training: bool, batch_size: int, seed: int, img_dir: str, mask_dir: str, img_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda path: tf_load_sample(path, img_dir, mask_dir, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def unet(img_size: int) -> tf.keras.Model:
    inputs = layers.Input((img_size, img_size, 3))
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPool2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPool2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPool2D()(c3)
    c4 = conv_block(p3, 512)
    u3 = layers.UpSampling2D()(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = conv_block(u3, 256)
    u2 = layers.UpSampling2D()(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = conv_block(u2, 128)
    u1 = layers.UpSampling2D()(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = conv_block(u1, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)
    return models.Model(inputs, outputs)


def weighted_mse(alpha: float) -> Any:
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        weights = 1.0 + alpha * y_true
        return tf.reduce_mean(weights * tf.square(y_true - y_pred))

    return loss


def soft_dice_loss(eps: float = 1e-6) -> Any:
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return tf.reduce_mean(1.0 - dice)

    return loss


def false_negative_loss(eps: float = 1e-6) -> Any:
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(y_true > 0.1, tf.float32)
        n_pos = tf.reduce_sum(mask) + eps
        return tf.reduce_sum(mask * tf.square(y_true - y_pred)) / n_pos

    return loss


def dice_coef(eps: float = 1e-6) -> Any:
    def metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return tf.reduce_mean(dice)

    return metric


def build_dice_metric(eps: float = 1e-6) -> Any:
    def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        return tf.reduce_mean((2.0 * inter + eps) / (denom + eps))
    return dice_coef


def resolve_hparam(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    default = rt.TRAIN_DEFAULTS[name]
    if value is not None:
        return value
    return default


def main() -> None:
    args = build_parser().parse_args()
    rt.ensure_layout()
    configure_gpu_memory_growth()
    device_summary = tensorflow_device_summary()
    print_device_summary(device_summary)
    if device_summary["visible_gpu_count"] == 0 and device_summary.get("cuda_visible_devices"):
        raise SystemExit(
            "SLURM assigned CUDA_VISIBLE_DEVICES but TensorFlow detected 0 GPUs. "
            "Your container/image is missing required GPU libraries for TensorFlow. "
            "Use a GPU-enabled image via --container-image or HCT_CONTAINER_IMAGE."
        )

    manifest_path = rt.training_dir() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}\nRun 03_masks.py first.")
    manifest = rt.load_json(manifest_path, {})

    img_dir = str(rt.training_dir() / "images")
    mask_dir = str(rt.training_dir() / "masks")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Training images not found: {img_dir}\nRun 03_masks.py first.")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Training masks not found: {mask_dir}\nRun 03_masks.py first.")

    images = sorted(os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.lower().endswith(".png"))
    original_pairs = int(manifest.get("original_pairs", len(images)))
    if original_pairs < 20:
        raise ValueError(
            f"Training data is too small ({original_pairs} original pairs). Generate at least 20 annotated image/mask pairs first."
        )

    img_size = resolve_hparam(args, "img_size")
    batch_size = resolve_hparam(args, "batch_size")
    epochs = resolve_hparam(args, "epochs")
    learning_rate = resolve_hparam(args, "learning_rate")
    val_split = resolve_hparam(args, "val_split")
    patience = resolve_hparam(args, "patience")
    wmse_alpha = resolve_hparam(args, "wmse_alpha")
    dice_weight = resolve_hparam(args, "dice_weight")
    fn_weight = resolve_hparam(args, "fn_weight")
    train_seed = args.train_seed if args.train_seed is not None else rt.TRAIN_DEFAULTS["train_seed"]

    default_model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    cli_model_name = args.model_name or os.environ.get("model_name")
    if cli_model_name:
        model_name = cli_model_name.strip()
        if not model_name:
            raise SystemExit("Model name cannot be empty.")
    else:
        model_name = default_model_name
    model_dir = rt.models_dir() / model_name
    if model_dir.exists():
        if args.batch and not args.overwrite:
            raise SystemExit(f"Model folder already exists: {model_dir}. Re-run with --overwrite.")
        if not args.batch and not args.overwrite:
            if not rt.prompt_yes_no(f"Model folder {model_name} already exists. Overwrite?", default=False):
                raise SystemExit("Cancelled.")
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training data: {len(images)} pairs in {rt.training_dir()}")
    print(f"Trained from: {', '.join(manifest.get('datasets', []))}")
    print(f"Saving model to: {model_dir}")

    aug_re = re.compile(r"_aug\d+$")
    groups: dict[str, list[str]] = {}
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        base = aug_re.sub("", stem)
        groups.setdefault(base, []).append(path)
    group_keys = sorted(groups.keys())
    if len(group_keys) < 2:
        raise ValueError("Not enough unique base images for train/val split.")

    rng = random.Random(train_seed)
    rng.shuffle(group_keys)
    val_count = max(1, int(len(group_keys) * val_split))
    val_groups = set(group_keys[:val_count])
    train_images = [path for base, paths in groups.items() if base not in val_groups for path in paths]
    val_images = [path for base, paths in groups.items() if base in val_groups for path in paths]

    print(f"Base groups: {len(group_keys)} ({len(group_keys) - val_count} train, {val_count} val)")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    train_ds = make_dataset(train_images, True, batch_size, train_seed, img_dir, mask_dir, img_size)
    val_ds = make_dataset(val_images, False, batch_size, train_seed, img_dir, mask_dir, img_size)

    model = unet(img_size)
    dice_metric = build_dice_metric()
    _wmse = weighted_mse(wmse_alpha)
    _dice = soft_dice_loss()
    _fn = false_negative_loss()

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return _wmse(y_true, y_pred) + dice_weight * _dice(y_true, y_pred) + fn_weight * _fn(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=combined_loss, metrics=[dice_metric])
    model.summary()

    best_path = str(model_dir / "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_dice_coef", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    hist = history.history
    best_epoch_index = int(np.argmax(hist.get("val_dice_coef", [0]))) if hist.get("val_dice_coef") else 0
    best_epoch_number = best_epoch_index + 1
    best_val_dice = float(hist["val_dice_coef"][best_epoch_index]) if hist.get("val_dice_coef") else None
    final_train_loss = float(hist["loss"][-1]) if hist.get("loss") else None
    final_val_loss = float(hist["val_loss"][-1]) if hist.get("val_loss") else None

    training_info = {
        "trained": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "best_model": "best.keras",
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": len(hist.get("loss", [])),
        "best_epoch": best_epoch_number,
        "best_val_dice": best_val_dice,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "patience": patience,
        "train_seed": train_seed,
        "wmse_alpha": wmse_alpha,
        "dice_weight": dice_weight,
        "fn_weight": fn_weight,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "tensorflow_devices": device_summary,
        "training_manifest": manifest,
    }
    rt.write_json(model_dir / "training_info.json", training_info)

    print(f"\nBest model: {best_path}")
    print(f"Training info: {model_dir / 'training_info.json'}")
    if best_val_dice is not None:
        print(f"Best val dice: {best_val_dice:.4f} (epoch {best_epoch_number})")


if __name__ == "__main__":
    main()

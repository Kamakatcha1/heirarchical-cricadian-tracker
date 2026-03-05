# ============================================================
# OVERRIDE -- leave empty to use values from _config.py
# ============================================================
_OVERRIDE = {}
# ============================================================

import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Load central config, apply overrides ---
import _config
for _k, _v in _OVERRIDE.items():
    setattr(_config, _k, _v)

TRAINING_DIR   = _config.TRAINING_DIR
EXPERIMENT_DIR = _config.EXPERIMENT_DIR
IMG_SIZE       = _config.IMG_SIZE
BATCH_SIZE     = _config.BATCH_SIZE
EPOCHS         = _config.EPOCHS
LR             = _config.LEARNING_RATE
VAL_SPLIT      = _config.VAL_SPLIT
PATIENCE       = _config.PATIENCE
SEED           = _config.TRAIN_SEED
WMSE_ALPHA     = _config.WMSE_ALPHA
DICE_WEIGHT    = _config.DICE_WEIGHT


# ---- Data loading ----

IMG_DIR  = str(Path(TRAINING_DIR) / "images")
MASK_DIR = str(Path(TRAINING_DIR) / "masks")


def load_sample_np(img_path_str: str):
    """Load one image+mask pair, resize to IMG_SIZE, normalise to [0,1]."""
    base = os.path.basename(img_path_str).replace(".png", "")
    mask_path = os.path.join(MASK_DIR, f"{base}.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    img_bgr = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path_str)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    # INTER_LINEAR preserves Gaussian shape better than INTER_AREA for masks
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    mask = mask.astype(np.float32) / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    mask = np.expand_dims(mask, axis=-1)

    if mask.max() < 0.05:
        raise ValueError(f"Mask too faint after resize for {base}: max={mask.max()}")

    return img, mask


def tf_load_sample(path):
    img, mask = tf.numpy_function(
        func=lambda p: load_sample_np(p.decode("utf-8")),
        inp=[path],
        Tout=[tf.float32, tf.float32],
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, mask


def make_dataset(paths, training: bool):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(tf_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---- Model ----

def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

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


# ---- Loss / Metrics ----

def weighted_mse(alpha=50.0):
    def loss(y_true, y_pred):
        w = 1.0 + alpha * y_true
        return tf.reduce_mean(w * tf.square(y_true - y_pred))
    return loss


def soft_dice_loss(eps=1e-6):
    def loss(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return tf.reduce_mean(1.0 - dice)
    return loss


def dice_coef(eps=1e-6):
    def metric(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return tf.reduce_mean(dice)
    return metric


# ---- Helpers ----

# ---- Main ----

def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = experiment_dir / "models" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model run: {run_name}")

    # Collect image files from training folder
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Training images not found: {IMG_DIR}\nRun 03_masks.py first.")
    if not os.path.isdir(MASK_DIR):
        raise FileNotFoundError(f"Training masks not found: {MASK_DIR}\nRun 03_masks.py first.")

    images = sorted(
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith(".png")
    )

    if len(images) < 2:
        raise ValueError(f"Not enough images found ({len(images)}). Need at least 2.")

    print(f"Training data: {IMG_DIR}")
    print(f"  {len(images)} image+mask pairs")
    print(f"  IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
    print(f"  LR={LR}, VAL_SPLIT={VAL_SPLIT}, PATIENCE={PATIENCE}")
    print(f"  WMSE_ALPHA={WMSE_ALPHA}, DICE_WEIGHT={DICE_WEIGHT}")

    # Group by base name (strip _augNN suffix) to prevent data leakage
    aug_re = re.compile(r"_aug\d+$")
    groups: dict[str, list[str]] = {}
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        base = aug_re.sub("", stem)
        groups.setdefault(base, []).append(path)

    group_keys = sorted(groups.keys())
    if len(group_keys) < 2:
        raise ValueError("Not enough unique base images for train/val split.")

    # Split by base group so augmented variants stay together
    rng = random.Random(SEED)
    rng.shuffle(group_keys)
    val_count = max(1, int(len(group_keys) * VAL_SPLIT))
    val_groups = set(group_keys[:val_count])

    train_images = [p for base, paths in groups.items() if base not in val_groups for p in paths]
    val_images = [p for base, paths in groups.items() if base in val_groups for p in paths]

    print(f"\n  Base groups: {len(group_keys)} ({len(group_keys) - val_count} train, {val_count} val)")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images:   {len(val_images)}")

    train_ds = make_dataset(train_images, training=True)
    val_ds = make_dataset(val_images, training=False)

    # Build model
    model = unet()
    loss_fn = lambda y_true, y_pred: (
        weighted_mse(alpha=WMSE_ALPHA)(y_true, y_pred)
        + DICE_WEIGHT * soft_dice_loss()(y_true, y_pred)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=loss_fn,
        metrics=[dice_coef()],
    )
    model.summary()

    # Callbacks
    best_path = str(model_dir / "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.5,
            patience=max(2, PATIENCE // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final model
    final_path = str(model_dir / "final.keras")
    model.save(final_path)
    print(f"\nModels saved to: {model_dir}")
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")

    # Extract metrics from history
    hist = history.history
    best_epoch = int(np.argmax(hist.get("val_dice_coef", [0])))
    best_val_dice = float(hist["val_dice_coef"][best_epoch]) if "val_dice_coef" in hist else None
    final_train_loss = float(hist["loss"][-1]) if "loss" in hist else None
    final_val_loss = float(hist["val_loss"][-1]) if "val_loss" in hist else None

    print(f"\n  Best val dice: {best_val_dice:.4f} (epoch {best_epoch + 1})")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss:   {final_val_loss:.4f}")

    # Save training info (provenance + metrics) alongside models
    training_info = {
        "trained": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": _config.EXPERIMENT_ID,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_run": len(hist.get("loss", [])),
        "best_epoch": best_epoch + 1,
        "best_val_dice": best_val_dice,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "learning_rate": LR,
        "wmse_alpha": WMSE_ALPHA,
        "dice_weight": DICE_WEIGHT,
        "train_images": len(train_images),
        "val_images": len(val_images),
    }
    # Copy training manifest from training folder if it exists
    manifest_path = Path(TRAINING_DIR) / "training_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            training_info["training_data"] = json.load(f)
    info_path = model_dir / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"  Info:  {info_path}")


if __name__ == "__main__":
    main()

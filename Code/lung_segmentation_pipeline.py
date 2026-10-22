#!/usr/bin/env python3
"""
Lung segmentation pipeline consolidated from the supplied project scripts.

Stages:
  1. Dataset analysis and filename pairing
  2. Image/mask preprocessing
  3. Memory-efficient Keras Sequence data generator
  4. U-Net model matching Code/session6_unet.py
  5. BCE + Dice loss, Dice/IoU metrics, Adam optimizer
  6. Training with checkpointing, early stopping, LR reduction and CSV logs
  7. Per-image test evaluation, summary JSON/CSV, selected visualizations

Expected dataset layout:
  DATASET_ROOT/
    Darwin/
      img/
      mask/

This script deliberately does not include dataset files. Review all metrics as
preliminary/internal: the project's current fixed split has known cross-split
duplicate-image and patient/study-identifier overlap. A leakage-safe grouped
split and retraining are Phase 2 work.

Example:
  python lung_segmentation_pipeline.py --dataset-root "/path/to/Medical AI" --mode analyze
  python lung_segmentation_pipeline.py --dataset-root "/path/to/Medical AI" --mode all
  python lung_segmentation_pipeline.py --dataset-root "/path/to/Medical AI" --mode evaluate \
      --checkpoint "/path/to/best_resume.weights.h5"

Dependencies are listed in requirements.txt. The script was consolidated from
the supplied source files; verify it in your own environment before relying on
its outputs. It has not been run against the full dataset in this packaging step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMG_SIZE = 512
SEED = 42
MASK_THRESHOLD = 127
PREDICTION_THRESHOLD = 0.5


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_image_names(folder: Path) -> List[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def paired_filenames(image_dir: Path, mask_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    image_names = list_image_names(image_dir)
    mask_names = list_image_names(mask_dir)
    image_set, mask_set = set(image_names), set(mask_names)
    paired = sorted(image_set & mask_set)
    images_without_masks = sorted(image_set - mask_set)
    masks_without_images = sorted(mask_set - image_set)
    if not paired:
        raise RuntimeError(
            "No matching image/mask filenames found. Expected exact filename matches."
        )
    return paired, images_without_masks, masks_without_images


def read_pair(
    filename: str,
    image_dir: Path,
    mask_dir: Path,
    image_size: int = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one X-ray and mask as float32 arrays shaped (H, W, 1)."""
    image_path = image_dir / filename
    mask_path = mask_dir / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # Nearest-neighbor keeps mask labels discrete when resizing.
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    x = image.astype(np.float32) / 255.0
    y = (mask > MASK_THRESHOLD).astype(np.float32)
    return x[..., np.newaxis], y[..., np.newaxis]


def analyze_dataset(image_dir: Path, mask_dir: Path, output_dir: Path) -> List[str]:
    paired, images_without_masks, masks_without_images = paired_filenames(image_dir, mask_dir)
    empty_masks = []
    bad_pairs = []
    image_shapes = {}
    mask_shapes = {}
    foreground_pixels = []

    for i, name in enumerate(paired, start=1):
        image = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            bad_pairs.append(name)
            continue
        image_shapes[str(tuple(image.shape))] = image_shapes.get(str(tuple(image.shape)), 0) + 1
        mask_shapes[str(tuple(mask.shape))] = mask_shapes.get(str(tuple(mask.shape)), 0) + 1
        mask_bin = mask > MASK_THRESHOLD
        fg = int(mask_bin.sum())
        foreground_pixels.append(fg)
        if fg == 0:
            empty_masks.append(name)
        if i % 500 == 0:
            print(f"Analysis: {i}/{len(paired)} pairs inspected")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "image_directory": str(image_dir),
        "mask_directory": str(mask_dir),
        "image_files": len(list_image_names(image_dir)),
        "mask_files": len(list_image_names(mask_dir)),
        "paired_files": len(paired),
        "images_without_masks": images_without_masks,
        "masks_without_images": masks_without_images,
        "unreadable_pairs": bad_pairs,
        "empty_masks_after_threshold": empty_masks,
        "image_shape_counts": image_shapes,
        "mask_shape_counts": mask_shapes,
        "mask_threshold": MASK_THRESHOLD,
        "foreground_pixels_min": int(min(foreground_pixels)) if foreground_pixels else None,
        "foreground_pixels_median": float(np.median(foreground_pixels)) if foreground_pixels else None,
        "foreground_pixels_max": int(max(foreground_pixels)) if foreground_pixels else None,
        "note": "This is file-level analysis, not a patient-level leakage audit."
    }
    with (output_dir / "dataset_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "image_files": report["image_files"],
        "mask_files": report["mask_files"],
        "paired_files": report["paired_files"],
        "unreadable_pairs": len(bad_pairs),
        "empty_masks": len(empty_masks),
        "images_without_masks": len(images_without_masks),
        "masks_without_images": len(masks_without_images),
    }, indent=2))
    return paired


def load_or_create_split(
    paired: List[str],
    split_file: Path,
    seed: int = SEED,
    allow_create: bool = True,
) -> Dict[str, List[str]]:
    """Use an existing split unchanged; otherwise create a seeded file-level split."""
    if split_file.is_file():
        with split_file.open("r", encoding="utf-8") as f:
            split = json.load(f)
        train = list(split.get("train", []))
        val = list(split.get("validation", split.get("val", [])))
        test = list(split.get("test", []))
        all_names = train + val + test
        if len(all_names) != len(set(all_names)):
            raise ValueError("The split JSON contains filenames in multiple partitions.")
        missing = sorted(set(all_names) - set(paired))
        if missing:
            raise ValueError(
                f"Split references {len(missing)} files not found in paired dataset; "
                f"example: {missing[0]}"
            )
        if not train or not val or not test:
            raise ValueError("Train, validation and test partitions must all be non-empty.")
        print(f"Using existing split unchanged: {split_file}")
        print(f"Split sizes: train={len(train)}, validation={len(val)}, test={len(test)}")
        print("WARNING: existing split has known cross-split duplicate/group overlap.")
        return {"train": train, "validation": val, "test": test}

    if not allow_create:
        raise FileNotFoundError(f"Split file not found: {split_file}")
    rng = random.Random(seed)
    names = list(paired)
    rng.shuffle(names)
    n = len(names)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split = {
        "train": names[:n_train],
        "validation": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"Created random file-level split: {split_file}")
    print("WARNING: this is NOT patient/study-grouped and must not be treated as leakage-safe.")
    return split


class LungDataSequence(tf.keras.utils.Sequence):
    """Loads only one batch into memory; avoids allocating full X/Y arrays."""
    def __init__(
        self,
        filenames: Sequence[str],
        image_dir: Path,
        mask_dir: Path,
        batch_size: int = 2,
        shuffle: bool = False,
        image_size: int = IMG_SIZE,
        seed: int = SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filenames = list(filenames)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.filenames))
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, index: int):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_names = [self.filenames[i] for i in batch_indices]
        xs, ys = [], []
        for name in batch_names:
            x, y = read_pair(name, self.image_dir, self.mask_dir, self.image_size)
            xs.append(x)
            ys.append(y)
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def dice_loss(y_true, y_pred):
    """Global batch Dice loss, matching the original Session 7 formulation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = tf.constant(1e-6, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )
    return 1.0 - dice


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def dice_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= PREDICTION_THRESHOLD, tf.float32)
    smooth = tf.constant(1e-6, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= PREDICTION_THRESHOLD, tf.float32)
    smooth = tf.constant(1e-6, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def build_unet(input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 1)) -> tf.keras.Model:
    """Reproduces the supplied Session 6 U-Net architecture (three downsampling levels)."""
    inputs = layers.Input(shape=input_shape, name="xray")

    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)

    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)

    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)

    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv4)

    up3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv4)
    concat3 = layers.Concatenate()([up3, conv3])
    dec3 = layers.Conv2D(256, 3, activation="relu", padding="same")(concat3)
    dec3 = layers.Conv2D(256, 3, activation="relu", padding="same")(dec3)

    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(dec3)
    concat2 = layers.Concatenate()([up2, conv2])
    dec2 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat2)
    dec2 = layers.Conv2D(128, 3, activation="relu", padding="same")(dec2)

    up1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(dec2)
    concat1 = layers.Concatenate()([up1, conv1])
    dec1 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat1)
    dec1 = layers.Conv2D(64, 3, activation="relu", padding="same")(dec1)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="lung_mask")(dec1)
    return models.Model(inputs, outputs, name="lung_unet")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=[dice_metric, iou_metric],
    )


def save_evaluation_visualization(
    filename: str, image: np.ndarray, mask: np.ndarray, pred: np.ndarray, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[..., 0], cmap="gray")
    axes[0].set_title("X-ray")
    axes[1].imshow(mask[..., 0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference mask")
    axes[2].imshow(pred[..., 0], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (threshold={PREDICTION_THRESHOLD})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(filename, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def binary_dice_iou(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    true = y_true.astype(bool)
    pred = y_pred.astype(bool)
    intersection = np.logical_and(true, pred).sum(dtype=np.float64)
    true_sum = true.sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    union = np.logical_or(true, pred).sum(dtype=np.float64)
    eps = 1e-6
    dice = (2.0 * intersection + eps) / (true_sum + pred_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def train_model(
    model: tf.keras.Model,
    split: Dict[str, List[str]],
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    initial_weights: Optional[Path] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = output_dir / "best_unet.weights.h5"
    last_weights = output_dir / "last_unet.weights.h5"
    log_path = output_dir / "training_log.csv"

    if initial_weights:
        if not initial_weights.is_file():
            raise FileNotFoundError(f"Initial weights not found: {initial_weights}")
        model.load_weights(str(initial_weights))
        print(f"Loaded initial weights only: {initial_weights}")
        print("Optimizer state is NOT restored when loading weights only.")

    train_seq = LungDataSequence(
        split["train"], image_dir, mask_dir, batch_size=batch_size, shuffle=True
    )
    val_seq = LungDataSequence(
        split["validation"], image_dir, mask_dir, batch_size=batch_size, shuffle=False
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_weights),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(last_weights),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        CSVLogger(str(log_path), append=False),
    ]
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"Best checkpoint: {best_weights}")
    print(f"Training log: {log_path}")
    return best_weights


def evaluate_model(
    model: tf.keras.Model,
    test_files: Sequence[str],
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    num_visualizations: int = 6,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if test_files:
        indices = np.linspace(
            0, len(test_files) - 1, num=min(num_visualizations, len(test_files)), dtype=int
        )
        visual_names = {test_files[i] for i in indices}
    else:
        visual_names = set()

    for i, name in enumerate(test_files, start=1):
        image, mask = read_pair(name, image_dir, mask_dir)
        probabilities = model.predict(image[np.newaxis, ...], verbose=0)[0]
        prediction = (probabilities >= PREDICTION_THRESHOLD).astype(np.float32)
        dice, iou = binary_dice_iou(mask, prediction)
        rows.append({
            "filename": name,
            "dice": dice,
            "iou": iou,
            "threshold": PREDICTION_THRESHOLD,
            "reference_foreground_pixels": int(mask.sum()),
            "predicted_foreground_pixels": int(prediction.sum()),
        })
        if name in visual_names:
            safe_stem = Path(name).stem
            save_evaluation_visualization(
                name, image, mask, prediction, vis_dir / f"{safe_stem}_comparison.png"
            )
        if i % 25 == 0 or i == len(test_files):
            print(f"Evaluation: {i}/{len(test_files)}")

    csv_path = output_dir / "test_metrics_per_image.csv"
    fieldnames = [
        "filename", "dice", "iou", "threshold",
        "reference_foreground_pixels", "predicted_foreground_pixels",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dice_values = np.asarray([r["dice"] for r in rows], dtype=np.float64)
    iou_values = np.asarray([r["iou"] for r in rows], dtype=np.float64)
    summary = {
        "test_images": len(rows),
        "threshold": PREDICTION_THRESHOLD,
        "mean_dice": float(dice_values.mean()) if len(rows) else None,
        "median_dice": float(np.median(dice_values)) if len(rows) else None,
        "std_dice": float(dice_values.std(ddof=0)) if len(rows) else None,
        "mean_iou": float(iou_values.mean()) if len(rows) else None,
        "median_iou": float(np.median(iou_values)) if len(rows) else None,
        "std_iou": float(iou_values.std(ddof=0)) if len(iou_values) else None,
        "empty_reference_masks": int(sum(r["reference_foreground_pixels"] == 0 for r in rows)),
        "preliminary_internal_result": True,
        "caveat": (
            "Known cross-split duplicate-image and patient/study-identifier overlap in the "
            "current split may make these estimates optimistic. Not clinical validation."
        ),
    }
    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== TEST SUMMARY (PRELIMINARY / INTERNAL) =====")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="U-Net lung segmentation end-to-end pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Folder containing Darwin/img and Darwin/mask")
    parser.add_argument("--mode", choices=["analyze", "train", "evaluate", "all"],
                        default="all")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Existing data_split.json. Defaults to Results/data_split.json under dataset root.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Defaults to Results/pipeline_run.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional initial weights for training or weights to evaluate.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-visualizations", type=int, default=6)
    args = parser.parse_args()

    set_seeds(args.seed)
    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = dataset_root / "Darwin" / "img"
    mask_dir = dataset_root / "Darwin" / "mask"
    split_file = args.split_file or (dataset_root / "Results" / "data_split.json")
    output_dir = args.output_dir or (dataset_root / "Results" / "pipeline_run")
    output_dir = output_dir.expanduser().resolve()

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive integers")

    if args.mode in {"analyze", "all", "train", "evaluate"}:
        paired, _, _ = paired_filenames(image_dir, mask_dir)
        if args.mode in {"analyze", "all"}:
            analyze_dataset(image_dir, mask_dir, output_dir)
        if args.mode == "analyze":
            return
        split = load_or_create_split(paired, split_file, seed=args.seed)

    model = build_unet()
    print(f"Model parameters: {model.count_params():,}")

    if args.mode in {"train", "all"}:
        compile_model(model, learning_rate=args.learning_rate)
        best_weights = train_model(
            model, split, image_dir, mask_dir, output_dir / "training",
            batch_size=args.batch_size, epochs=args.epochs,
            initial_weights=args.checkpoint,
        )
        model.load_weights(str(best_weights))
        evaluate_model(
            model, split["test"], image_dir, mask_dir, output_dir / "test",
            num_visualizations=args.num_visualizations,
        )
    elif args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("--mode evaluate requires --checkpoint")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model.load_weights(str(args.checkpoint))
        evaluate_model(
            model, split["test"], image_dir, mask_dir, output_dir / "test",
            num_visualizations=args.num_visualizations,
        )


if __name__ == "__main__":
    main()

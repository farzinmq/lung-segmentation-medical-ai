import os
import json
import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

ROOT = "/Users/farzin/Downloads/Medical AI"
IMG = os.path.join(ROOT, "Darwin", "img")
MASK = os.path.join(ROOT, "Darwin", "mask")
SPLIT = os.path.join(ROOT, "Results", "data_split.json")
WEIGHTS = os.path.join(
    ROOT, "Results", "session9_resume", "best_resume.weights.h5"
)
OUT = os.path.join(ROOT, "Results", "session9_test")

os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "visualizations"), exist_ok=True)

# Load the existing fixed split; do not regenerate it.
with open(SPLIT, "r") as f:
    split = json.load(f)

test_files = split["test"]
assert len(test_files) == 603, f"Expected 603 test images, got {len(test_files)}"

if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")

# Verify test images and masks exist before evaluation.
for name in test_files:
    for folder in (IMG, MASK):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")

# Rebuild the same U-Net architecture used during training.
inputs = layers.Input(shape=(512, 512, 1))

c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(c1)
p1 = layers.MaxPooling2D(2)(c1)

c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(c2)
p2 = layers.MaxPooling2D(2)(c2)

c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(c3)
p3 = layers.MaxPooling2D(2)(c3)

c4 = layers.Conv2D(512, 3, activation="relu", padding="same")(p3)
c4 = layers.Conv2D(512, 3, activation="relu", padding="same")(c4)

u3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c4)
d3 = layers.Concatenate()([u3, c3])
d3 = layers.Conv2D(256, 3, activation="relu", padding="same")(d3)
d3 = layers.Conv2D(256, 3, activation="relu", padding="same")(d3)

u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d3)
d2 = layers.Concatenate()([u2, c2])
d2 = layers.Conv2D(128, 3, activation="relu", padding="same")(d2)
d2 = layers.Conv2D(128, 3, activation="relu", padding="same")(d2)

u1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
d1 = layers.Concatenate()([u1, c1])
d1 = layers.Conv2D(64, 3, activation="relu", padding="same")(d1)
d1 = layers.Conv2D(64, 3, activation="relu", padding="same")(d1)

outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
model = models.Model(inputs, outputs)

model.load_weights(WEIGHTS)
print("Model weights loaded successfully.")
print("Parameters:", model.count_params())
print("Test images:", len(test_files))

def load_pair(name):
    x = cv2.imread(os.path.join(IMG, name), cv2.IMREAD_GRAYSCALE)
    y = cv2.imread(os.path.join(MASK, name), cv2.IMREAD_GRAYSCALE)

    if x is None or y is None:
        raise ValueError(f"Could not read image or mask: {name}")

    x = cv2.resize(x, (512, 512)).astype(np.float32) / 255.0
    y = cv2.resize(
        y, (512, 512), interpolation=cv2.INTER_NEAREST
    )
    y = (y > 0).astype(np.uint8)
    return x, y

def calculate_metrics(y_true, y_pred):
    true_sum = int(y_true.sum())
    pred_sum = int(y_pred.sum())
    intersection = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    union = int(np.logical_or(y_true == 1, y_pred == 1).sum())

    # If both masks are empty, overlap is defined as 1.
    if true_sum == 0 and pred_sum == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * intersection) / (true_sum + pred_sum) \
            if true_sum + pred_sum else 0.0
        iou = intersection / union if union else 0.0

    return dice, iou, true_sum, pred_sum

rows = []
empty_true_masks = 0

for index, name in enumerate(test_files, start=1):
    x, y_true = load_pair(name)

    # Predict one image at a time to limit memory usage.
    probability = model.predict(
        x[None, ..., None], verbose=0
    )[0, ..., 0]

    # Convert probabilities into a binary mask at threshold 0.5.
    y_pred = (probability >= 0.5).astype(np.uint8)

    dice, iou, true_pixels, pred_pixels = calculate_metrics(
        y_true, y_pred
    )

    if true_pixels == 0:
        empty_true_masks += 1

    rows.append({
        "filename": name,
        "dice": dice,
        "iou": iou,
        "true_mask_pixels": true_pixels,
        "predicted_mask_pixels": pred_pixels
    })

    # Save six representative examples, spread across the test list.
    selected = {
        1,
        max(1, len(test_files) // 5),
        max(1, 2 * len(test_files) // 5),
        max(1, 3 * len(test_files) // 5),
        max(1, 4 * len(test_files) // 5),
        len(test_files)
    }

    if index in selected:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(x, cmap="gray")
        axes[0].set_title("X-ray")
        axes[1].imshow(y_true, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground-truth mask")
        axes[2].imshow(y_pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Prediction\nDice={dice:.4f}, IoU={iou:.4f}")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(name)
        fig.tight_layout()
        safe_name = os.path.splitext(name)[0]
        fig.savefig(
            os.path.join(OUT, "visualizations", safe_name + ".png"),
            dpi=120,
            bbox_inches="tight"
        )
        plt.close(fig)

    if index % 25 == 0 or index == len(test_files):
        print(f"Evaluated {index}/{len(test_files)}")

# Save per-image metrics.
csv_path = os.path.join(OUT, "test_metrics_per_image.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

dice_values = np.array([r["dice"] for r in rows], dtype=np.float64)
iou_values = np.array([r["iou"] for r in rows], dtype=np.float64)

summary = {
    "test_images": len(rows),
    "threshold": 0.5,
    "mean_dice": float(np.mean(dice_values)),
    "median_dice": float(np.median(dice_values)),
    "std_dice": float(np.std(dice_values)),
    "mean_iou": float(np.mean(iou_values)),
    "median_iou": float(np.median(iou_values)),
    "std_iou": float(np.std(iou_values)),
    "empty_ground_truth_masks": empty_true_masks,
    "checkpoint": WEIGHTS
}

summary_path = os.path.join(OUT, "test_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n===== FINAL TEST RESULTS =====")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved CSV:", csv_path)
print("Saved summary:", summary_path)
print("Saved visualizations:", os.path.join(OUT, "visualizations"))

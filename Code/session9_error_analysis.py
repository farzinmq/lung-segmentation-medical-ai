import os
import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

ROOT = "/Users/farzin/Downloads/Medical AI"
IMG = os.path.join(ROOT, "Darwin", "img")
MASK = os.path.join(ROOT, "Darwin", "mask")
CSV_PATH = os.path.join(
    ROOT, "Results", "session9_test", "test_metrics_per_image.csv"
)
WEIGHTS = os.path.join(
    ROOT, "Results", "session9_resume", "best_resume.weights.h5"
)
OUT = os.path.join(ROOT, "Results", "session9_test", "error_analysis")
os.makedirs(OUT, exist_ok=True)

# Select the 10 lowest-Dice test cases from the completed evaluation.
with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: float(r["dice"]))
worst = rows[:10]

# Rebuild the exact U-Net architecture used for training.
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
print("Model weights loaded. Analyzing", len(worst), "cases.")

summary_path = os.path.join(OUT, "worst10_summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename", "dice", "iou", "false_positive_pixels",
                       "false_negative_pixels"]
    )
    writer.writeheader()

    for i, row in enumerate(worst, 1):
        name = row["filename"]
        x = cv2.imread(os.path.join(IMG, name), cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(os.path.join(MASK, name), cv2.IMREAD_GRAYSCALE)
        if x is None or y is None:
            raise ValueError("Could not read image or mask: " + name)

        x = cv2.resize(x, (512, 512)).astype(np.float32) / 255.0
        y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_NEAREST)
        y_true = (y > 0)
        probability = model.predict(x[None, ..., None], verbose=0)[0, ..., 0]
        y_pred = probability >= 0.5

        false_positive = y_pred & ~y_true
        false_negative = y_true & ~y_pred

        # Error map: 0=background, 1=true positive, 2=false positive,
        # 3=false negative. Displayed with a legend in the title.
        error_map = np.zeros((512, 512), dtype=np.uint8)
        error_map[y_true & y_pred] = 1
        error_map[false_positive] = 2
        error_map[false_negative] = 3

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        axes[0].imshow(x, cmap="gray")
        axes[0].set_title("X-ray")
        axes[1].imshow(y_true, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")
        axes[2].imshow(y_pred, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[3].imshow(error_map, vmin=0, vmax=3)
        axes[3].set_title("Error map\n1=TP, 2=FP, 3=FN")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{name}\nDice={float(row['dice']):.4f}, "
            f"IoU={float(row['iou']):.4f}"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUT, f"{i:02d}_{os.path.splitext(name)[0]}.png"),
            dpi=130, bbox_inches="tight"
        )
        plt.close(fig)

        writer.writerow({
            "filename": name,
            "dice": row["dice"],
            "iou": row["iou"],
            "false_positive_pixels": int(false_positive.sum()),
            "false_negative_pixels": int(false_negative.sum())
        })
        print(f"{i}/10: {name} | Dice={float(row['dice']):.4f}")

print("Done. Images and summary saved to:", OUT)
print("Summary CSV:", summary_path)

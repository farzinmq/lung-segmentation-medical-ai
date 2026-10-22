# Evaluation results

`test_summary.json` and `test_metrics_per_image.csv` contain the preliminary/internal evaluation on the fixed 603-image test partition at threshold 0.5. The per-image CSV includes source filenames.

- `visualizations/` contains six representative panels with X-ray, reference mask, and prediction.
- `error_analysis/` contains the ten lowest-Dice cases with ground truth, prediction, and pixel-level error maps.

These are qualitative illustrations of model behavior, including failures; they are not clinical validation. The images are included under the user's authorization for this private prototype repository. Verify dataset redistribution terms before making the repository public. Raw dataset folders are not included.

The Darwin subset contains 6,106 image-mask pairs. Inspection identified 81 empty masks using the current rule (no pixels above 127); all were excluded from the fixed split of 6,025 pairs (4,820 train, 602 validation, 603 test). The reasons for empty masks remain under investigation.

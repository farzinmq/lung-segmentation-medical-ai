# Chest X-ray Lung Segmentation with U-Net

> **Research prototype.** This repository documents an educational/research implementation for segmenting lung fields in chest radiographs. It is not a diagnostic tool and has not been clinically validated.

## Project status

A 2D U-Net was trained on paired chest radiographs and lung masks. The current internal evaluation used a fixed held-out split of 603 images and a probability threshold of 0.5. These are preliminary results from the project's current dataset and pipeline; they do not establish generalization to other institutions or patient populations.

| Test metric | Mean | Median | SD | N |
|---|---:|---:|---:|---:|
| Dice | 0.9470 (94.70%) | 0.9562 | 0.0318 | 603 |
| IoU | 0.9010 (90.10%) | 0.9161 | 0.0543 | 603 |

Metrics were computed per image after thresholding predictions at 0.5. The values match `Results/test_summary.json`. Per-image metrics are in `Results/test_metrics_per_image.csv`. Examples of failure modes reported during analysis include false-negative regions and false positives around devices, opacities, and image borders.

## Dataset quality and empty masks

The Darwin subset contains 6,106 image-mask pairs. Inspection identified 81 empty masks using the current foreground rule (no mask pixels above 127); these samples are absent from the recorded split. The fixed split contains 6,025 unique pairs: 4,820 training, 602 validation, and 603 test. A programmatic check confirmed that all split filenames have paired files and none of the 81 empty-mask samples appears in any partition. The causes of the empty masks have not been fully investigated; their exclusion criterion and implications remain part of the data-quality audit.

## Method

- **Task:** binary lung-field segmentation (both lungs as foreground).
- **Input:** grayscale chest X-rays resized to 512 × 512 and scaled to [0, 1].
- **Target:** masks resized with nearest-neighbor interpolation and binarized. Confirm the threshold used by the specific training/evaluation scripts before reproducing a run; source scripts have used different mask thresholds.
- **Model:** encoder-decoder U-Net with convolutional blocks, max-pooling, transposed-convolution upsampling and skip connections; sigmoid output.
- **Objective:** binary cross-entropy plus soft Dice loss.
- **Optimizer:** Adam, initial learning rate 1e-4. Training scripts use batch size 2 and validation-based checkpointing/early stopping in the documented runs.
- **Split:** fixed image-level lists in `Results/data_split.json`: train 4,820, validation 602, test 603 (approximately 80/10/10).

The recommended entry point is `Code/lung_segmentation_pipeline.py`; the session-specific scripts are retained for the recorded training and evaluation workflow. The evaluation scripts currently contain machine-specific paths and need configuration before reuse.

## Repository structure

- `Code/`: consolidated pipeline plus Session 9 evaluation and error-analysis scripts.
- `Weights/`: checkpoint files are excluded from the repository. See the model-weights note below.
- `Results/`: aggregate and per-image metrics, six representative prediction comparisons, and ten worst-case error-analysis figures.
- `requirements.txt`: Python dependencies. Exact environment versions and a dependency lock file are not yet maintained.

Do not add raw images or masks. Per-image metric rows and filenames may reveal source identifiers; confirm dataset terms and privacy before making the repository public.

## Model weights

The trained checkpoint is not included in this repository. To request access to the current weights-only checkpoint, email [farzinmoqadasi@gmail.com](mailto:farzinmoqadasi@gmail.com). Requests are subject to dataset terms and project availability. The checkpoint does not include optimizer state.

## Reproducibility

1. Obtain the source dataset and place paired images and masks under `Darwin/img/` and `Darwin/mask/`. Dataset terms and redistribution rights must be checked independently; dataset files are not included here.
2. Install a compatible Python/TensorFlow/OpenCV/NumPy environment. Exact environment versions and a dependency lock file are not yet maintained.
3. Review and configure data/output paths in the scripts. Some source scripts contain machine-specific absolute paths.
4. Use the intended split manifest consistently and verify that every listed image and mask exists.
5. Train using the appropriate Colab or local script, then evaluate the selected checkpoint using matching preprocessing and threshold settings. The reported summary is a record of the current run, not an independently reproduced result.

## Limitations

- This is an internal, single-project evaluation; no external or independent dataset test has been documented.
- The current split has not been audited for patient/study grouping or duplicate images across partitions; possible leakage remains an unresolved limitation.
- The source dataset composition, patient-level grouping, label provenance, and licensing must be explicitly documented and checked. The current image-level split is not a leakage-safe patient/study-level evaluation.
- The displayed aggregate metrics can obscure clinically important failure modes. Review the per-image distribution and worst-case examples.
- No confidence intervals, subgroup analysis, annotation agreement study, calibration analysis, or prospective/clinical evaluation has been completed.
- Segmentation quality does not imply disease detection or clinical utility. Do not use for patient care.

## Research and paper-readiness plan

1. **Audit data and labels:** identify dataset sources, licenses, duplicates, patient/study identifiers, acquisition characteristics, mask-generation process, and split leakage; create a reproducible patient-level split if identifiers permit.
2. **Freeze protocol:** document exact architecture, preprocessing, software/hardware versions, seeds, checkpoint-selection rule, threshold choice, and all exclusions. Keep the test set untouched during development.
3. **Strengthen evaluation:** compute bootstrap confidence intervals for per-image Dice/IoU; report distributions and failure rates; perform subgroup/error analysis where metadata and permissions permit.
4. **Independent testing:** evaluate a frozen model on a genuinely independent, appropriately licensed dataset and report domain shift and limitations.
5. **Baselines and robustness:** compare against established segmentation baselines under the same split; assess sensitivity to threshold, resolution, and preprocessing; consider multiple seeds.
6. **Literature and reporting:** conduct a documented literature review, define a specific research question and contribution, and prepare methods/results using relevant medical-imaging reporting guidance. Publication suitability depends on the resulting evidence and novelty assessment.

## Responsible use

For research and education only. Not for diagnosis, treatment, triage, or other clinical decisions. No claim of novelty, clinical validation, publication, or external testing is made.

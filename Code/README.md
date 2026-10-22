# Code layout

`lung_segmentation_pipeline.py` is the recommended single entry point for dataset inspection, training, and evaluation. The `session9_*` scripts are retained as historical/session-specific implementations and are not the recommended route for a fresh run.

## Important review notes

- The consolidated pipeline creates a random file-level split if no split manifest exists. This is not patient/study grouped and must not be interpreted as leakage-safe.
- Preserve the supplied split for reproducing the recorded internal result. Do not retrain/re-split as part of Phase 1.
- Script outputs include filenames. Check data-use/privacy terms before sharing per-image CSVs or split manifests.
- The script reports results as preliminary/internal and carries a split-overlap caveat.

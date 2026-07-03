# Fusion Runtime Migration Contract

This document fixes the migration boundary for the AGD20K geometry-interaction
fusion heavy runtime. It does not move FLUX generation, DINO token extraction,
PCA, ROI, geometry mask generation, or metric aggregation code yet.

## Lightweight API

`pba.fusion.runtime` is safe to import without PyTorch, diffusers, OpenCV, PIL,
NumPy, DINO, DINOv3, or FLUX model assets. It records the accepted runtime
surface:

- `FUSION_RUNTIME_MODULES`
- `FUSION_SUMMARY_COLUMNS`
- `FUSION_BASE_METRIC_KEYS`
- `FUSION_GLOBAL_METRIC_KEYS`
- `FUSION_SAMPLE_ARTIFACTS`
- `FusionRuntimeBoundary`
- `build_runtime_boundary()`
- `validate_fusion_detail(detail)`
- `validate_global_metrics(metrics)`

## Current Boundary

The legacy entrypoint remains:

```text
fusion_zero_shot/src/agd20k_eval/run_flux_kontext_eval.py
```

The default config remains:

```text
fusion_zero_shot/src/agd20k_eval/config.yaml
```

The migration unit includes:

- `run_flux_kontext_eval.process_sample`
- `run_flux_kontext_eval.run_geom_pipeline`
- `run_kontext_generation`
- `warp_heatmap_cli`
- AGD20K sample iteration
- CSV and JSON logging helpers
- ROI mask construction and token selection
- `extract_dino_tokens`
- `run_pca`
- `generate_geometry_mask`
- final mask postprocessing and overlays
- per-sample and global metric aggregation

## Cache Dependency

The current public config can run with `geom_pipeline.dino_cache_only: true`.
That means fusion evaluation depends on existing DINO token cache files. The
cache format is documented in `docs/fusion_cache_contract.md`.

Do not claim one-click fusion reproduction until `fusion-cache-build` creates
the required cache files before `fusion-eval`.

## Summary CSV Contract

`FUSION_SUMMARY_COLUMNS` preserves the per-sample `summary.csv` shape. It must
include base interaction metrics:

- `mKLD`
- `mSIM`
- `mNSS`

It also preserves geometry and final-fusion fields such as:

- `object_heatmap`
- `object_warped_heatmap`
- `geom_mask`
- `geom_selected_pc`
- `geom_mKLD`, `geom_mSIM`, `geom_mNSS`
- `soft_heatmap`
- `soft_mKLD`, `soft_mSIM`, `soft_mNSS`
- `final_mask`
- `final_threshold`
- `final_mKLD`, `final_mSIM`, `final_mNSS`
- `adaptive_fusion_enabled`
- `adaptive_lambda`
- `interaction_confidence`, `geometry_confidence`, `alignment_confidence`
- `base_lambda`, `lambda_range`
- `adaptive_fusion_detail`
- `final_masks_detail`

`validate_fusion_detail` checks the required per-sample row shape without
importing heavy runtime dependencies.

## Global Metrics Contract

Fusion evaluation writes `global_metrics.json` with:

- `total_samples`
- `processed_samples`
- `resume_reused`
- `skipped_samples`
- `failed_samples`
- `overall_metrics`
- `per_affordance_metrics`
- `skipped_detail`
- `failed_detail`

When geometry fusion is enabled, the runtime may also include
`geom_pipeline_config`.

## Sample Artifact Contract

`FUSION_SAMPLE_ARTIFACTS` names the expected per-sample directories/files:

- `kontext/`
- `mapped/`
- `metrics.json`
- `geom_pipeline/stage_roi/`
- `geom_pipeline/stage_dino/`
- `geom_pipeline/stage_geom/`
- `geom_pipeline/stage_final/`

The migrated runtime must preserve these artifact locations or provide legacy
wrappers that keep existing analysis scripts compatible.

## Migration Rule

Move the fusion heavy runtime only after cache generation and true development
fusion code are available. Keep the legacy AGD20K eval script as a wrapper until
existing configs, summary CSVs, and downstream analysis remain compatible.

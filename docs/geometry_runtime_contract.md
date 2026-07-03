# Geometry Runtime Migration Contract

This document fixes the migration boundary for the geometry probing heavy
runtime. It does not move trainer, evaluator, backbone, transform, dataset, or
logging code yet.

## Lightweight API

`pba.geometry.runtime` is safe to import without PyTorch, torchvision, OpenCV,
SciPy, PIL, diffusers, or backbone source trees. It records the accepted runtime
surface:

- `GEOMETRY_RUNTIME_MODULES`: legacy classes/functions that must move together
- `GEOMETRY_BACKBONE_TARGETS`: supported config `model.target` values
- `GEOMETRY_TRAINING_OUTPUT_FILES`: files the migrated trainer must preserve
- `GeometryRuntimeBoundary`
- `build_runtime_boundary()`
- `validate_geometry_metrics(metrics)`

## Current Boundary

The package entrypoints are already public:

- `pba.geometry.linear_probe.run_train`
- `pba.geometry.linear_probe.run_eval`

The heavy implementation still lives under:

```text
geometry_probing/umd_linear_probing/src
```

The migration unit includes:

- `LinearProbeExperiment`
- `evaluate_linear_probe`
- `UMDAffordanceDataset`
- collate and image transform functions
- backbone builders for DINO, DINOv2, DINOv3, OpenCLIP, Flux, Stable Diffusion,
  SigLIP2, and SAM
- linear and multi-layer probe heads
- logging, random seed, metric, and visualization helpers

## Backbone Targets

`GEOMETRY_BACKBONE_TARGETS` must stay aligned with model targets accepted by the
legacy trainer:

- `dino`
- `dinov2`
- `dinov3`
- `open_clip`
- `flux`
- `stable_diffusion`
- `siglip2`
- `sam`

Do not rename these targets during migration unless configs and reproduction
commands are migrated in the same change.

## Dataset Dependency

The heavy runtime depends on the UMD dataset contract in
`docs/umd_dataset_contract.md`. The migrated trainer must preserve split-record
schema, sample keys, optional `geom_depth` / `geom_normal` attachment, and
collate behavior.

## Metrics Contract

Geometry eval returns a mapping with:

- `loss`
- `miou`
- `per_class_iou`

`validate_geometry_metrics` checks this public shape without importing torch.

## Output Contract

The migrated trainer must preserve these output files:

- `master.log`
- `summary.json`
- `training_history.json`
- `val_examples.pt`
- `test_examples.pt`
- `{sweep}/train.log`
- `{sweep}/config_snapshot.yaml`
- `{sweep}/linear_probe.pth`

The eval entrypoint additionally writes checkpoint-adjacent JSON metrics and
optional example tensors through `pba.geometry.linear_probe.run_eval`.

## Migration Rule

Move the geometry heavy runtime only when the true development code is ready.
The move should keep legacy wrappers under
`geometry_probing/umd_linear_probing/src` until existing scripts and configs are
verified against the migrated package modules.

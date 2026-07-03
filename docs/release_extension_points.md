# Release Extension Points

This document describes missing release pieces that should be filled from the
true development environment. Do not implement these by guessing from the public
GitHub source.

Run the audit with gap details:

```bash
bash scripts/audit_release_readiness.sh --skip-asset-check --show-gaps
```

## `interaction-batch-eval`

Current status:

- `run.py interaction-probe` supports a single image, prompt, and affordance.
- `configs/reproduce/main.yaml` keeps `jobs.interaction` empty.
- The expected future batch interface is documented in
  `docs/interaction_batch_contract.md`.

Expected extension:

- Add a batch AGD20K interaction-only runner that uses existing Flux Kontext
  attention extraction logic.
- Inputs should include AGD20K root, Flux model path, prompt templates, device,
  output root, optional affordance subset, and optional sample cap.
- Outputs should include per-sample verb attention maps, selected token
  metadata, and aggregate KLD, SIM, and NSS CSVs.
- Output paths and metric columns should follow
  `docs/interaction_batch_contract.md`.

Integration point:

- Add a new `run.py` command only after the real batch runner is available.
- Add that command under `configs/reproduce/main.yaml` `jobs.interaction`.

## `fusion-cache-build`

Current status:

- `fusion_zero_shot/src/agd20k_eval/config.yaml` sets
  `geom_pipeline.dino_cache_only: true`.
- Fusion evaluation expects DINO token cache under `outputs/fusion_cache`.
- The required cache file contract is documented in
  `docs/fusion_cache_contract.md`.

Expected extension:

- Add a cache build command that creates the exact artifacts consumed by
  `fusion_zero_shot`.
- Inputs should include AGD20K root, DINO backend, target size, patch size,
  checkpoint/source path, cache root, and device.
- Outputs should include deterministic token cache files and metadata.
- Cache files must follow `docs/fusion_cache_contract.md`.

Integration point:

- Add the cache build command before `fusion-eval` in
  `configs/reproduce/main.yaml`.
- Keep `dino_cache_only: true` for evaluation once cache generation is explicit.

## `dinov3-external-source`

Current status:

- Public release direction requires a user-provided `models/dinov3/` source
  tree.
- Current GitHub source still contains bundled
  `fusion_zero_shot/src/dino/third_party/dinov3`.

Expected extension:

- Confirm all imports that depend on the bundled tree.
- Update path resolution to prefer `models/dinov3/`.
- Remove bundled source only after the replacement path is verified.

Integration point:

- `models/dinov3/`
- DINOv3 geometry config and fusion DINO backend setup.

## `sd21-local-model`

Current status:

- `geometry_probing/umd_linear_probing/configs/sd21.yaml` uses Hugging Face
  model id `stabilityai/stable-diffusion-2-1`.
- The accepted local/offline path convention is documented in
  `docs/sd21_local_model_contract.md`.

Expected extension:

- Define whether paper reproduction accepts online Hugging Face loading or
  requires a local model directory.
- If local mode is required, add a config key and loader path convention backed
  by real development code.
- Local model validation should follow `docs/sd21_local_model_contract.md`.

Integration point:

- `geometry_probing/umd_linear_probing/configs/sd21.yaml`
- Stable Diffusion backbone loader.

## `umd-pytorch-dataset-migration`

Current status:

- UMD split/path utilities now live in `pba.data.geometry`.
- UMD dataset schema helpers now live in `pba.data.umd` and are documented in
  `docs/umd_dataset_contract.md`.
- The full PyTorch `UMDAffordanceDataset`, image/mask transforms, geometry
  feature attachment, and collate behavior remain under
  `geometry_probing/umd_linear_probing/src/data`.

Expected extension:

- Move the heavy dataset code only when geometry probing train/eval code is
  migrated as one coherent unit.
- Keep optional dependencies explicit: PyTorch, OpenCV, SciPy, PIL, and NumPy.
- Preserve existing sample keys: `image`, `pixel_mask`, `patch_mask`, `meta`,
  and optional `geom_depth` / `geom_normal`.
- Preserve split-record and optional geometry manifest schema from
  `docs/umd_dataset_contract.md`.

Integration point:

- `pba.data.geometry_dataset` or a similarly scoped module.
- Legacy `geometry_probing/umd_linear_probing/src/data/dataset.py` wrapper.
- Geometry train/eval loader construction.

## `geometry-heavy-runtime-migration`

Current status:

- Geometry config loading now lives in `pba.geometry.config`.
- Geometry train/eval argument parsing and dispatch now live in
  `pba.geometry.linear_probe`.
- Geometry runtime boundary helpers now live in `pba.geometry.runtime` and are
  documented in `docs/geometry_runtime_contract.md`.
- The actual training runtime remains under
  `geometry_probing/umd_linear_probing/src`: trainer, evaluator, backbone
  builders, transforms, logging, and visualization.

Expected extension:

- Move heavy runtime modules only after the real development code is ready.
- Preserve existing behavior for `LinearProbeExperiment`, `evaluate_linear_probe`,
  checkpoint loading, metrics logging, and qualitative example saving.
- Preserve backbone targets, metric keys, output files, and dataset dependency
  from `docs/geometry_runtime_contract.md`.
- Keep optional runtime dependencies explicit: PyTorch, torchvision, OpenCV,
  SciPy, PIL, and backbone-specific packages.

Integration point:

- `pba.geometry.training` or similarly scoped modules.
- Legacy wrappers under `geometry_probing/umd_linear_probing/src`.
- Existing `pba.geometry.linear_probe` dependency-loading helpers.

## `fusion-heavy-runtime-migration`

Current status:

- Fusion config, Kontext resolution, prompt/token, and heatmap path helpers now
  live in `pba.fusion`.
- Fusion runtime boundary helpers now live in `pba.fusion.runtime` and are
  documented in `docs/fusion_runtime_contract.md`.
- The actual AGD20K fusion runtime remains in
  `fusion_zero_shot/src/agd20k_eval/run_flux_kontext_eval.py` and related
  modules: FLUX generation, heatmap warping, DINO token extraction, PCA, ROI
  construction, geometry masks, and metric aggregation.

Expected extension:

- Move heavy runtime modules only after cache generation and true development
  fusion code are available.
- Preserve existing behavior for FLUX token selection, object heatmap warping,
  DINO cache use, geometry mask generation, and summary CSV writing.
- Preserve summary columns, global metrics shape, sample artifacts, and cache
  dependency from `docs/fusion_runtime_contract.md`.
- Keep optional runtime dependencies explicit: PyTorch, diffusers/Flux runtime,
  OpenCV, PIL, NumPy, and DINO/DINOv3 source/checkpoints.

Integration point:

- `pba.fusion.runtime` or similarly scoped modules.
- Existing AGD20K eval script under `fusion_zero_shot/src/agd20k_eval`.
- Future `fusion-cache-build` command.

## `interaction-heavy-runtime-migration`

Current status:

- Interaction parser, token matching, attention accumulation, output directory,
  and metadata helpers now live in `pba.interaction`.
- Interaction runtime boundary helpers now live in `pba.interaction.runtime`
  and are documented in `docs/interaction_runtime_contract.md`.
- The actual Flux runtime remains in
  `interaction_probing/cross_attention_probe/cross_attention_probe.py`: Flux
  pipeline loading, attention processor injection, denoising, and generated
  image saving.

Expected extension:

- Move heavy Flux runtime only after the real interaction probing code is ready.
- Preserve existing behavior for tokenizer matching, attention processor
  placement, image-query/text-key probability capture, generated image saving,
  and heatmap/overlay export.
- Preserve runtime stages, output files, metadata shape, and batch dependency
  from `docs/interaction_runtime_contract.md`.
- Keep optional runtime dependencies explicit: PyTorch, diffusers, Flux Kontext
  model weights, PIL, NumPy, and matplotlib.

Integration point:

- `pba.interaction.runtime` or similarly scoped modules.
- Existing single-image probe script under `interaction_probing`.
- Future `interaction-batch-eval` command.

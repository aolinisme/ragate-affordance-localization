# Fusion Cache Contract

This document defines the DINO token cache consumed by AGD20K fusion evaluation.
It is a contract, not a cache builder implementation.

## Cache Filename

Cache files live under `geom_pipeline.cache_root` and use:

```text
{stem}_{width}x{height}_p{patch}.npz
```

Example:

```text
outputs/fusion_cache/knife_1280x960_p16.npz
```

Where:

- `stem`: image filename stem
- `width`, `height`: DINO target size from `geom_pipeline.dino_target_wh`
- `patch`: DINO patch size from `geom_pipeline.dino_patch_size`

The public helper is `pba.fusion.cache.build_cache_path`.

## Required NPZ Keys

Each cache file must contain:

- `tokens`: 2D float array, shape `[Hp * Wp, C]`
- `Hp`: token-grid height
- `Wp`: token-grid width
- `meta`: resize metadata accepted by `ResizeMeta`

Use `pba.fusion.cache.validate_cache_file` for lightweight validation.

## cache_only Behavior

`fusion_zero_shot/src/agd20k_eval/config.yaml` currently sets:

```yaml
geom_pipeline:
  dino_cache_only: true
  cache_root: ../../../outputs/fusion_cache
```

When `cache_only` is true, fusion evaluation must fail on cache miss instead of
silently loading DINO and creating cache files. This keeps public evaluation
separate from missing `fusion-cache-build` logic.

## Expected Future Builder

The `fusion-cache-build` gap should be filled from the true development code.
Expected inputs:

- AGD20K testset root
- DINO backend
- target width/height
- patch size
- DINO source/checkpoint paths
- cache root
- device

Expected output: one NPZ cache file per image, following this contract.

## Do Not Guess Builder Logic

Do not invent cache generation from the public source alone. The current release
only documents the cache format and keeps evaluation in cache-only mode until
the real builder is available.

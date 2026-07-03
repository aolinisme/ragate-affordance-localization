# UMD Dataset Migration Contract

This document fixes the public migration boundary for the UMD PyTorch dataset.
It does not move the heavy runtime yet. The current dataset implementation
stays in `geometry_probing/umd_linear_probing/src/data/dataset.py` until
geometry train/eval runtime migration is ready.

## Lightweight API

`pba.data.umd` is safe to import without PyTorch, SciPy, OpenCV, or PIL. It
documents and validates the schema used by the existing UMD dataset code:

- `UMD_SPLIT_RECORD_KEYS`: `tool`, `frame_id`, `rgb`, `depth`, `label_mat`
- `UMD_SAMPLE_OUTPUT_KEYS`: `image`, `pixel_mask`, `patch_mask`, `meta`
- `UMD_OPTIONAL_GEOMETRY_KEYS`: `geom_depth`, `geom_normal`
- `UMD_GEOMETRY_MANIFEST_KEYS`: `pred_depth_npy`, `pred_normal_npy`
- `validate_umd_split_record(record)`
- `build_umd_sample_paths(dataset_root, record)`
- `build_geometry_manifest_index(manifest)`
- `validate_umd_sample(sample, require_geometry=...)`

## Split Record Schema

Split records are produced by `pba.data.geometry.build_split_record` and use
UMD-root-relative asset paths:

```json
{
  "tool": "hammer",
  "frame_id": "hammer_01_000001",
  "rgb": "tools/hammer/hammer_01_000001_rgb.jpg",
  "depth": "tools/hammer/hammer_01_000001_depth.png",
  "label_mat": "tools/hammer/hammer_01_000001_label.mat"
}
```

`build_umd_sample_paths` resolves `rgb`, `depth`, and `label_mat` against the
dataset root while preserving absolute paths when provided.

## Runtime Sample Schema

The migrated PyTorch dataset must preserve the current output contract:

- `image`: transformed RGB tensor
- `pixel_mask`: full-resolution affordance label tensor
- `patch_mask`: patch-grid affordance label tensor
- `meta`: mapping with `tool` and `frame_id`
- `geom_depth`: optional pooled depth feature
- `geom_normal`: optional pooled normal feature

Collate behavior must keep `meta` as a list of per-sample mappings while
default-collating tensor-like fields.

## Geometry Manifest Schema

Optional geometry feature attachment accepts either a flat list of records or a
mapping with split keys such as `train`, `val`, `test`, or `data`.

Each record may include:

```json
{
  "frame_id": "hammer_01_000001",
  "pred_depth_npy": "geometry/depth/hammer_01_000001.npy",
  "pred_normal_npy": "geometry/normal/hammer_01_000001.npy"
}
```

The migrated loader should use `build_geometry_manifest_index` to align
geometry assets by `frame_id`.

## Migration Boundary

Keep the current heavy code in
`geometry_probing/umd_linear_probing/src/data/dataset.py` until the geometry
runtime migration can move dataset construction, transforms, collate behavior,
trainer/evaluator wiring, and checkpoint-compatible behavior together.

Do not invent missing dataset behavior from this contract. Use this contract to
verify that future migrated code preserves current public inputs and outputs.

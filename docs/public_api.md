# Public API

This page lists the lightweight public API exposed by `pba`. These modules are
safe to import without loading PyTorch, diffusers, Flux, DINO, or dataset assets.
For the consolidated release state, see `docs/release_status.md`.

## `pba.metrics`

- `pba.metrics.affordance`: `cal_kl`, `cal_sim`, `cal_nss`
- `pba.metrics.segmentation`: `ConfusionMatrix`, `update_confusion_matrix`,
  `compute_iou`

Legacy metric modules re-export these functions.

## `pba.data`

- `pba.data.geometry`: `CategorySplitEntry`, `parse_category_split`,
  `build_instance_index`, `build_split_record`, `train_val_test_split`,
  `save_split_mapping`
- `pba.data.umd`: `validate_umd_split_record`, `build_umd_sample_paths`,
  `build_geometry_manifest_index`, `validate_umd_sample`
- `pba.data.affordance`: `SampleEntry`, `iter_agd20k_samples`

## `pba.geometry`

- `pba.geometry.config`: `Config`, `load_config`
- `pba.geometry.linear_probe`: `parse_train_args`, `run_train`,
  `parse_eval_args`, `run_eval`
- `pba.geometry.sd21`: `resolve_sd21_model_reference`,
  `validate_sd21_local_model_dir`
- `pba.geometry.runtime`: `build_runtime_boundary`,
  `validate_geometry_metrics`

`run_train` and `run_eval` import the heavy legacy geometry runtime only when
execution needs it.

## `pba.fusion`

- `pba.fusion.config`: `load_config`
- `pba.fusion.kontext`: `PREFERRED_KONTEXT_RESOLUTIONS`,
  `pick_preferred_resolution`
- `pba.fusion.prompts`: `format_object_name`, `select_token`,
  `sanitize_token_name`, `build_object_token_candidates`
- `pba.fusion.paths`: `ensure_dir`, `get_heatmap_path`
- `pba.fusion.cache`: `cache_filename`, `build_cache_path`,
  `validate_cache_file`
- `pba.fusion.runtime`: `build_runtime_boundary`,
  `validate_fusion_detail`, `validate_global_metrics`

## `pba.interaction`

- `pba.interaction.config`: `build_probe_parser`, `parse_probe_args`
- `pba.interaction.tokens`: `collect_token_index`
- `pba.interaction.attention`: `AttentionAccumulator`
- `pba.interaction.outputs`: `prepare_output_root`, `build_probe_metadata`
- `pba.interaction.batch`: `build_batch_sample_id`,
  `build_batch_sample_paths`, `validate_metric_columns`
- `pba.interaction.runtime`: `build_runtime_boundary`,
  `validate_interaction_metadata`, `validate_interaction_outputs`

## Heavy Runtime Boundaries

These public APIs intentionally do not implement missing paper-release gaps:

- AGD20K batch interaction evaluation
- DINO/DINOv3 fusion cache generation
- DINOv3 external source replacement
- SD2.1 local model handling
- UMD PyTorch dataset migration, whose public schema is documented in
  `docs/umd_dataset_contract.md`
- Geometry heavy runtime migration, whose boundary is documented in
  `docs/geometry_runtime_contract.md`
- Fusion heavy runtime migration, whose boundary is documented in
  `docs/fusion_runtime_contract.md`
- Interaction heavy runtime migration, whose boundary is documented in
  `docs/interaction_runtime_contract.md`

Heavy runtime code remains in the original experiment directories until the true
development code is available.

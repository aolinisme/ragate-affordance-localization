# Release Status

This page is the consolidated public-release state. It is the first file to
read before deciding whether to push the repository or continue migration work.

## Runnable Surface

The current public source exposes these maintained entrypoints:

- `python -m pba.audit --config configs/reproduce/main.yaml --skip-asset-check --show-gaps --public-release`
- `python -m pba.reproduce main --config configs/reproduce/main.yaml --skip-asset-check --dry-run`
- `python -m pba.run --list`
- `python -m pba.run geometry-train -- --config geometry_probing/umd_linear_probing/configs/dinov2.yaml`
- `python -m pba.run geometry-eval -- /path/to/linear_probe.pth --config geometry_probing/umd_linear_probing/configs/dinov2.yaml --split test`
- `python -m pba.run interaction-probe -- --model-id models/FLUX.1-Kontext-dev --image /path/to/input.png --prompt "hold toothbrush" --affordance hold --output-root probe_outputs`
- `python -m pba.run fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml`

Heavy commands still require local datasets, checkpoints, model source trees,
and runtime dependencies. The release check path is lightweight.

## Contract Coverage

The current release has explicit contracts for all known missing or partially
migrated public surfaces:

- `dinov3-external-source`: `docs/dinov3_source_strategy.md`
- `fusion-cache-build`: `docs/fusion_cache_contract.md`
- `interaction-batch-eval`: `docs/interaction_batch_contract.md`
- `sd21-local-model`: `docs/sd21_local_model_contract.md`
- `umd-pytorch-dataset-migration`: `docs/umd_dataset_contract.md`
- `geometry-heavy-runtime-migration`: `docs/geometry_runtime_contract.md`
- `fusion-heavy-runtime-migration`: `docs/fusion_runtime_contract.md`
- `interaction-heavy-runtime-migration`: `docs/interaction_runtime_contract.md`

Contracts define boundaries, schemas, output files, and future integration
points. They do not close gaps that require true development code.

## Known Warnings

`scripts/check_public_release.sh` may print warning lines in the current public
source:

- `interaction-jobs-empty`: `configs/reproduce/main.yaml` has no AGD20K batch
  interaction job.
- `fusion-cache-only`: fusion evaluation expects an existing DINO cache because
  `geom_pipeline.dino_cache_only` is enabled.

These warnings are accepted for the current GitHub source because they map to
documented gaps.

## Open Gaps

The current GitHub source intentionally keeps these gaps open:

- `interaction-batch-eval`
- `fusion-cache-build`
- `dinov3-external-source`
- `sd21-local-model`
- `umd-pytorch-dataset-migration`
- `geometry-heavy-runtime-migration`
- `fusion-heavy-runtime-migration`
- `interaction-heavy-runtime-migration`

Do not remove these gaps until the real implementation is present and verified.

## Before Push

Run:

```bash
bash scripts/check_public_release.sh
python -m pytest -q
git status --short
```

Before pushing, confirm:

- no private paths are tracked
- `docs/superpowers/` is ignored
- `.worktrees/` is ignored
- datasets, checkpoints, generated caches, and model weights are not tracked
- every accepted warning maps to a documented gap above

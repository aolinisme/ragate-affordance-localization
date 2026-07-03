# Release Usage

This page gives the public release entrypoints. It separates reproducible
release checks from heavy model execution.

For consolidated current state, accepted warnings, and open gaps, see
`docs/release_status.md`.

## Audit First

List configured jobs and known release gaps:

```bash
python -m pba.audit --config configs/reproduce/main.yaml --skip-asset-check --show-gaps
```

Require local assets before reporting readiness:

```bash
python -m pba.audit --config configs/reproduce/main.yaml --show-gaps
```

Before pushing a public release branch, run maintainer checks:

```bash
bash scripts/check_public_release.sh
```

This adds private-path scanning, `docs/superpowers/` ignore validation,
third-party DINOv3 notice validation, reproduction dry-run, and lightweight
release tests.

For a lightweight release-check environment:

```bash
python -m pip install --upgrade pip setuptools
pip install -e .
pip install -r requirements-smoke.txt
pba-audit --config configs/reproduce/main.yaml --skip-asset-check --show-gaps --public-release
```

## One-Click Dry Run

Print the main reproduction command graph without running models:

```bash
bash scripts/reproduce_main.sh --config configs/reproduce/main.yaml --skip-asset-check --dry-run
```

Inspect registered commands:

```bash
python -m pba.run --list
```

## Single-Image Interaction Probing

Run one FLUX Kontext cross-attention probe:

```bash
python -m pba.run interaction-probe -- \
  --model-id models/FLUX.1-Kontext-dev \
  --image /path/to/input.png \
  --prompt "hold toothbrush" \
  --affordance hold \
  --output-root probe_outputs
```

This writes `generated.png`, `meta.json`, and affordance heatmap files under
the output root.

## Geometry Probing

Train a UMD linear probe:

```bash
python -m pba.run geometry-train -- \
  --config geometry_probing/umd_linear_probing/configs/dinov2.yaml
```

Evaluate a checkpoint:

```bash
python -m pba.run geometry-eval -- \
  /path/to/linear_probe.pth \
  --config geometry_probing/umd_linear_probing/configs/dinov2.yaml \
  --split test
```

## Fusion Evaluation

Run the AGD20K fusion evaluator after required assets and cache files exist:

```bash
python -m pba.run fusion-eval -- \
  --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

The current public config expects an existing DINO cache because
`geom_pipeline.dino_cache_only` is enabled.
See `docs/fusion_cache_contract.md` for the expected cache filename and NPZ
keys.

## Known Missing Pieces

The current GitHub source intentionally does not guess missing development code:

- `interaction-batch-eval`: AGD20K batch interaction-only evaluation
- `fusion-cache-build`: DINO/DINOv3 fusion cache generation
- `dinov3-external-source`: verified external DINOv3 source path
- `sd21-local-model`: offline Stable Diffusion 2.1 model convention
- `umd-pytorch-dataset-migration`: UMD dataset heavy runtime migration; schema
  contract lives in `docs/umd_dataset_contract.md`
- `geometry-heavy-runtime-migration`: trainer/evaluator/backbone migration;
  boundary lives in `docs/geometry_runtime_contract.md`
- `fusion-heavy-runtime-migration`: FLUX/DINO/PCA/ROI/geometry-mask fusion
  runtime migration; boundary lives in `docs/fusion_runtime_contract.md`
- `interaction-heavy-runtime-migration`: FLUX attention processor/runtime
  migration; boundary lives in `docs/interaction_runtime_contract.md`

See `docs/release_extension_points.md` for integration details.

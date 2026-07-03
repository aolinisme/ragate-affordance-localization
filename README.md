# RA-Gate Affordance Localization

This is a clean paper-facing release for **Reliability-Aware
Geometry-Interaction Fusion for Low-Compute Affordance Localization**.

The release contains the implementation of RA-Gate, the SD-Turbo + DINOv3
low-compute path, TinyRouter compute-policy experiments, configuration
templates, tests, and reproduction notes. It intentionally excludes datasets,
checkpoints, generated heatmaps, caches, and submission manuscripts.

## Contributions Implemented Here

- **RA-Gate:** a training-free reliability gate that adapts the fusion weight
  between FLUX interaction heatmaps and DINOv3/PCA geometry primitives.
- **SD-Turbo + DINOv3 path:** a low-compute online branch that uses SD-Turbo
  verb/object attention with DINOv3/PCA geometry.
- **TinyRouter:** a small compute-policy network for deciding when FLUX fallback
  is useful.
- **Low-resource geometry probing:** supporting single-GPU experiments for
  geometry-aware frozen-backbone affordance probing.

## Official Model Links

The repository does not redistribute third-party model weights. Use the
official providers and follow their licenses and access conditions:

- DINOv3-S/16 checkpoint: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- DINOv3 code release: https://github.com/facebookresearch/dinov3
- FLUX.1 Kontext-dev: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- SD-Turbo: https://huggingface.co/stabilityai/sd-turbo

## Expected Local Layout

Place datasets and models outside git-tracked files:

```text
datasets/
  AGD20K/AGD20K/Unseen/testset/
  UMD/part-affordance-dataset/
models/
  FLUX.1-Kontext-dev/
  dinov3/
  dinov3_vits16_pretrain_lvd1689m.pth
outputs/
  fusion_cache/
  fusion_zero_shot/
```

Machine-specific paths should be placed in ignored local config files such as
`config.local.yaml` or `fusion_zero_shot/src/agd20k_eval/config.local.yaml`.

## Quick Check

```bash
python -m pip install -e .
python -m pytest tests/test_public_api_smoke.py tests/test_pba_run.py tests/test_fusion_adaptive_gate.py tests/test_tiny_reliability_router.py -q
```

## Main Entry Points

Inspect available commands:

```bash
python -m pba.run --list
```

Run the AGD20K fusion pipeline:

```bash
python run.py fusion-eval -- --config fusion_zero_shot/src/agd20k_eval/config.yaml
```

Run the SD-Turbo/DINOv3 analysis scripts:

```bash
python fusion_zero_shot/scripts/run_sd_turbo_attention_pilot.py --help
python fusion_zero_shot/scripts/run_sd_dino_fusion_from_attention.py --help
python fusion_zero_shot/scripts/run_tiny_reliability_router.py --help
```

## Notes

- No datasets, checkpoints, private caches, or generated manuscript files are
  included in this release.
- Third-party model access should follow the corresponding provider terms.
- Local reproduction is hardware-dependent; the experiments in the paper were
  developed for single-GPU execution on an RTX 3070 Laptop GPU.

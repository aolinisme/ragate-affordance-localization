# Code and Model Availability

This repository contains the paper-facing implementation for reliability-aware
geometry-interaction affordance localization:

- RA-Gate for adaptive DINOv3/PCA and FLUX Kontext fusion.
- SD-Turbo + DINOv3 low-compute affordance localization experiments.
- TinyRouter compute-policy experiments for deciding when to call FLUX.
- UMD low-resource geometry probing utilities used as supporting experiments.

The repository does not redistribute datasets, generated caches, model
checkpoints, or third-party model weights.

## Official Model Sources

Use the official providers and follow their licenses and access conditions.

| Component | Role in the paper | Official source |
| --- | --- | --- |
| DINOv3-S/16 | Dense geometry and object-part patch tokens | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m |
| DINOv3 code | Reference implementation for DINOv3 backbones | https://github.com/facebookresearch/dinov3 |
| FLUX.1 Kontext-dev | Slow interaction reference and fallback branch | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev |
| SD-Turbo | Fast verb-conditioned online interaction prior | https://huggingface.co/stabilityai/sd-turbo |

## Reproduction Policy

The code is released without datasets, checkpoints, or cached heatmaps. Place
local assets under ignored `datasets/`, `models/`, and `outputs/` directories,
or point the config templates to equivalent local paths.

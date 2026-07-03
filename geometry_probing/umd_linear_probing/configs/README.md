# Geometry Configs

This folder keeps the public default configs for geometry probing.

Available configs:

- `clip.yaml`
- `dino.yaml`
- `dinov2.yaml`
- `dinov3.yaml`
- `sam.yaml`
- `sd21.yaml`
- `siglip.yaml`

Asset checklist:

| Config | Required dataset | Required model assets |
| --- | --- | --- |
| `dino.yaml` | `datasets/UMD/part-affordance-dataset/` | `models/dino/`, `models/dino_vitbase16_pretrain.pth` |
| `dinov2.yaml` | `datasets/UMD/part-affordance-dataset/` | `models/dinov2/`, `models/dinov2_vitb14_pretrain.pth` |
| `dinov3.yaml` | `datasets/UMD/part-affordance-dataset/` | `models/dinov3/`, `models/dinov3_vit7b16_pretrain_lvd1689m.pth` |
| `clip.yaml` | `datasets/UMD/part-affordance-dataset/` | `models/open_clip/src/` |
| `siglip.yaml` | `datasets/UMD/part-affordance-dataset/` | Hugging Face model id `google/siglip-base-patch16-384` |
| `sam.yaml` | `datasets/UMD/part-affordance-dataset/` | `models/sam_vit_b_01ec64.pth` |
| `sd21.yaml` | `datasets/UMD/part-affordance-dataset/` | Hugging Face model id `stabilityai/stable-diffusion-2-1` |

All configs also reference
`metadata/splits/category_split_seed42_v20.json` and
`metadata/splits/metric3d_predictions.json`.

Usage:

```bash
python scripts/train.py --config configs/dinov2.yaml
python scripts/eval.py /path/to/linear_probe.pth --config configs/dinov2.yaml --split test
```

Conventions:

- dataset assets resolve from `datasets/`
- model source trees and checkpoints resolve from `models/`
- `sam.yaml` expects `models/sam_vit_b_01ec64.pth`; it does not rely on
  implicit checkpoint download for paper reproduction
- geometry side-data manifest resolves from `metadata/splits/`

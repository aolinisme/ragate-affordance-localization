# DINOv3 Source Strategy

This clean public release uses user-provided DINOv3 source and checkpoint
files. It does not vendor the upstream DINOv3 source tree and does not
redistribute DINOv3 checkpoints.

## Current Release Policy

Current mode: `external-source`.

Users should obtain the official DINOv3 implementation from:

- https://github.com/facebookresearch/dinov3

and place or symlink it under:

```text
models/dinov3/
```

The DINOv3-S/16 checkpoint used by the paper-facing experiments is available
from the official Hugging Face page:

- https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

Keep checkpoints outside git, for example:

```text
models/dinov3_vits16_pretrain_lvd1689m.pth
```

## Local Development Note

Some development workspaces may still use a bundled-source layout under:

```text
fusion_zero_shot/src/dino/third_party/dinov3
```

That directory is intentionally not included in this clean release. If a local
fork uses the bundled path, keep the upstream DINOv3 license and model card
with it and do not commit checkpoints or generated caches.

## Do Not Commit

Do not commit DINOv3 checkpoints, generated token caches, or third-party model
weights. They belong under ignored local `models/` and `outputs/` directories.

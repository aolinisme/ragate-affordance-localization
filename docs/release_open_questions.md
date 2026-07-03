# Release Open Questions

These questions record gaps between the paper-release goal and the clean public
source. Do not invent implementation to answer them.

Run the readiness audit to surface current-source gaps from configs:

```bash
bash scripts/audit_release_readiness.sh --skip-asset-check
```

See [`release_extension_points.md`](./release_extension_points.md) for the
expected interfaces where future development code should fill these gaps.

## Interaction Reproduction

- Current `configs/reproduce/main.yaml` leaves `jobs.interaction` empty because
  the existing `interaction-probe` entrypoint requires a single image, prompt,
  and affordance. A batch AGD20K interaction-only reproduction entrypoint must
  be confirmed before adding it to the main reproduction command.

## DINOv3 Source Handling

- The clean public release requires user-provided DINOv3 source under
  `models/dinov3/` or an equivalent path configured locally.
- The release does not vendor the upstream DINOv3 source tree and does not
  redistribute DINOv3 checkpoints. Users should obtain them from the official
  Meta/Facebook Research sources.
  See [`dinov3_source_strategy.md`](./dinov3_source_strategy.md).

## Stable Diffusion Offline Handling

- Current `sd21.yaml` uses Hugging Face model id
  `stabilityai/stable-diffusion-2-1`. Offline or local Stable Diffusion model
  handling is not resolved in every entrypoint, so the release should not claim
  full offline SD2.1 reproduction until the accepted local model path convention
  in [`sd21_local_model_contract.md`](./sd21_local_model_contract.md) is wired
  into the loader.

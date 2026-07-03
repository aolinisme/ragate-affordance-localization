# Stable Diffusion 2.1 Local Model Contract

This document defines the accepted Stable Diffusion 2.1 model references for
geometry probing. It is a contract, not an offline loader implementation.

## Online Mode

Current `sd21.yaml` uses the Hugging Face model id:

```text
stabilityai/stable-diffusion-2-1
```

This remains the public default because the current loader calls
`from_pretrained(model_id)`.

## Offline Mode

Future offline reproduction should use a local model directory:

```text
models/stable-diffusion-2-1
```

The public helper is `pba.geometry.sd21.resolve_sd21_model_reference`.

## Required Subdirectories

The local model directory must contain at least:

- `unet`
- `scheduler`
- `tokenizer`
- `text_encoder`
- `vae`

Use `pba.geometry.sd21.validate_sd21_local_model_dir` for lightweight checks.

## Reproduction Integration

When true development code is available:

- add config keys for `local_model_dir` and `offline`
- thread the resolved model reference into the Stable Diffusion backbone loader
- keep online Hugging Face mode as an explicit option
- add an audit check that blocks offline SD2.1 reproduction if local model files
  are missing

## Do Not Claim Offline Reproduction Yet

Do not claim offline SD2.1 reproduction in the current public release. The
current release documents the path convention and validation surface only.

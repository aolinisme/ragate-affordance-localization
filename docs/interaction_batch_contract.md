# Interaction Batch Contract

This document defines the missing AGD20K interaction-only batch evaluation
interface. It is a contract, not a runner implementation.

## Missing Runner

Current code exposes single-image Flux Kontext probing through
`interaction-probe`. The public GitHub source does not include the AGD20K batch
runner needed for interaction-only paper reproduction.

## Inputs

Expected future runner inputs:

- AGD20K Unseen/testset root
- Flux Kontext model path
- prompt templates
- output root
- device list
- denoising steps
- guidance scale
- seed policy
- optional affordance subset
- optional max images per object

## Outputs

For each AGD20K sample, write outputs under:

```text
{output_root}/{affordance}/{object}/{image_stem}/
```

Required files:

- `verb_heat.png`
- `verb_heat.npy`
- `meta.json`

The public helper is `pba.interaction.batch.build_batch_sample_paths`.

## Metrics

The batch runner should write `metrics.csv` with at least:

```text
sample_id,affordance,object_name,image_path,gt_path,verb_heatmap,kld,sim,nss
```

The metrics must use the same KLD, SIM, and NSS implementations exposed by
`pba.metrics.affordance`.

## Reproduction Integration

After true development code is available:

- add a new `run.py` command for AGD20K batch interaction evaluation
- add that command to `configs/reproduce/main.yaml` under `jobs.interaction`
- keep the single-image `interaction-probe` command for manual probing

## Do Not Guess Batch Logic

Do not fabricate the batch runner from the public source alone. Use this
contract to keep future development code compatible with current release docs,
tests, and metric expectations.

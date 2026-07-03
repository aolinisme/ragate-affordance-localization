# Interaction Runtime Migration Contract

This document fixes the migration boundary for the single-image FLUX Kontext
interaction probing heavy runtime. It does not move diffusers, torch, FLUX
pipeline loading, or attention processor execution code yet.

## Lightweight API

`pba.interaction.runtime` is safe to import without PyTorch, diffusers, PIL, or
model assets. It records the accepted runtime surface:

- `INTERACTION_RUNTIME_MODULES`
- `INTERACTION_RUNTIME_STAGES`
- `INTERACTION_OUTPUT_FILES`
- `INTERACTION_TOKEN_METADATA_KEYS`
- `InteractionRuntimeBoundary`
- `build_runtime_boundary()`
- `validate_interaction_metadata(metadata)`
- `validate_interaction_outputs(outputs, affordance=...)`

## Current Boundary

The legacy entrypoint remains:

```text
interaction_probing/cross_attention_probe/cross_attention_probe.py
```

The registered command remains:

```text
interaction-probe
```

The migration unit includes:

- `FluxImg2ImgPipeline.from_pretrained`
- `FluxAttnRecorderProcessor`
- `_iter_flux_attention_modules`
- affordance token matching
- image-query/text-key probability capture
- attention accumulation and summary
- generated image saving
- heatmap, overlay, and NPY export
- metadata writing

## Runtime Stages

`INTERACTION_RUNTIME_STAGES` preserves the single-image probe flow:

- `load_input_image`
- `load_flux_pipeline`
- `collect_affordance_token`
- `install_attention_processors`
- `run_generation`
- `summarize_attention_maps`
- `export_generated_image_attention_and_metadata`

The migrated runtime must keep these stages separable enough that future
AGD20K batch interaction evaluation can reuse the same Flux execution and
attention export behavior.

## Output Contract

`INTERACTION_OUTPUT_FILES` preserves the current output shape:

- `generated.png`
- `meta.json`
- `attention/{affordance}_heat.png`
- `attention/{affordance}_overlay.png`
- `attention/{affordance}_heat.npy`

The heatmap and overlay are resized to the input image resolution by
`AttentionAccumulator.export`.

## Metadata Contract

`meta.json` must preserve:

- `model_id`
- `prompt`
- `affordance`
- `tokens_tracked`
- `steps`
- `guidance`
- `seed`

`validate_interaction_metadata` checks this shape without importing the heavy
runtime.

## Batch Dependency

`interaction-batch-eval` should reuse this runtime once the true batch
development code is available. The batch output and metric contract is
documented in `docs/interaction_batch_contract.md`.

Do not claim AGD20K interaction-only reproduction until batch sampling,
attention export, and metric aggregation are implemented against the real
development code.

## Migration Rule

Move the interaction heavy runtime only after the real interaction probing code
is ready. Keep the legacy single-image probe script as a wrapper until the
registered command, output files, metadata, and future batch runner are verified.

"""Lightweight interaction runtime migration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "INTERACTION_OUTPUT_FILES",
    "INTERACTION_RUNTIME_MODULES",
    "INTERACTION_RUNTIME_STAGES",
    "INTERACTION_TOKEN_METADATA_KEYS",
    "InteractionRuntimeBoundary",
    "build_runtime_boundary",
    "validate_interaction_metadata",
    "validate_interaction_outputs",
]


INTERACTION_RUNTIME_MODULES = (
    "FluxImg2ImgPipeline.from_pretrained",
    "FluxAttnRecorderProcessor",
    "_iter_flux_attention_modules",
    "pba.interaction.tokens.collect_token_index",
    "pba.interaction.attention.AttentionAccumulator",
    "AttentionAccumulator.add_from_probs",
    "AttentionAccumulator.summary",
    "AttentionAccumulator.export",
    "pba.interaction.outputs.prepare_output_root",
    "pba.interaction.outputs.build_probe_metadata",
)

INTERACTION_RUNTIME_STAGES = (
    "load_input_image",
    "load_flux_pipeline",
    "collect_affordance_token",
    "install_attention_processors",
    "run_generation",
    "summarize_attention_maps",
    "export_generated_image_attention_and_metadata",
)

INTERACTION_OUTPUT_FILES = (
    "generated.png",
    "meta.json",
    "attention/{affordance}_heat.png",
    "attention/{affordance}_overlay.png",
    "attention/{affordance}_heat.npy",
)

INTERACTION_TOKEN_METADATA_KEYS = (
    "model_id",
    "prompt",
    "affordance",
    "tokens_tracked",
    "steps",
    "guidance",
    "seed",
)


@dataclass(frozen=True)
class InteractionRuntimeBoundary:
    """Public migration boundary for Flux interaction probing runtime."""

    package_module: str
    legacy_entrypoint: str
    command: str
    modules: tuple[str, ...]
    stages: tuple[str, ...]
    output_files: tuple[str, ...]


def build_runtime_boundary() -> InteractionRuntimeBoundary:
    """Return the accepted interaction heavy-runtime migration boundary."""

    return InteractionRuntimeBoundary(
        package_module="pba.interaction",
        legacy_entrypoint="interaction_probing/cross_attention_probe/cross_attention_probe.py",
        command="interaction-probe",
        modules=INTERACTION_RUNTIME_MODULES,
        stages=INTERACTION_RUNTIME_STAGES,
        output_files=INTERACTION_OUTPUT_FILES,
    )


def validate_interaction_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate single-image interaction metadata shape."""

    missing = [key for key in INTERACTION_TOKEN_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Interaction metadata missing required keys: {', '.join(missing)}")
    token_map = metadata["tokens_tracked"]
    if not isinstance(token_map, Mapping) or not token_map:
        raise ValueError("Interaction metadata tokens_tracked must be a non-empty mapping.")
    return metadata


def validate_interaction_outputs(
    outputs: Mapping[str, Any],
    *,
    affordance: str,
) -> Mapping[str, Any]:
    """Validate output file presence flags using the public output contract."""

    expected = [template.format(affordance=affordance) for template in INTERACTION_OUTPUT_FILES]
    missing = [path for path in expected if not outputs.get(path)]
    if missing:
        raise ValueError(f"Interaction outputs missing required files: {', '.join(missing)}")
    return outputs

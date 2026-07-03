from pba.interaction.runtime import (
    INTERACTION_OUTPUT_FILES,
    INTERACTION_RUNTIME_MODULES,
    INTERACTION_RUNTIME_STAGES,
    INTERACTION_TOKEN_METADATA_KEYS,
    InteractionRuntimeBoundary,
    build_runtime_boundary,
    validate_interaction_metadata,
    validate_interaction_outputs,
)


def test_build_runtime_boundary_documents_legacy_flux_runtime() -> None:
    boundary = build_runtime_boundary()

    assert boundary == InteractionRuntimeBoundary(
        package_module="pba.interaction",
        legacy_entrypoint="interaction_probing/cross_attention_probe/cross_attention_probe.py",
        command="interaction-probe",
        modules=INTERACTION_RUNTIME_MODULES,
        stages=INTERACTION_RUNTIME_STAGES,
        output_files=INTERACTION_OUTPUT_FILES,
    )
    assert "FluxImg2ImgPipeline.from_pretrained" in boundary.modules
    assert "FluxAttnRecorderProcessor" in boundary.modules
    assert "_iter_flux_attention_modules" in boundary.modules
    assert "AttentionAccumulator.export" in boundary.modules


def test_interaction_runtime_stages_name_flux_hook_and_export_flow() -> None:
    assert INTERACTION_RUNTIME_STAGES == (
        "load_input_image",
        "load_flux_pipeline",
        "collect_affordance_token",
        "install_attention_processors",
        "run_generation",
        "summarize_attention_maps",
        "export_generated_image_attention_and_metadata",
    )


def test_interaction_output_contract_names_generated_attention_and_metadata() -> None:
    assert INTERACTION_OUTPUT_FILES == (
        "generated.png",
        "meta.json",
        "attention/{affordance}_heat.png",
        "attention/{affordance}_overlay.png",
        "attention/{affordance}_heat.npy",
    )


def test_validate_interaction_metadata_accepts_probe_metadata_shape() -> None:
    metadata = {
        "model_id": "models/FLUX.1-Kontext-dev",
        "prompt": "hold toothbrush",
        "affordance": "hold",
        "tokens_tracked": {"hold": 2},
        "steps": 28,
        "guidance": 2.5,
        "seed": 0,
    }

    assert validate_interaction_metadata(metadata) == metadata
    assert INTERACTION_TOKEN_METADATA_KEYS == tuple(metadata.keys())


def test_validate_interaction_metadata_rejects_missing_token_map() -> None:
    metadata = {
        "model_id": "models/FLUX.1-Kontext-dev",
        "prompt": "hold toothbrush",
        "affordance": "hold",
    }

    try:
        validate_interaction_metadata(metadata)
    except ValueError as exc:
        assert "tokens_tracked" in str(exc)
    else:
        raise AssertionError("validate_interaction_metadata should reject incomplete metadata")


def test_validate_interaction_outputs_accepts_expected_file_flags() -> None:
    outputs = {
        "generated.png": True,
        "meta.json": True,
        "attention/hold_heat.png": True,
        "attention/hold_overlay.png": True,
        "attention/hold_heat.npy": True,
    }

    assert validate_interaction_outputs(outputs, affordance="hold") == outputs

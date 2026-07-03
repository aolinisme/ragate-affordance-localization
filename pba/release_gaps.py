from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseGap:
    code: str
    title: str
    current_status: str
    expected_extension: str
    integration_point: str


def get_release_gaps() -> tuple[ReleaseGap, ...]:
    return (
        ReleaseGap(
            code="interaction-batch-eval",
            title="AGD20K batch interaction-only evaluation",
            current_status=(
                "Current GitHub source exposes single-image Flux interaction probing, "
                "but main reproduction has no AGD20K batch interaction job."
            ),
            expected_extension=(
                "Provide a batch runner that extracts verb attention maps for AGD20K "
                "samples and reports interaction-only KLD, SIM, and NSS following "
                "docs/interaction_batch_contract.md."
            ),
            integration_point="configs/reproduce/main.yaml jobs.interaction",
        ),
        ReleaseGap(
            code="fusion-cache-build",
            title="DINO/DINOv3 fusion cache generation",
            current_status=(
                "Fusion config sets geom_pipeline.dino_cache_only=true, so evaluation "
                "expects outputs/fusion_cache to already exist."
            ),
            expected_extension=(
                "Provide a cache build command that writes the DINO token artifacts "
                "consumed by fusion_zero_shot."
            ),
            integration_point="fusion_zero_shot/src/agd20k_eval/config.yaml geom_pipeline.cache_root",
        ),
        ReleaseGap(
            code="dinov3-external-source",
            title="External DINOv3 source tree",
            current_status="Current GitHub source still contains a bundled DINOv3 third-party tree.",
            expected_extension=(
                "Switch imports to a user-provided models/dinov3 source tree once the "
                "true development dependency path is confirmed."
            ),
            integration_point="models/dinov3/",
        ),
        ReleaseGap(
            code="sd21-local-model",
            title="Stable Diffusion 2.1 local model handling",
            current_status="sd21.yaml uses Hugging Face model id stabilityai/stable-diffusion-2-1.",
            expected_extension=(
                "Define an accepted local/offline model path convention before claiming "
                "offline SD2.1 reproduction. See docs/sd21_local_model_contract.md."
            ),
            integration_point="geometry_probing/umd_linear_probing/configs/sd21.yaml",
        ),
        ReleaseGap(
            code="umd-pytorch-dataset-migration",
            title="UMD PyTorch dataset migration",
            current_status=(
                "UMD split/path helpers live in pba.data.geometry, but the full PyTorch "
                "dataset remains in geometry_probing."
            ),
            expected_extension=(
                "Move UMDAffordanceDataset, transforms, collate behavior, and optional "
                "geometry feature attachment when geometry train/eval code is migrated. "
                "Preserve the schema documented in docs/umd_dataset_contract.md."
            ),
            integration_point="geometry_probing/umd_linear_probing/src/data/dataset.py",
        ),
        ReleaseGap(
            code="geometry-heavy-runtime-migration",
            title="Geometry heavy runtime migration",
            current_status=(
                "Geometry config loading and train/eval entrypoints live in pba.geometry, "
                "but trainer, evaluator, backbone builders, transforms, and logging remain "
                "under geometry_probing."
            ),
            expected_extension=(
                "Move heavy runtime modules only after the real development code is ready "
                "to preserve the current paper reproduction behavior. Preserve the "
                "boundary documented in docs/geometry_runtime_contract.md."
            ),
            integration_point="geometry_probing/umd_linear_probing/src",
        ),
        ReleaseGap(
            code="fusion-heavy-runtime-migration",
            title="Fusion heavy runtime migration",
            current_status=(
                "Fusion config, Kontext resolution, prompt/token, and heatmap path "
                "helpers live in pba.fusion, but FLUX, DINO, PCA, ROI, geometry mask, "
                "and metric aggregation runtime remains under fusion_zero_shot."
            ),
            expected_extension=(
                "Move heavy fusion runtime modules only after cache generation and true "
                "development fusion code are available. Preserve the boundary documented "
                "in docs/fusion_runtime_contract.md."
            ),
            integration_point="fusion_zero_shot/src/agd20k_eval/run_flux_kontext_eval.py",
        ),
        ReleaseGap(
            code="interaction-heavy-runtime-migration",
            title="Interaction heavy runtime migration",
            current_status=(
                "Interaction parser, token matching, attention accumulation, output, "
                "and metadata helpers live in pba.interaction, but Flux loading, "
                "attention processor injection, denoising, and generated image saving "
                "remain under interaction_probing."
            ),
            expected_extension=(
                "Move heavy Flux runtime only after the real interaction probing code "
                "is ready and before adding AGD20K batch interaction evaluation. "
                "Preserve the boundary documented in docs/interaction_runtime_contract.md."
            ),
            integration_point="interaction_probing/cross_attention_probe/cross_attention_probe.py",
        ),
    )

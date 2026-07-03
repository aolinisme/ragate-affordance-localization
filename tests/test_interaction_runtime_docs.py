from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_interaction_runtime_contract_doc_exists_and_names_boundaries() -> None:
    text = (REPO_ROOT / "docs/interaction_runtime_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Interaction Runtime Migration Contract",
        "pba.interaction.runtime",
        "cross_attention_probe.py",
        "FluxImg2ImgPipeline.from_pretrained",
        "FluxAttnRecorderProcessor",
        "_iter_flux_attention_modules",
        "AttentionAccumulator.export",
        "INTERACTION_OUTPUT_FILES",
        "INTERACTION_RUNTIME_STAGES",
    ):
        assert phrase in text


def test_release_docs_link_interaction_runtime_contract() -> None:
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/public_api.md",
        REPO_ROOT / "docs/release_extension_points.md",
        REPO_ROOT / "docs/release_usage.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "docs/interaction_runtime_contract.md" in text

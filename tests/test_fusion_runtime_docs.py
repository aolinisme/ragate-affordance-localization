from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fusion_runtime_contract_doc_exists_and_names_boundaries() -> None:
    text = (REPO_ROOT / "docs/fusion_runtime_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Fusion Runtime Migration Contract",
        "pba.fusion.runtime",
        "run_flux_kontext_eval.py",
        "run_kontext_generation",
        "warp_heatmap_cli",
        "extract_dino_tokens",
        "run_pca",
        "generate_geometry_mask",
        "FUSION_SUMMARY_COLUMNS",
        "FUSION_SAMPLE_ARTIFACTS",
    ):
        assert phrase in text


def test_release_docs_link_fusion_runtime_contract() -> None:
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/public_api.md",
        REPO_ROOT / "docs/release_extension_points.md",
        REPO_ROOT / "docs/release_usage.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "docs/fusion_runtime_contract.md" in text

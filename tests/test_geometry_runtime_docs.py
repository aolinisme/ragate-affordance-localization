from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_geometry_runtime_contract_doc_exists_and_names_boundaries() -> None:
    text = (REPO_ROOT / "docs/geometry_runtime_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Geometry Runtime Migration Contract",
        "pba.geometry.runtime",
        "LinearProbeExperiment",
        "evaluate_linear_probe",
        "UMDAffordanceDataset",
        "GEOMETRY_BACKBONE_TARGETS",
        "GEOMETRY_TRAINING_OUTPUT_FILES",
        "geometry_probing/umd_linear_probing/src",
    ):
        assert phrase in text


def test_release_docs_link_geometry_runtime_contract() -> None:
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/public_api.md",
        REPO_ROOT / "docs/release_extension_points.md",
        REPO_ROOT / "docs/release_usage.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "docs/geometry_runtime_contract.md" in text

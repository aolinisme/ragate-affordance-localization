from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_usage_doc_exists_and_covers_public_workflows() -> None:
    text = (REPO_ROOT / "docs/release_usage.md").read_text(encoding="utf-8")

    for heading in (
        "Release Usage",
        "Audit First",
        "One-Click Dry Run",
        "Single-Image Interaction Probing",
        "Geometry Probing",
        "Fusion Evaluation",
        "Known Missing Pieces",
    ):
        assert heading in text
    assert "python -m pba.run --list" in text
    assert "scripts/reproduce_main.sh" in text
    assert "interaction-batch-eval" in text
    assert "fusion-cache-build" in text


def test_readme_links_public_api_and_release_usage_docs() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/public_api.md" in text
    assert "docs/release_usage.md" in text
    assert "docs/release_status.md" in text
    assert "docs/dinov3_source_strategy.md" in text
    assert "docs/umd_dataset_contract.md" in text
    assert "docs/geometry_runtime_contract.md" in text
    assert "docs/fusion_runtime_contract.md" in text
    assert "docs/interaction_runtime_contract.md" in text


def test_release_checklist_documents_public_release_risks() -> None:
    text = (REPO_ROOT / "docs/release_checklist.md").read_text(encoding="utf-8")

    for phrase in (
        "Public Release Checklist",
        "Private Paths",
        "Large Tracked Files",
        "Ignored Local Agent Files",
        "Third-Party Source",
        "DINOv3",
        "docs/dinov3_source_strategy.md",
        "Missing Development Code",
        "scripts/check_public_release.sh",
    ):
        assert phrase in text


def test_public_release_check_script_exists() -> None:
    text = (REPO_ROOT / "scripts/check_public_release.sh").read_text(encoding="utf-8")

    assert "python -m pba.audit" in text
    assert "--public-release" in text
    assert "python -m pba.reproduce main" in text
    assert "python -m pytest" in text
    assert "tests/test_release_status_docs.py" in text
    assert "tests/test_packaging_metadata.py" in text
    assert "tests/test_fusion_cache_contract.py" in text
    assert "tests/test_fusion_cache_docs.py" in text
    assert "tests/test_interaction_batch_contract.py" in text
    assert "tests/test_interaction_batch_docs.py" in text
    assert "tests/test_sd21_model_contract.py" in text
    assert "tests/test_sd21_model_docs.py" in text
    assert "tests/test_umd_dataset_contract.py" in text
    assert "tests/test_umd_dataset_docs.py" in text
    assert "tests/test_geometry_runtime_contract.py" in text
    assert "tests/test_geometry_runtime_docs.py" in text
    assert "tests/test_fusion_runtime_contract.py" in text
    assert "tests/test_fusion_runtime_docs.py" in text
    assert "tests/test_interaction_runtime_contract.py" in text
    assert "tests/test_interaction_runtime_docs.py" in text

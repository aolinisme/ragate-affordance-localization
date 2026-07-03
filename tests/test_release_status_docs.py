from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


CONTRACT_DOCS = (
    "docs/dinov3_source_strategy.md",
    "docs/fusion_cache_contract.md",
    "docs/interaction_batch_contract.md",
    "docs/sd21_local_model_contract.md",
    "docs/umd_dataset_contract.md",
    "docs/geometry_runtime_contract.md",
    "docs/fusion_runtime_contract.md",
    "docs/interaction_runtime_contract.md",
)

RELEASE_GAPS = (
    "interaction-batch-eval",
    "fusion-cache-build",
    "dinov3-external-source",
    "sd21-local-model",
    "umd-pytorch-dataset-migration",
    "geometry-heavy-runtime-migration",
    "fusion-heavy-runtime-migration",
    "interaction-heavy-runtime-migration",
)


def test_release_status_doc_summarizes_current_public_state() -> None:
    text = (REPO_ROOT / "docs/release_status.md").read_text(encoding="utf-8")

    for heading in (
        "Release Status",
        "Runnable Surface",
        "Contract Coverage",
        "Known Warnings",
        "Open Gaps",
        "Before Push",
    ):
        assert heading in text

    for doc in CONTRACT_DOCS:
        assert doc in text
    for gap in RELEASE_GAPS:
        assert gap in text
    assert "scripts/check_public_release.sh" in text
    assert "135 passed, 1 skipped" not in text


def test_release_guides_link_release_status_doc() -> None:
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/release_usage.md",
        REPO_ROOT / "docs/release_checklist.md",
        REPO_ROOT / "docs/public_api.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "docs/release_status.md" in text


def test_release_checklist_lists_all_open_gaps() -> None:
    text = (REPO_ROOT / "docs/release_checklist.md").read_text(encoding="utf-8")
    missing_section = text.split("## Missing Development Code", 1)[1].split("## Required Commands", 1)[0]

    for gap in RELEASE_GAPS:
        assert gap in missing_section


def test_public_release_check_runs_release_status_tests() -> None:
    text = (REPO_ROOT / "scripts/check_public_release.sh").read_text(encoding="utf-8")

    assert "tests/test_release_status_docs.py" in text

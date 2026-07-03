from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fusion_cache_contract_doc_describes_required_artifacts() -> None:
    text = (REPO_ROOT / "docs/fusion_cache_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Fusion Cache Contract",
        "Cache Filename",
        "{stem}_{width}x{height}_p{patch}.npz",
        "Required NPZ Keys",
        "tokens",
        "Hp",
        "Wp",
        "meta",
        "cache_only",
        "fusion-cache-build",
        "Do Not Guess Builder Logic",
    ):
        assert phrase in text


def test_release_docs_link_fusion_cache_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    release_usage = (REPO_ROOT / "docs/release_usage.md").read_text(encoding="utf-8")
    extension_points = (REPO_ROOT / "docs/release_extension_points.md").read_text(encoding="utf-8")

    assert "docs/fusion_cache_contract.md" in readme
    assert "docs/fusion_cache_contract.md" in release_usage
    assert "docs/fusion_cache_contract.md" in extension_points

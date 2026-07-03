from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_interaction_batch_contract_doc_describes_missing_runner_interface() -> None:
    text = (REPO_ROOT / "docs/interaction_batch_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Interaction Batch Contract",
        "Missing Runner",
        "Inputs",
        "AGD20K",
        "Outputs",
        "verb_heat.png",
        "verb_heat.npy",
        "metrics.csv",
        "kld",
        "sim",
        "nss",
        "Do Not Guess Batch Logic",
    ):
        assert phrase in text


def test_release_docs_link_interaction_batch_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    extension_points = (REPO_ROOT / "docs/release_extension_points.md").read_text(encoding="utf-8")
    public_api = (REPO_ROOT / "docs/public_api.md").read_text(encoding="utf-8")

    assert "docs/interaction_batch_contract.md" in readme
    assert "docs/interaction_batch_contract.md" in extension_points
    assert "pba.interaction.batch" in public_api

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sd21_local_model_contract_doc_describes_online_and_offline_modes() -> None:
    text = (REPO_ROOT / "docs/sd21_local_model_contract.md").read_text(encoding="utf-8")

    for phrase in (
        "Stable Diffusion 2.1 Local Model Contract",
        "Online Mode",
        "stabilityai/stable-diffusion-2-1",
        "Offline Mode",
        "models/stable-diffusion-2-1",
        "Required Subdirectories",
        "unet",
        "scheduler",
        "tokenizer",
        "text_encoder",
        "vae",
        "Do Not Claim Offline Reproduction Yet",
    ):
        assert phrase in text


def test_release_docs_link_sd21_contract() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    extension_points = (REPO_ROOT / "docs/release_extension_points.md").read_text(encoding="utf-8")
    public_api = (REPO_ROOT / "docs/public_api.md").read_text(encoding="utf-8")

    assert "docs/sd21_local_model_contract.md" in readme
    assert "docs/sd21_local_model_contract.md" in extension_points
    assert "pba.geometry.sd21" in public_api

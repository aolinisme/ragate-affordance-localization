from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dinov3_source_strategy_doc_states_current_and_future_policy() -> None:
    text = (REPO_ROOT / "docs/dinov3_source_strategy.md").read_text(encoding="utf-8")

    for phrase in (
        "DINOv3 Source Strategy",
        "Current Release Policy",
        "external-source",
        "https://github.com/facebookresearch/dinov3",
        "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m",
        "models/dinov3/",
        "Do Not Commit",
    ):
        assert phrase in text


def test_models_readme_points_dinov3_users_to_strategy_doc() -> None:
    text = (REPO_ROOT / "models/README.md").read_text(encoding="utf-8")

    assert "dinov3/" in text
    assert "https://github.com/facebookresearch/dinov3" in text
    assert "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m" in text


def test_release_open_questions_do_not_claim_bundled_dinov3_removal_is_approved() -> None:
    text = (REPO_ROOT / "docs/release_open_questions.md").read_text(encoding="utf-8")

    assert "approved release direction is to remove bundled DINOv3 source" not in text
    assert "requires user-provided DINOv3 source" in text

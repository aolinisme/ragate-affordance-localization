from pathlib import Path

from pba.public_release import PublicReleaseFinding
from pba.public_release import check_gitignore_entry
from pba.public_release import check_private_paths
from pba.public_release import check_dinov3_source_policy
from pba.public_release import check_tracked_file_sizes


def test_check_private_paths_reports_tracked_source_paths(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text("dataset_root: /Users/qing/private/data\n", encoding="utf-8")

    findings = check_private_paths(tmp_path, tracked_files=["config.yaml"])

    assert PublicReleaseFinding(
        level="BLOCKED",
        code="private-path",
        path="config.yaml",
        message="private local path pattern '/Users/' found",
    ) in findings


def test_check_private_paths_ignores_documented_public_placeholders(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("cd /path/to/Probing_Bridging_Affordance\n", encoding="utf-8")

    findings = check_private_paths(tmp_path, tracked_files=["README.md"])

    assert findings == []


def test_check_gitignore_entry_reports_missing_pattern(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("outputs/\n", encoding="utf-8")

    findings = check_gitignore_entry(tmp_path, "docs/superpowers/")

    assert findings == [
        PublicReleaseFinding(
            level="BLOCKED",
            code="missing-gitignore-entry",
            path=".gitignore",
            message="docs/superpowers/ is not ignored",
        )
    ]


def test_check_tracked_file_sizes_reports_large_files(tmp_path: Path) -> None:
    large_file = tmp_path / "models/checkpoint.bin"
    large_file.parent.mkdir()
    large_file.write_bytes(b"abcd")

    findings = check_tracked_file_sizes(tmp_path, tracked_files=["models/checkpoint.bin"], max_bytes=3)

    assert findings == [
        PublicReleaseFinding(
            level="BLOCKED",
            code="large-tracked-file",
            path="models/checkpoint.bin",
            message="tracked file is 4 bytes; max public-release size is 3 bytes",
        )
    ]


def test_check_dinov3_source_policy_blocks_bundled_tree_without_license(tmp_path: Path) -> None:
    third_party = tmp_path / "fusion_zero_shot/src/dino/third_party/dinov3"
    third_party.mkdir(parents=True)

    findings = check_dinov3_source_policy(tmp_path)

    assert PublicReleaseFinding(
        level="BLOCKED",
        code="dinov3-license-missing",
        path="fusion_zero_shot/src/dino/third_party/dinov3/LICENSE.md",
        message="bundled DINOv3 source must keep upstream LICENSE.md",
    ) in findings


def test_check_dinov3_source_policy_warns_when_strategy_doc_missing(tmp_path: Path) -> None:
    third_party = tmp_path / "fusion_zero_shot/src/dino/third_party/dinov3"
    third_party.mkdir(parents=True)
    (third_party / "LICENSE.md").write_text("DINOv3 License\n", encoding="utf-8")

    findings = check_dinov3_source_policy(tmp_path)

    assert PublicReleaseFinding(
        level="WARN",
        code="dinov3-strategy-doc-missing",
        path="docs/dinov3_source_strategy.md",
        message="DINOv3 bundled/external source policy is not documented",
    ) in findings

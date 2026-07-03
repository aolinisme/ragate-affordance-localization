from pathlib import Path

from pba.data.geometry import (
    CategorySplitEntry,
    build_split_record,
    parse_category_split,
    train_val_test_split,
)


def test_parse_category_split_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "category_split.txt"
    path.write_text("# comment\n\n1 hammer\n2 knife\n", encoding="utf-8")

    entries = parse_category_split(path)

    assert entries == [
        CategorySplitEntry(split_id=1, tool_name="hammer"),
        CategorySplitEntry(split_id=2, tool_name="knife"),
    ]


def test_build_split_record_uses_umd_relative_paths() -> None:
    record = build_split_record("hammer", "hammer_01_000001")

    assert record == {
        "tool": "hammer",
        "frame_id": "hammer_01_000001",
        "rgb": "tools/hammer/hammer_01_000001_rgb.jpg",
        "depth": "tools/hammer/hammer_01_000001_depth.png",
        "label_mat": "tools/hammer/hammer_01_000001_label.mat",
    }


def test_train_val_test_split_expands_tool_records(tmp_path: Path) -> None:
    for tool in ("hammer", "spoon", "knife"):
        tool_dir = tmp_path / "tools" / tool
        tool_dir.mkdir(parents=True)
        (tool_dir / f"{tool}_01_000001_rgb.jpg").write_bytes(b"fake")

    split = train_val_test_split(
        [
            CategorySplitEntry(split_id=1, tool_name="hammer"),
            CategorySplitEntry(split_id=1, tool_name="spoon"),
            CategorySplitEntry(split_id=2, tool_name="knife"),
        ],
        tmp_path,
        val_ratio=0.49,
        val_seed=7,
    )

    assert len(split["train"]) == 1
    assert len(split["val"]) == 1
    assert split["test"][0]["tool"] == "knife"

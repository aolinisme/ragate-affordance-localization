from pathlib import Path
import subprocess
import sys

from pba.runner import CommandRegistry, load_command_registry


def test_load_command_registry_reads_top_level_run_py(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    run_py = tmp_path / "run.py"
    run_py.write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "COMMANDS = {'geometry-train': ROOT / 'train.py'}\n",
        encoding="utf-8",
    )

    registry = load_command_registry(tmp_path)

    assert isinstance(registry, CommandRegistry)
    assert registry.commands == {"geometry-train": script}


def test_load_command_registry_rejects_missing_run_py(tmp_path: Path) -> None:
    try:
        load_command_registry(tmp_path)
    except FileNotFoundError as error:
        assert "run.py not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_pba_run_list_prints_commands(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "eval.py").write_text("print('eval')\n", encoding="utf-8")
    (tmp_path / "run.py").write_text(
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "COMMANDS = {'geometry-train': ROOT / 'train.py', 'geometry-eval': ROOT / 'eval.py'}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "pba.run", "--repo-root", str(tmp_path), "--list"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.splitlines() == ["geometry-eval", "geometry-train"]


def test_pba_run_reports_unknown_command(tmp_path: Path) -> None:
    (tmp_path / "run.py").write_text("COMMANDS = {}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pba.run", "--repo-root", str(tmp_path), "missing"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "unknown command: missing" in result.stderr

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_api_doc_exists_and_names_package_surface() -> None:
    doc = REPO_ROOT / "docs/public_api.md"

    text = doc.read_text(encoding="utf-8")

    for section in (
        "pba.metrics",
        "pba.data",
        "pba.geometry",
        "pba.fusion",
        "pba.interaction",
    ):
        assert section in text
    assert "Heavy Runtime Boundaries" in text


def test_lightweight_pba_modules_import_without_torch_or_diffusers() -> None:
    imports = [
        "pba.config",
        "pba.runner",
        "pba.reproduce",
        "pba.audit",
        "pba.metrics.affordance",
        "pba.data.geometry",
        "pba.data.umd",
        "pba.data.affordance",
        "pba.geometry.config",
        "pba.geometry.linear_probe",
        "pba.geometry.multiseed",
        "pba.geometry.sd21",
        "pba.geometry.runtime",
        "pba.fusion.config",
        "pba.fusion.kontext",
        "pba.fusion.prompts",
        "pba.fusion.paths",
        "pba.fusion.cache",
        "pba.fusion.runtime",
        "pba.interaction.config",
        "pba.interaction.tokens",
        "pba.interaction.outputs",
        "pba.interaction.batch",
        "pba.interaction.runtime",
    ]

    for module_name in imports:
        __import__(module_name)


def test_legacy_scripts_import_without_heavy_runtime_dependencies() -> None:
    legacy_scripts = {
        "legacy_geometry_train": REPO_ROOT / "geometry_probing/umd_linear_probing/scripts/train.py",
        "legacy_geometry_eval": REPO_ROOT / "geometry_probing/umd_linear_probing/scripts/eval.py",
        "legacy_interaction_probe": REPO_ROOT / "interaction_probing/cross_attention_probe/cross_attention_probe.py",
    }

    for name, path in legacy_scripts.items():
        load_module(path, name)


def test_package_runner_lists_release_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pba.run", "--list"],
        check=True,
        text=True,
        capture_output=True,
    )

    commands = set(result.stdout.splitlines())
    assert {
        "geometry-train",
        "geometry-eval",
        "geometry-multiseed",
        "interaction-probe",
        "fusion-eval",
    }.issubset(commands)

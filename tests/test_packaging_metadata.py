from pathlib import Path

import tomli


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_pba_package_and_cli_scripts() -> None:
    metadata = tomli.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "probing-bridging-affordance"
    assert metadata["project"]["requires-python"] == ">=3.9"
    assert "pip" in metadata["build-system"]["requires"]
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["pba*"]
    assert metadata["project"]["scripts"] == {
        "pba-run": "pba.run:main",
        "pba-audit": "pba.audit:main",
        "pba-reproduce": "pba.cli.reproduce:main",
    }


def test_setup_py_shim_keeps_legacy_editable_install_working() -> None:
    text = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "from setuptools import setup" in text
    assert "setup()" in text


def test_smoke_requirements_cover_lightweight_release_tests() -> None:
    requirements = (REPO_ROOT / "requirements-smoke.txt").read_text(encoding="utf-8").splitlines()
    normalized = {line.strip().split(">=")[0] for line in requirements if line.strip() and not line.startswith("#")}

    assert {"pytest", "pyyaml", "numpy", "pillow", "matplotlib", "tomli"}.issubset(normalized)
    assert "torch" not in normalized
    assert "diffusers" not in normalized


def test_packaging_build_outputs_are_gitignored() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "*.egg-info/" in text
    assert "build/" in text
    assert "dist/" in text


def test_installation_docs_describe_smoke_and_full_dependency_paths() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    release_usage = (REPO_ROOT / "docs/release_usage.md").read_text(encoding="utf-8")

    assert "pip install -e ." in readme
    assert "pip install -r requirements-smoke.txt" in readme
    assert "pip install -r requirements.txt" in readme
    assert "requirements-smoke.txt" in release_usage
    assert "pba-audit" in release_usage

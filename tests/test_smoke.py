"""Smoke test — vérifie la structure de base du projet et que pytest tourne."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_layout():
    assert (PROJECT_ROOT / "README.md").is_file()
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "data").is_dir()


def test_src_is_a_python_package():
    assert (PROJECT_ROOT / "src" / "__init__.py").is_file()

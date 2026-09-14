import importlib.util
from pathlib import Path

import pytest


def _load_model_comparison_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "model_comparison.py"
    spec = importlib.util.spec_from_file_location("model_comparison_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_safe_repo_relative_path_accepts_relative_and_absolute_repo_paths(tmp_path: Path) -> None:
    module = _load_model_comparison_module()
    repo_root = tmp_path / "repo"
    artifact_dir = repo_root / "artifacts" / "models" / "official_model"
    artifact_dir.mkdir(parents=True)

    assert module.safe_repo_relative_path(
        Path("artifacts/models/official_model"),
        repo_root=repo_root,
    ) == "artifacts/models/official_model"
    assert module.safe_repo_relative_path(
        artifact_dir,
        repo_root=repo_root,
    ) == "artifacts/models/official_model"


def test_safe_repo_relative_path_rejects_outside_paths(tmp_path: Path) -> None:
    module = _load_model_comparison_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="inside the repository"):
        module.safe_repo_relative_path(tmp_path / "outside", repo_root=repo_root)

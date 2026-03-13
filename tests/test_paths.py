from quickdraw_cnn.config import load_config
from quickdraw_cnn.paths import build_paths


def test_build_paths_has_expected_project_folder() -> None:
    cfg = load_config()
    paths = build_paths(cfg)
    assert paths.project_dir.name == "quickdraw-cnn"


def test_model_path_ends_with_model_keras() -> None:
    cfg = load_config()
    paths = build_paths(cfg)
    assert paths.final_model_path.name == "model.keras"
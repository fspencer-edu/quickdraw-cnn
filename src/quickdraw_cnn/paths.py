from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quickdraw_cnn.config import AppConfig


@dataclass
class ProjectPaths:
    nas_root: Path
    dataset_root: Path
    dataset_dir: Path
    project_dir: Path
    checkpoint_dir: Path
    experiment_dir: Path
    log_dir: Path
    export_dir: Path
    model_root: Path
    model_version_dir: Path
    staging_dir: Path
    production_dir: Path
    class_names_path: Path
    checkpoint_path: Path
    final_model_path: Path
    metrics_path: Path
    history_path: Path
    evaluation_path: Path


def build_paths(cfg: AppConfig) -> ProjectPaths:
    nas_root = Path(cfg.paths.nas_root)
    dataset_root = Path(cfg.paths.dataset_root)
    dataset_dir = dataset_root

    project_dir = nas_root / "projects" / cfg.project_name
    checkpoint_dir = project_dir / "checkpoints"
    experiment_dir = project_dir / "experiments"
    log_dir = project_dir / "logs"
    export_dir = project_dir / "exports"

    model_root = nas_root / "models" / cfg.project_name
    model_version_dir = model_root / cfg.versioning.model_version

    staging_dir = nas_root / "registry" / "staging" / cfg.project_name
    production_dir = nas_root / "registry" / "production" / cfg.project_name

    return ProjectPaths(
        nas_root=nas_root,
        dataset_root=dataset_root,
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        checkpoint_dir=checkpoint_dir,
        experiment_dir=experiment_dir,
        log_dir=log_dir,
        export_dir=export_dir,
        model_root=model_root,
        model_version_dir=model_version_dir,
        staging_dir=staging_dir,
        production_dir=production_dir,
        class_names_path=experiment_dir / "class_names.json",
        checkpoint_path=checkpoint_dir / "best_model.keras",
        final_model_path=model_version_dir / "model.keras",
        metrics_path=experiment_dir / "metrics.json",
        history_path=experiment_dir / "history.json",
        evaluation_path=experiment_dir / "evaluation.json",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    dirs = [
        paths.nas_root,
        paths.dataset_root,
        paths.project_dir,
        paths.checkpoint_dir,
        paths.experiment_dir,
        paths.log_dir,
        paths.export_dir,
        paths.model_root,
        paths.model_version_dir,
        paths.staging_dir,
        paths.production_dir,
    ]

    for path in dirs:
        print(f"[PATHS] ensuring: {path}", flush=True)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[PATHS] done: {path}", flush=True)
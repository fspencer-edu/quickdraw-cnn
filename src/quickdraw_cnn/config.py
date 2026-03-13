from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    nas_root: str
    dataset_root: str | None = None


@dataclass
class ModelConfig:
    image_size: int
    channels: int
    num_filters: list[int]
    dense_units: int
    dropout: float


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    random_seed: int
    validation_fraction: float
    test_fraction: float = 0.10
    shuffle_buffer: int = 20000
    early_stopping_patience: int = 3


@dataclass
class RuntimeConfig:
    use_mixed_precision: bool


@dataclass
class VersioningConfig:
    model_version: str


@dataclass
class AppConfig:
    project_name: str
    dataset_name: str
    tfds_name: str
    paths: PathsConfig
    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeConfig
    versioning: VersioningConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")
    return data


def load_config(config_path: str | Path = "configs/base.yaml") -> AppConfig:
    path = Path(config_path)
    data = _load_yaml(path)

    return AppConfig(
        project_name=data["project_name"],
        dataset_name=data["dataset_name"],
        tfds_name=data["tfds_name"],
        paths=PathsConfig(**data["paths"]),
        model=ModelConfig(**data["model"]),
        train=TrainConfig(**data["train"]),
        runtime=RuntimeConfig(**data["runtime"]),
        versioning=VersioningConfig(**data["versioning"]),
    )
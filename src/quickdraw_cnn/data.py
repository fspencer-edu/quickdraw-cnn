from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds

from quickdraw_cnn.config import AppConfig
from quickdraw_cnn.paths import ProjectPaths
from quickdraw_cnn.utils import save_json


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    num_classes: int
    class_names: list[str]
    train_examples: int
    test_examples: int


def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)
    return image, label


def load_datasets(cfg: AppConfig, paths: ProjectPaths) -> DatasetBundle:
    builder = tfds.builder(cfg.tfds_name, data_dir=str(paths.dataset_dir))
    builder.download_and_prepare()

    info = builder.info
    num_classes = info.features["label"].num_classes
    class_names = info.features["label"].names

    save_json({"class_names": class_names}, paths.class_names_path)

    train_ds = tfds.load(
        cfg.tfds_name,
        split="train",
        data_dir=str(paths.dataset_dir),
        as_supervised=True,
    )

    test_ds = tfds.load(
        cfg.tfds_name,
        split="test",
        data_dir=str(paths.dataset_dir),
        as_supervised=True,
    )

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_examples = info.splits["train"].num_examples
    test_examples = info.splits["test"].num_examples
    val_size = int(train_examples * cfg.train.validation_fraction)

    val_ds = train_ds.take(val_size)
    train_ds = train_ds.skip(val_size)

    train_ds = (
        train_ds
        .shuffle(cfg.train.shuffle_buffer, seed=cfg.train.random_seed)
        .batch(cfg.train.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = val_ds.batch(cfg.train.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(cfg.train.batch_size).prefetch(tf.data.AUTOTUNE)

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=num_classes,
        class_names=class_names,
        train_examples=train_examples,
        test_examples=test_examples,
    )
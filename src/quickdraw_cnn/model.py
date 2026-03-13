from __future__ import annotations

import tensorflow as tf

from quickdraw_cnn.config import AppConfig


def build_model(cfg: AppConfig, num_classes: int) -> tf.keras.Model:
    image_size = cfg.model.image_size
    channels = cfg.model.channels
    filters = cfg.model.num_filters

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size, image_size, channels)),
        tf.keras.layers.Conv2D(filters[0], 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters[1], 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters[2], 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(cfg.model.dense_units, activation="relu"),
        tf.keras.layers.Dropout(cfg.model.dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def compile_model(model: tf.keras.Model, cfg: AppConfig) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
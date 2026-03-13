from __future__ import annotations

import tensorflow as tf

from quickdraw_cnn.config import load_config
from quickdraw_cnn.data import load_datasets
from quickdraw_cnn.model import build_model, compile_model
from quickdraw_cnn.paths import build_paths, ensure_dirs
from quickdraw_cnn.utils import save_json, set_global_seed


def main() -> None:
    cfg = load_config()
    paths = build_paths(cfg)
    ensure_dirs(paths)

    set_global_seed(cfg.train.random_seed)

    if cfg.runtime.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    dataset_bundle = load_datasets(cfg, paths)

    model = build_model(cfg, dataset_bundle.num_classes)
    model = compile_model(model, cfg)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        dataset_bundle.train_ds,
        validation_data=dataset_bundle.val_ds,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(dataset_bundle.test_ds, verbose=1)
    model.save(paths.final_model_path)

    save_json(history.history, paths.history_path)
    save_json(
        {
            "project_name": cfg.project_name,
            "dataset_name": cfg.dataset_name,
            "model_version": cfg.versioning.model_version,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "epochs_ran": len(history.history["loss"]),
            "train_examples": dataset_bundle.train_examples,
            "test_examples": dataset_bundle.test_examples,
        },
        paths.metrics_path,
    )

    print(f"Saved final model to: {paths.final_model_path}")
    print(f"Saved checkpoint to: {paths.checkpoint_path}")
    print(f"Saved metrics to: {paths.metrics_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import tensorflow as tf

from quickdraw_cnn.config import load_config
from quickdraw_cnn.data import load_datasets
from quickdraw_cnn.model import build_model, compile_model
from quickdraw_cnn.paths import build_paths, ensure_dirs
from quickdraw_cnn.utils import save_json, set_global_seed


def main() -> None:
    print("[TRAIN] Loading config...", flush=True)
    cfg = load_config()
    print("[TRAIN] Config loaded.", flush=True)

    print("[TRAIN] Building paths...", flush=True)
    paths = build_paths(cfg)
    print(f"[TRAIN] dataset_dir = {paths.dataset_dir}", flush=True)

    print("[TRAIN] Ensuring directories...", flush=True)
    ensure_dirs(paths)
    print("[TRAIN] Directories ready.", flush=True)

    print("[TRAIN] Setting seed...", flush=True)
    set_global_seed(cfg.train.random_seed)
    print("[TRAIN] Seed set.", flush=True)

    if cfg.runtime.use_mixed_precision:
        print("[TRAIN] Enabling mixed precision...", flush=True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[TRAIN] Mixed precision enabled.", flush=True)

    print("[TRAIN] Loading datasets...", flush=True)
    dataset_bundle = load_datasets(cfg, paths)
    print("[TRAIN] Datasets loaded.", flush=True)

    print("[TRAIN] Building model...", flush=True)
    model = build_model(cfg, dataset_bundle.num_classes)
    print("[TRAIN] Model built.", flush=True)

    print("[TRAIN] Compiling model...", flush=True)
    model = compile_model(model, cfg)
    print("[TRAIN] Model compiled.", flush=True)

    print("[TRAIN] Creating callbacks...", flush=True)
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
    print("[TRAIN] Callbacks ready.", flush=True)

    print("[TRAIN] Starting training...", flush=True)
    history = model.fit(
        dataset_bundle.train_ds,
        validation_data=dataset_bundle.val_ds,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("[TRAIN] Training finished.", flush=True)

    print("[TRAIN] Evaluating model...", flush=True)
    test_loss, test_acc = model.evaluate(dataset_bundle.test_ds, verbose=1)

    print("[TRAIN] Saving model...", flush=True)
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
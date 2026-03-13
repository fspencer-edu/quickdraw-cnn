from __future__ import annotations

import tensorflow as tf

from quickdraw_cnn.config import load_config
from quickdraw_cnn.data import load_datasets
from quickdraw_cnn.paths import build_paths, ensure_dirs
from quickdraw_cnn.utils import save_json


def main() -> None:
    cfg = load_config()
    paths = build_paths(cfg)
    ensure_dirs(paths)

    dataset_bundle = load_datasets(cfg, paths)
    model = tf.keras.models.load_model(paths.final_model_path)

    test_loss, test_acc = model.evaluate(dataset_bundle.test_ds, verbose=1)

    save_json(
        {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        },
        paths.evaluation_path,
    )

    print(f"Evaluation saved to: {paths.evaluation_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
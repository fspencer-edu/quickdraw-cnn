from __future__ import annotations

import numpy as np
import tensorflow as tf

from quickdraw_cnn.config import load_config
from quickdraw_cnn.paths import build_paths
from quickdraw_cnn.utils import load_json


def main() -> None:
    cfg = load_config()
    paths = build_paths(cfg)

    model = tf.keras.models.load_model(paths.final_model_path)
    class_name_data = load_json(paths.class_names_path)
    class_names = class_name_data["class_names"]

    sample = np.random.randint(
        0,
        256,
        size=(cfg.model.image_size, cfg.model.image_size),
        dtype=np.uint8,
    )

    x = sample.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)[0]
    label_idx = int(np.argmax(pred))
    label_name = class_names[label_idx]
    confidence = float(pred[label_idx])

    print(f"Predicted class: {label_name}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
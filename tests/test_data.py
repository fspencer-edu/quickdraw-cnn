import tensorflow as tf

from quickdraw_cnn.data import preprocess


def test_preprocess_adds_channel_dimension() -> None:
    image = tf.zeros((28, 28), dtype=tf.uint8)
    label = tf.constant(3, dtype=tf.int64)

    processed_image, processed_label = preprocess(image, label)

    assert processed_image.shape == (28, 28, 1)
    assert int(processed_label.numpy()) == 3
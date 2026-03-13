from quickdraw_cnn.config import load_config
from quickdraw_cnn.model import build_model, compile_model


def test_model_output_shape() -> None:
    cfg = load_config()
    model = build_model(cfg, num_classes=10)
    model = compile_model(model, cfg)
    assert model.output_shape[-1] == 10
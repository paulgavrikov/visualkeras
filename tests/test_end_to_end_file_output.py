import pytest
from PIL import Image

import visualkeras
from visualkeras.options import FunctionalOptions, GraphOptions, LayeredOptions, LenetOptions


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


def _build_dense_model():
    inputs = tf.keras.layers.Input(shape=(8,), name="input")
    x = tf.keras.layers.Dense(6, activation="relu", name="dense1")(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="dense2")(x)
    model = tf.keras.Model(inputs, outputs, name="dense_model")
    model.build((None, 8))
    return model


def _build_conv_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(16, 16, 3), name="input"),
            tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv1"),
            tf.keras.layers.MaxPool2D(2, name="pool1"),
            tf.keras.layers.Flatten(name="flat"),
            tf.keras.layers.Dense(4, activation="softmax", name="out"),
        ],
        name="conv_model",
    )
    model.build((None, 16, 16, 3))
    return model


@pytest.mark.parametrize(
    "mode, options_factory, model_builder",
    [
        ("layered", lambda p: LayeredOptions(to_file=str(p), legend_text_spacing_offset=0), _build_dense_model),
        ("graph", lambda p: GraphOptions(to_file=str(p), show_neurons=False), _build_dense_model),
        ("functional", lambda p: FunctionalOptions(to_file=str(p), draw_volume=False), _build_dense_model),
        ("lenet", lambda p: LenetOptions(to_file=str(p), draw_connections=False), _build_conv_model),
    ],
)
def test_show_end_to_end_writes_file(tmp_path, mode, options_factory, model_builder):
    model = model_builder()
    out_file = tmp_path / f"{mode}.png"
    options = options_factory(out_file)

    image = visualkeras.show(model, mode=mode, options=options)

    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0

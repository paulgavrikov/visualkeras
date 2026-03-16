import pytest
from PIL import Image

import visualkeras
from visualkeras.options import FunctionalOptions


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


def _build_reference_model():
    inputs = tf.keras.layers.Input(shape=(16,), name="input")
    x1 = tf.keras.layers.Dense(12, activation="relu", name="dense_a")(inputs)
    x2 = tf.keras.layers.Dense(12, activation="relu", name="dense_b")(inputs)
    merged = tf.keras.layers.Add(name="add")([x1, x2])
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="output")(merged)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_reference")
    model.build((None, 16))
    return model


def test_functional_view_smoke():
    model = _build_reference_model()
    image = visualkeras.functional_view(
        model,
        draw_volume=False,
        simple_text_visualization=True,
        simple_text_label_mode="below",
    )
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0


def test_functional_show_mode_with_options():
    model = _build_reference_model()
    options = FunctionalOptions(
        draw_volume=False,
        connector_arrow=True,
        simple_text_visualization=True,
        component_spacing=40,
    )
    image = visualkeras.show(
        model,
        mode="functional",
        preset="compact",
        options=options,
    )
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0

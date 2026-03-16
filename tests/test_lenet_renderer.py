import pytest
from PIL import Image

import visualkeras
from visualkeras.options import LenetOptions


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


def _build_reference_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(16, 16, 3), name="input"),
            tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same", name="conv1"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool1"),
            tf.keras.layers.Conv2D(12, (3, 3), activation="relu", padding="same", name="conv2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(16, activation="relu", name="dense"),
            tf.keras.layers.Dense(4, activation="softmax", name="output"),
        ],
        name="lenet_reference",
    )
    model.build((None, 16, 16, 3))
    return model


def test_lenet_view_smoke():
    model = _build_reference_model()
    image = visualkeras.lenet_view(
        model,
        max_visual_channels=8,
        draw_connections=True,
        draw_patches=True,
    )
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0


def test_lenet_show_mode_with_options():
    model = _build_reference_model()
    options = LenetOptions(
        max_visual_channels=10,
        draw_connections=False,
        layer_spacing=35,
    )
    image = visualkeras.show(
        model,
        mode="lenet",
        preset="presentation",
        options=options,
    )
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0

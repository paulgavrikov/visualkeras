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


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
def test_lenet_1d_convolution_and_pooling_use_spatial_maps(data_format):
    import visualkeras.lenet as lenet

    shape = (3, 12) if data_format == "channels_first" else (12, 3)
    inputs = tf.keras.Input(shape, name="signal")
    convolution = tf.keras.layers.Conv1D(5, 3, data_format=data_format, name="conv_1d")
    pool = tf.keras.layers.MaxPooling1D(2, data_format=data_format, name="pool_1d")
    outputs = pool(convolution(inputs))
    model = tf.keras.Model(inputs, outputs)
    conv_shape = lenet._canonicalize_shape(convolution, convolution.output.shape)
    pool_shape = lenet._canonicalize_shape(pool, pool.output.shape)
    assert (conv_shape.w, conv_shape.c) == (10, 5)
    assert (pool_shape.kind, pool_shape.w, pool_shape.c) == ("spatial", 5, 5)
    image = visualkeras.lenet_view(model, seed=7)
    assert image.width > 0 and image.height > 0


def test_lenet_contained_face_image_blends_once(tmp_path):
    icon = tmp_path / "face.png"
    Image.new("RGBA", (8, 4), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8, 8, 1), name="face_input")
    outputs = tf.keras.layers.Conv2D(1, 1, name="face_conv")(inputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.lenet_view(
        model, background_fill="white", draw_connections=False, draw_patches=False,
        styles={"face_conv": {"face_image": str(icon), "face_image_fit": "contain", "fill": "white"}},
    )
    assert (255, 127, 127, 255) in image.getdata()


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


def test_lenet_fixed_batch_scalar_has_one_output_unit():
    import visualkeras.lenet as lenet

    inputs = tf.keras.Input((3,), batch_size=4, name="scalar_input")
    scalar = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="scalar")
    model = tf.keras.Model(inputs, scalar(inputs))
    shape = lenet._canonicalize_shape(scalar, scalar.output.shape)
    assert shape.c == 1
    image = visualkeras.lenet_view(model, seed=7)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("text", ["A\ng", "A\n\ng", "jazz"])
@pytest.mark.parametrize("above", [False, True])
def test_lenet_captions_keep_full_glyphs_outside_the_maps(text, above):
    from PIL import ImageDraw, ImageFont

    model = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(3, name="out")])
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    caption = lambda layer, shape: text if layer.name == "out" else ""
    image = visualkeras.lenet_view(
        model, padding=0, font=font, font_color="magenta", background_fill="white",
        top_label=above, bottom_label=not above, top_label_callable=caption,
        bottom_label_callable=caption, draw_connections=False, draw_patches=False,
        styles={object: {"fill": "blue", "outline": "blue", "shade_step": 0}},
    )
    reference = Image.new("RGBA", (200, 200), "white")
    ImageDraw.Draw(reference).multiline_text((10, 10), text, font=font, fill="magenta", spacing=2, align="center")
    rows = [i // image.width for i, (r, g, b, a) in enumerate(image.getdata()) if r == b == 255 and g < 255]
    expected_rows = [i // reference.width for i, (r, g, b, a) in enumerate(reference.getdata()) if r == b == 255 and g < 255]
    map_rows = [i // image.width for i, (r, g, b, a) in enumerate(image.getdata()) if b == 255 and r == g < 255]
    assert max(rows) - min(rows) == max(expected_rows) - min(expected_rows)
    assert max(rows) < min(map_rows) if above else min(rows) > max(map_rows)


@pytest.mark.parametrize("face_alpha", [128, 255])
def test_lenet_translucent_face_uses_one_background_blend(tmp_path, face_alpha):
    from collections import Counter
    icon = tmp_path / "face.png"
    Image.new("RGBA", (8, 4), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8, 8, 1), name="in")
    outputs = tf.keras.layers.Conv2D(1, 1, name="face")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {
        "background_fill": (0, 0, 0, 0), "draw_connections": False, "draw_patches": False,
        "top_label": False, "bottom_label": False,
    }
    reference = visualkeras.lenet_view(model, **settings, styles={object: {"fill": (0, 0, 255, 128)}})
    face_pixels = Counter(pixel for pixel in reference.getdata() if pixel[2] > 0 and pixel[0] == pixel[1] == 0)
    existing_fill = face_pixels.most_common(1)[0][0]
    expected = Image.alpha_composite(
        Image.new("RGBA", (1, 1), existing_fill),
        Image.new("RGBA", (1, 1), (255, 0, 0, int(128 * face_alpha / 255))),
    ).getpixel((0, 0))
    image = visualkeras.lenet_view(
        model, **settings, styles={object: {"fill": (0, 0, 255, 128)},
                                  "face": {"face_image": str(icon), "face_image_fit": "contain", "face_image_alpha": face_alpha}},
    )
    assert expected in image.getdata()

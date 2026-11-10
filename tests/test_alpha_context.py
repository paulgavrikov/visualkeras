from collections import Counter

import aggdraw
import pytest
from PIL import Image

from visualkeras import utils


@pytest.mark.parametrize("mode", ["lenet", "layered", "graph", "functional"])
@pytest.mark.parametrize("background", [(0, 0, 0, 0), (37, 53, 97, 0)])
def test_public_translucent_fills_keep_straight_alpha_colors(mode, background):
    tf = pytest.importorskip("tensorflow")
    import visualkeras

    inputs = tf.keras.Input((8, 8, 1), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Conv2D(1, 1, name="out")(inputs))
    settings = {"draw_connections": False, "draw_patches": False, "top_label": False, "bottom_label": False}
    if mode == "layered":
        settings = {"draw_volume": False, "draw_funnel": False}
    elif mode == "functional":
        settings = {"draw_volume": False, "connector_fill": (0, 0, 0, 0)}
    elif mode == "graph":
        settings = {"show_neurons": False, "connector_fill": (0, 0, 0, 0)}
    image = visualkeras.show(
        model, mode=mode, padding=20, background_fill=background,
        styles={object: {"fill": (0, 0, 255, 128), "outline": (0, 0, 255, 128)}}, **settings,
    )
    pixels = Counter(pixel for pixel in image.getdata() if pixel[3] > 0 and pixel[2] > pixel[0] + pixel[1])
    assert pixels
    interior = pixels.most_common(1)[0][0]
    assert interior[:3] == (0, 0, 255)
    assert 0 < interior[3] < 255
    assert image.getpixel((0, 0)) == background


def test_alpha_context_preserves_untouched_pixels_and_repeated_flushes():
    image = Image.new("RGBA", (40, 32), (37, 53, 97, 0))
    draw = utils._aggdraw_context(image)
    draw.rectangle((8, 8, 24, 24), None, aggdraw.Brush((0, 0, 255, 128)))
    draw.flush()
    assert image.getpixel((16, 16))[:3] == (0, 0, 255)
    assert image.getpixel((0, 0)) == (37, 53, 97, 0)
    first = image.tobytes()
    draw.flush()
    assert image.tobytes() == first
    draw.rectangle((16, 8, 32, 24), None, aggdraw.Brush((255, 0, 0, 128)))
    draw.flush()
    overlap = image.getpixel((20, 16))
    assert overlap[0] > 0 and overlap[2] > 0 and sum(overlap[:3]) >= 250
    assert image.getpixel((0, 0)) == (37, 53, 97, 0)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_opaque_alpha_context_matches_native_aggdraw_exactly(mode):
    image = Image.new(mode, (40, 32), "white")
    reference = image.copy()
    for draw in [utils._aggdraw_context(image), aggdraw.Draw(reference)]:
        draw.ellipse((4, 3, 29, 26), aggdraw.Pen((10, 20, 30, 120), 2), aggdraw.Brush((210, 80, 35, 128)))
        draw.line((2, 25, 35, 4), aggdraw.Pen((100, 210, 40, 90), 3))
        draw.flush()
    assert image.tobytes() == reference.tobytes()


def test_layered_translucent_legend_keeps_straight_alpha_colors():
    tf = pytest.importorskip("tensorflow")
    import visualkeras

    inputs = tf.keras.Input((4,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, name="out")(inputs))
    settings = {"background_fill": (0, 0, 0, 0), "draw_volume": False, "draw_funnel": False,
                "styles": {object: {"fill": (0, 0, 255, 128), "outline": (0, 0, 255, 128)}}}
    base = visualkeras.layered_view(model, **settings)
    image = visualkeras.layered_view(model, legend=True, **settings)
    legend = image.crop((0, base.height, image.width, image.height))
    blue = [pixel for pixel in legend.getdata() if pixel[0] == pixel[1] == 0 and pixel[2] > 0 and pixel[3] > 0]
    assert blue
    assert Counter(blue).most_common(1)[0][0][:3] == (0, 0, 255)


def test_alpha_context_blends_over_partially_transparent_destination():
    image = Image.new("RGBA", (40, 32), (0, 128, 0, 128))
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    native = aggdraw.Draw(mask)
    native.rectangle((8, 8, 24, 24), None, aggdraw.Brush((255, 255, 255, 128)))
    native.flush()
    foreground = Image.new("RGBA", image.size, "blue")
    foreground.putalpha(mask.getchannel("A"))
    expected = Image.alpha_composite(image, foreground)
    draw = utils._aggdraw_context(image)
    draw.rectangle((8, 8, 24, 24), None, aggdraw.Brush((0, 0, 255, 128)))
    draw.flush()
    # An 8-bit premultiplied buffer can round a color channel by one unit.
    assert all(abs(a - b) <= 1 for a, b in zip(image.getpixel((16, 16)), expected.getpixel((16, 16))))
    assert image.getpixel((0, 0)) == (0, 128, 0, 128)

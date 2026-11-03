import pytest
from PIL import ImageDraw, ImageFont

import visualkeras


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("above", [False, True])
@pytest.mark.parametrize("reversed_view", [False, True])
@pytest.mark.parametrize("volume", [False, True])
def test_layered_external_captions_have_clear_horizontal_space(monkeypatch, above, reversed_view, volume):
    inputs = tf.keras.Input((4,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, name="out")(inputs))
    text = "Wide label\nshape"
    rectangles, calls = [], []
    original = ImageDraw.ImageDraw.multiline_text

    def record(draw, xy, value, **kwargs):
        if value == text:
            rectangles.append(draw.multiline_textbbox(xy, value, **{
                key: kwargs[key] for key in ["font", "spacing", "align", "anchor"] if key in kwargs
            }))
        return original(draw, xy, value, **kwargs)

    def caption(index, layer):
        calls.append((index, layer.name))
        return text, above

    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record)
    image = visualkeras.layered_view(
        model, spacing=0, padding=0, draw_volume=volume, draw_reversed=reversed_view,
        font=ImageFont.truetype("DejaVuSans.ttf", 24), font_color="magenta", text_callable=caption,
    )
    assert calls == [(0, "in"), (1, "out")]
    assert len(rectangles) == 2
    left, right = sorted(rectangles)
    assert left[2] + 4 <= right[0]
    for x1, y1, x2, y2 in rectangles:
        assert 0 <= x1 < x2 <= image.width
        assert 0 <= y1 < y2 <= image.height

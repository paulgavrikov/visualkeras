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


@pytest.mark.parametrize("mode", ["layered"])
@pytest.mark.parametrize("above", [False, True])
@pytest.mark.parametrize("group_padding", [0, 10, 60])
def test_group_captions_clear_external_node_captions(monkeypatch, mode, above, group_padding):
    inputs = tf.keras.Input((4,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, name="out")(inputs))
    texts = {"Node\nshape": [], "Group\ncaption": []}
    original = ImageDraw.ImageDraw.multiline_text

    def record(draw, xy, text, **kwargs):
        if text in texts:
            texts[text].append(draw.multiline_textbbox(xy, text, **{
                key: kwargs[key] for key in ["font", "spacing", "align", "anchor"] if key in kwargs
            }))
        return original(draw, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record)
    image = visualkeras.show(
        model, mode=mode, padding=0, draw_volume=False,
        font=ImageFont.truetype("DejaVuSans.ttf", 12), font_color="magenta",
        text_callable=lambda i, layer: ("Node\nshape", above),
        layered_groups=[{"name": "Group\ncaption", "layers": ["out"], "padding": group_padding,
                         "font": ImageFont.truetype("DejaVuSans.ttf", 26), "font_color": "blue"}],
    )
    assert len(texts["Node\nshape"]) == 2
    assert len(texts["Group\ncaption"]) == 1
    group = texts["Group\ncaption"][0]
    for caption in texts["Node\nshape"]:
        assert (group[2] <= caption[0] or group[0] >= caption[2]
                or group[3] <= caption[1] or group[1] >= caption[3])
    for bounds in texts.values():
        for left, top, right, bottom in bounds:
            assert 0 <= left < right <= image.width
            assert 0 <= top < bottom <= image.height

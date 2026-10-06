import pytest
from PIL import ImageDraw, ImageFont

import visualkeras


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["layered", "functional"])
@pytest.mark.parametrize("font_size", [None, 12])
def test_group_font_does_not_replace_diagram_caption_font(monkeypatch, mode, font_size):
    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(4, name="out")(inputs))
    font = ImageFont.truetype("DejaVuSans.ttf", font_size) if font_size else None
    expected_font = font or ImageFont.load_default()
    group_font = ImageFont.truetype("DejaVuSans.ttf", 26)
    text = "Node\nshape"
    used_fonts = []
    original = ImageDraw.ImageDraw.multiline_text

    def record_text(draw, xy, value, **kwargs):
        if value == text:
            used_fonts.append(kwargs["font"])
        return original(draw, xy, value, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record_text)
    image = visualkeras.show(
        model, mode=mode, draw_volume=False, font=font, font_color="magenta",
        text_callable=lambda i, layer: (text, False),
        layered_groups=[{"name": "Group", "layers": ["out"], "font": group_font, "font_color": "blue"}],
    )
    assert image.width > 0 and image.height > 0
    assert len(used_fonts) == 2
    assert all(used.getbbox(text) == expected_font.getbbox(text) for used in used_fonts)

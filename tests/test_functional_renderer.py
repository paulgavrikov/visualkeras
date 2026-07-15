import pytest
from PIL import Image

import visualkeras
import visualkeras.functional as functional
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


@pytest.mark.parametrize("via_show", [False, True])
def test_functional_captions_remain_under_their_own_nodes(monkeypatch, via_show):
    from PIL import ImageDraw

    model = _build_reference_model()
    rectangles = {}
    captions = {}
    original_render = functional._render_graph
    original_text = ImageDraw.ImageDraw.multiline_text

    def record_render(graph, **kwargs):
        for node in graph.nodes.values():
            rectangles[node.name] = (node.x, node.x + node.width)
        return original_render(graph, **kwargs)

    def record_text(draw, xy, text, **kwargs):
        name = text.split("\n")[0]
        bbox = draw.multiline_textbbox(xy, text, font=kwargs.get("font"), spacing=kwargs.get("spacing", 4))
        captions[name] = (bbox[0] + bbox[2]) / 2
        return original_text(draw, xy, text, **kwargs)

    monkeypatch.setattr(functional, "_render_graph", record_render)
    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record_text)
    settings = {"simple_text_visualization": True, "simple_text_label_mode": "below"}
    image = visualkeras.show(model, mode="functional", **settings) if via_show else visualkeras.functional_view(model, **settings)
    for name in ["input", "add", "output"]:
        left, right = rectangles[name]
        assert left - 1 <= captions[name] <= right + 1
    assert image.width > 0 and image.height > 0


def test_functional_box_labels_preserve_requested_line_breaks():
    inputs = tf.keras.Input((4,), name="line_input")
    outputs = tf.keras.layers.Dense(3, name="line_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {
        "simple_text_visualization": True, "simple_text_label_mode": "inside",
        "styles": {"line_input": {"box_size": (80, 80)}, "line_output": {
            "box_size": (140, 80), "box_text": "First\nSecond",
            "box_text_rotation": 0, "box_text_padding": 0, "box_text_autoshrink": False,
        }},
    }
    image = visualkeras.functional_view(model, **settings)
    settings["styles"]["line_output"]["box_text_wrap"] = "none"
    reference = visualkeras.functional_view(model, **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_functional_supplied_box_font_autoshrinks_like_font_path():
    from PIL import ImageFont

    inputs = tf.keras.Input((4,), name="font_input")
    outputs = tf.keras.layers.Dense(3, name="font_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    style = {"box_size": (100, 50), "box_text": "Classify", "box_text_rotation": 0,
             "box_text_wrap": "none", "box_text_font_size": 26, "box_text_font": font}
    image = visualkeras.functional_view(
        model, simple_text_visualization=True, simple_text_label_mode="inside",
        styles={"font_input": {"box_size": (80, 80)}, "font_output": style},
    )
    reference = visualkeras.functional_view(
        model, simple_text_visualization=True, simple_text_label_mode="inside",
        styles={"font_input": {"box_size": (80, 80)},
                "font_output": {**style, "box_text_font": "DejaVuSans.ttf"}},
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()

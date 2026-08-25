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


@pytest.mark.parametrize("via_options", [False, True])
@pytest.mark.parametrize("root_style", [None, {"fill": "cyan"}])
def test_functional_rendering_preserves_reusable_style_maps(via_options, root_style):
    from copy import deepcopy

    inputs = tf.keras.Input((4,), name="in")
    outputs = tf.keras.layers.Dense(3, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    styles = {"in": {"box_size": (80, 80)}, "out": {"box_size": (80, 80)}}
    if root_style is not None:
        styles[object] = deepcopy(root_style)
    original = deepcopy(styles)
    kwargs = {"options": FunctionalOptions(styles=styles)} if via_options else {"styles": styles}
    below = visualkeras.functional_view(model, simple_text_visualization=True, **kwargs)
    assert styles == original
    image = visualkeras.functional_view(
        model, simple_text_visualization=True, simple_text_label_mode="inside", **kwargs,
    )
    reference = visualkeras.functional_view(
        model, simple_text_visualization=True, simple_text_label_mode="inside", styles=original,
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_functional_output_markers_show_each_tensor_shape():
    inputs = tf.keras.Input((8,), name="in")
    outputs = tf.keras.layers.Lambda(lambda x: (x[:, :6], x[:, 6:]), name="split")(inputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.functional_view(
        model, add_output_nodes=True, simple_text_visualization=True,
        text_callable="name_shape", styles={object: {"box_size": (60, 40)}},
    )
    assert image.width > 0 and image.height > 0


def test_functional_collapsed_caption_matches_last_output_shape():
    inputs = tf.keras.Input((8,), name="in")
    first = tf.keras.layers.Dense(6, name="first")(inputs)
    last = tf.keras.layers.Dense(3, name="last")(first)
    model = tf.keras.Model(inputs, last)
    image = visualkeras.functional_view(
        model, collapse_enabled=True, collapse_annotations=False,
        collapse_rules=[{"kind": "layer", "selector": tf.keras.layers.Dense, "repeat_count": 2}],
        simple_text_visualization=True, text_callable="shape",
        styles={object: {"box_size": (80, 60)}},
    )
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("override", [False, True])
def test_functional_output_markers_honor_global_and_named_styles(monkeypatch, override):
    inputs = tf.keras.Input((4,), name="in")
    outputs = tf.keras.layers.Dense(3, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    styles = {object: {"fill": "cyan", "box_size": (80, 60)}}
    if override:
        styles["output_0"] = {"fill": "red"}
    captured = []
    render = functional._render_graph

    def record(graph, **kwargs):
        captured.extend(dict(node.style) for node in graph.nodes.values() if node.name == "output_0")
        return render(graph, **kwargs)

    monkeypatch.setattr(functional, "_render_graph", record)
    image = visualkeras.functional_view(
        model, add_output_nodes=True, simple_text_visualization=True,
        text_callable="name_shape", styles=styles,
    )
    assert captured[0]["box_text_enabled"] is False
    assert captured[0]["fill"] == ("red" if override else "cyan")
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("padding", [0, 8])
@pytest.mark.parametrize("group_padding", [10, 40])
def test_functional_group_caption_and_frame_fit_inside_canvas(padding, group_padding):
    from PIL import ImageDraw, ImageFont

    inputs = tf.keras.Input((3,), name="in")
    outputs = tf.keras.layers.Dense(2, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    label = "Long group caption"
    image = visualkeras.functional_view(
        model, padding=padding,
        layered_groups=[{"name": label, "layers": ["in"], "padding": group_padding,
                         "font": font, "font_color": "magenta"}],
    )
    reference = Image.new("RGBA", (320, 80), "white")
    ImageDraw.Draw(reference).text((10, 10), label, font=font, fill="magenta")
    points = [(i % image.width, i // image.width) for i, (r, g, b, a) in enumerate(image.getdata())
              if r == b == 255 and g < 255]
    expected = [(i % reference.width, i // reference.width) for i, (r, g, b, a) in enumerate(reference.getdata())
                if r == b == 255 and g < 255]
    assert max(x for x, y in points) - min(x for x, y in points) == max(x for x, y in expected) - min(x for x, y in expected)
    assert max(y for x, y in points) - min(y for x, y in points) == max(y for x, y in expected) - min(y for x, y in expected)
    assert min(x for x, y in points) >= 0


def test_functional_unwrapped_box_labels_honor_line_limit():
    inputs = tf.keras.Input((4,), name="in")
    outputs = tf.keras.layers.Dense(3, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    style = {"box_size": (140, 80), "box_text": "First\nSecond", "box_text_wrap": "none",
             "box_text_max_lines": 1, "box_text_rotation": 0}
    settings = {"simple_text_visualization": True, "simple_text_label_mode": "inside"}
    image = visualkeras.functional_view(
        model, **settings, styles={"in": {"box_size": (80, 80)}, "out": style},
    )
    reference = visualkeras.functional_view(
        model, **settings, styles={"in": {"box_size": (80, 80)}, "out": {**style, "box_text": "First"}},
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()

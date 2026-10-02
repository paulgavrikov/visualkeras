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


def test_functional_centered_italic_multiline_label():
    from PIL import ImageFont

    inputs = tf.keras.Input((4,), name="in")
    outputs = tf.keras.layers.Dense(3, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSans-Oblique.ttf", 26)
    image = visualkeras.functional_view(
        model, simple_text_visualization=True, simple_text_label_mode="inside",
        styles={"in": {"box_size": (80, 80)}, "out": {
            "box_size": (180, 110), "box_text": "j\nMMMMMM", "box_text_font": font,
            "box_text_autoshrink": False, "box_text_rotation": 0,
        }},
    )
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("above", [False, True])
@pytest.mark.parametrize("text", ["A\ng", "A\n\ng"])
@pytest.mark.parametrize("italic", [False, True])
def test_functional_external_captions_preserve_glyphs_and_diagram(simple, above, text, italic):
    from PIL import ImageChops, ImageDraw, ImageFont

    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(6, name="out")(inputs))
    font = ImageFont.truetype("DejaVuSans-Oblique.ttf", 24) if italic else ImageFont.load_default()
    image = visualkeras.functional_view(
        model, draw_volume=False, padding=2, font=font, font_color="magenta", text_vspacing=10,
        simple_text_visualization=simple, styles={object: {"box_text_enabled": False}},
        text_callable=lambda i, layer: (text if layer.name == "out" else "", above),
    )
    reference = Image.new("RGBA", (160, 160), "white")
    ImageDraw.Draw(reference).multiline_text((10, 10), text, font=font, spacing=10, fill="magenta")

    def glyph_mask(im):
        mask = Image.new("L", im.size)
        mask.putdata([255 - g if r == b == 255 else 0 for r, g, b, a in im.getdata()])
        return mask

    actual_mask, expected_mask = glyph_mask(image), glyph_mask(reference)
    actual_bounds, expected_bounds = actual_mask.getbbox(), expected_mask.getbbox()
    assert actual_bounds is not None
    actual_glyphs = actual_mask.crop(actual_bounds)
    expected_glyphs = expected_mask.crop(expected_bounds)
    assert actual_glyphs.size == expected_glyphs.size
    assert actual_glyphs.tobytes() == expected_glyphs.tobytes()

    if not simple:
        diagram = image.copy()
        diagram.paste("white", mask=actual_mask.point(lambda value: 255 if value else 0))
        plain = visualkeras.functional_view(model, draw_volume=False, padding=2)
        diagram_bounds = ImageChops.difference(diagram.convert("RGB"), Image.new("RGB", diagram.size, "white")).getbbox()
        plain_bounds = ImageChops.difference(plain.convert("RGB"), Image.new("RGB", plain.size, "white")).getbbox()
        assert diagram.crop(diagram_bounds).size == plain.crop(plain_bounds).size
        assert diagram.crop(diagram_bounds).tobytes() == plain.crop(plain_bounds).tobytes()
        if above:
            assert actual_bounds[3] <= diagram_bounds[1] - 3
        else:
            assert actual_bounds[1] >= diagram_bounds[3] + 3


def test_functional_default_name_shape_captions_fit_and_callbacks_run_once():
    from visualkeras.options import LAYERED_TEXT_CALLABLES

    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(6, name="out")(inputs))
    calls = []

    def caption(index, layer):
        calls.append(layer.name)
        return LAYERED_TEXT_CALLABLES["name_shape"](index, layer)

    image = visualkeras.functional_view(model, draw_volume=False, text_callable=caption, font_color="magenta")
    reference = visualkeras.functional_view(model, draw_volume=False, text_callable="name_shape", font_color="magenta", padding=80)

    def glyphs(im):
        mask = Image.new("L", im.size)
        mask.putdata([255 - g if r == b == 255 else 0 for r, g, b, a in im.getdata()])
        return mask.crop(mask.getbbox())

    actual, expected = glyphs(image), glyphs(reference)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    assert calls == ["in", "out"]


def test_collapsed_block_annotations_preserve_opaque_canvas():
    inputs = tf.keras.Input((8,), name="in")
    outputs = inputs
    for index in range(4):
        outputs = tf.keras.layers.Dense(6, name=f"dense_{index}")(outputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.functional_view(
        model, simple_text_visualization=True, collapse_enabled=True,
        collapse_rules=[{"kind": "block", "selector": [tf.keras.layers.Dense, tf.keras.layers.Dense],
                         "repeat_count": 2, "label": "2x", "annotation_position": "below"}],
        styles={object: {"box_size": (100, 80)}},
    )
    assert image.getextrema()[3] == (255, 255)


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("caption_above", [False, True])
@pytest.mark.parametrize("annotation_position", ["above", "below"])
@pytest.mark.parametrize("padding", [0, 40])
def test_collapsed_captions_and_annotations_have_separate_space(monkeypatch, simple, caption_above, annotation_position, padding):
    from PIL import ImageDraw, ImageFont

    inputs = tf.keras.Input((8,), name="in")
    outputs = inputs
    for index in range(4):
        outputs = tf.keras.layers.Dense(6, name=f"dense_{index}")(outputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    annotation_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    text = "Caption\n6"
    caption_origin = []
    calls = []
    original_text = ImageDraw.ImageDraw.multiline_text

    def record_caption(draw, xy, value, **kwargs):
        if value == text:
            caption_origin.append(xy)
        return original_text(draw, xy, value, **kwargs)

    def caption(index, layer):
        calls.append(layer.name)
        return text if layer.name.startswith("collapsed_") else "", caption_above

    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record_caption)
    image = visualkeras.functional_view(
        model, simple_text_visualization=simple, draw_volume=False, padding=padding, min_xy=140,
        font=font, font_color="magenta", collapse_enabled=True,
        collapse_rules=[{"kind": "block", "selector": [tf.keras.layers.Dense, tf.keras.layers.Dense],
                         "repeat_count": 2, "label": "2x", "annotation_position": annotation_position}],
        styles={object: {"box_size": (140, 80), "box_text_enabled": False,
                        "collapse_badge_enabled": False,
                        "collapse_annotation_font": annotation_font, "collapse_annotation_color": "blue"}},
        text_callable=caption,
    )
    reference = Image.new("RGBA", (200, 100), "white")
    assert len(calls) == len(set(calls)) == 2
    assert len(caption_origin) == 1
    # Preserve fractional x rasterization while giving the reference ample margins.
    reference_x = 10 + caption_origin[0][0] % 1
    original_text(ImageDraw.Draw(reference), (reference_x, 10), text, font=font, fill="magenta")

    def glyphs(im):
        mask = Image.new("L", im.size)
        mask.putdata([255 - g if r == b == 255 else 0 for r, g, b, a in im.getdata()])
        return mask.crop(mask.getbbox())

    actual, expected = glyphs(image), glyphs(reference)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()
    caption_rows = [i // image.width for i, (r, g, b, a) in enumerate(image.getdata()) if r == b == 255 and g < 255]
    annotation_rows = [i // image.width for i, (r, g, b, a) in enumerate(image.getdata()) if b == 255 and r == g < 255]
    assert caption_rows and annotation_rows
    assert max(caption_rows) + 4 <= min(annotation_rows) or max(annotation_rows) + 4 <= min(caption_rows)


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("above", [False, True])
def test_collapsed_caption_spacing_keeps_neighboring_rows_clear(monkeypatch, simple, above):
    from PIL import ImageDraw, ImageFont

    inputs = tf.keras.Input((8,), name="in")
    outputs = []
    for branch in range(2):
        value = inputs
        for index in range(4):
            value = tf.keras.layers.Dense(6, name=f"branch_{branch}_{index}")(value)
        outputs.append(value)
    model = tf.keras.Model(inputs, outputs)
    rectangles, captions = [], []
    original_annotations = functional._draw_collapsed_annotations
    original_text = ImageDraw.ImageDraw.multiline_text

    def record_nodes(image, nodes, **kwargs):
        rectangles.extend(rect for node, rect in nodes)
        return original_annotations(image, nodes, **kwargs)

    def record_text(draw, xy, text, **kwargs):
        if text == "Caption\n6":
            captions.append(draw.multiline_textbbox(xy, text, font=kwargs["font"], spacing=kwargs["spacing"]))
        return original_text(draw, xy, text, **kwargs)

    monkeypatch.setattr(functional, "_draw_collapsed_annotations", record_nodes)
    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", record_text)
    image = visualkeras.functional_view(
        model, simple_text_visualization=simple, draw_volume=False, padding=0, row_spacing=0, min_xy=140,
        font=ImageFont.truetype("DejaVuSans.ttf", 24), collapse_enabled=True,
        collapse_rules=[{"kind": "block", "selector": [tf.keras.layers.Dense, tf.keras.layers.Dense],
                         "repeat_count": 2, "label": "2x", "annotation_position": "above" if above else "below"}],
        styles={object: {"box_size": (140, 80), "box_text_enabled": False}},
        text_callable=lambda i, layer: ("Caption\n6" if layer.name.startswith("collapsed_") else "", above),
    )
    assert len(rectangles) == len(captions) == 2
    for left, top, right, bottom in captions:
        assert 0 <= top < bottom <= image.height
        for _, node_top, _, node_bottom in rectangles:
            assert bottom + 4 <= node_top or top >= node_bottom + 4
    first, second = sorted(captions, key=lambda rect: rect[1])
    assert first[3] + 4 <= second[1]


@pytest.mark.parametrize("above", [False, True])
@pytest.mark.parametrize("target", ["in", "out"])
@pytest.mark.parametrize("padding", [0, 20])
def test_wide_external_captions_keep_complete_glyphs(above, target, padding):
    from PIL import ImageFont

    inputs = tf.keras.Input((8,), name="in")
    outputs = tf.keras.layers.Dense(6, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {
        "font": ImageFont.truetype("DejaVuSansMono.ttf", 24), "font_color": "magenta",
        "text_callable": lambda i, layer: ("Caption\n6" if layer.name == target else "", above),
    }
    image = visualkeras.functional_view(model, padding=padding, **settings)
    reference = visualkeras.functional_view(model, padding=160, **settings)

    def glyphs(im):
        mask = Image.new("L", im.size)
        mask.putdata([255 - g if r == b == 255 else 0 for r, g, b, a in im.getdata()])
        return mask.crop(mask.getbbox())

    actual, expected = glyphs(image), glyphs(reference)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()

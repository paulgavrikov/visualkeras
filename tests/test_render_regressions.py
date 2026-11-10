"""Regression tests for rendering bugs found when comparing against v0.2.0."""

import pytest
from PIL import Image, ImageChops

import visualkeras
from visualkeras import functional, layered, utils
from visualkeras.options import LAYERED_PRESETS, LAYERED_TEXT_CALLABLES, LayeredOptions


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


def _deep_ends_model():
    """Model whose first and last layers have the deepest boxes."""
    inputs = tf.keras.Input((28, 28, 16), name="in")
    x = tf.keras.layers.Conv2D(4, 3, strides=2, padding="same", name="down")(inputs)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", name="up")(x)
    return tf.keras.Model(inputs, x, name="deep_ends")


def _content_margins(img):
    """Return (left, right) distance from drawn content to the canvas edges."""
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, rgba.getpixel((0, 0)))
    bbox = ImageChops.difference(rgba, background).getbbox(alpha_only=False)
    assert bbox is not None
    return bbox[0], rgba.width - bbox[2]


@pytest.mark.parametrize("reversed_view", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
def test_layered_back_faces_stay_inside_canvas(reversed_view, with_text):
    kwargs = {"draw_reversed": reversed_view, "padding": 10}
    if with_text:
        kwargs["text_callable"] = "name"
    left, right = _content_margins(visualkeras.layered_view(_deep_ends_model(), **kwargs))
    # Anti-aliased outlines may reach a pixel or two into the padding (v0.2.0
    # renders 8-10 px here); clipped back faces used to leave 0 px.
    assert left >= 8 and right >= 8, (left, right)


def _record_fills(monkeypatch):
    fills = {}
    original = utils.Box.draw

    def record(box, draw, draw_reversed=False):
        layer = getattr(box, "layer", None)
        if layer is not None:
            fills[layer.name] = utils.get_rgba_tuple(box.fill)
        return original(box, draw, draw_reversed)

    monkeypatch.setattr(layered.Box, "draw", record)
    return fills


def _three_conv_model():
    inputs = tf.keras.Input((16, 16, 3), name="in")
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="c1")(inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="c2")(x)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="c3")(x)
    return tf.keras.Model(inputs, x)


def test_layered_name_style_does_not_bleed_into_same_class(monkeypatch):
    fills = _record_fills(monkeypatch)
    model = _three_conv_model()
    color_map = {tf.keras.layers.Conv2D: {"fill": "#00ff00"}}
    snapshot = {key: dict(value) for key, value in color_map.items()}

    visualkeras.layered_view(model, color_map=color_map, styles={"c2": {"fill": "#000000"}}, legend=True)

    assert fills["c1"][:3] == (0, 255, 0)
    assert fills["c2"][:3] == (0, 0, 0)
    assert fills["c3"][:3] == (0, 255, 0)
    assert color_map == snapshot, "the caller's color_map must not be mutated"


def test_layered_legend_uses_class_color_not_name_override(monkeypatch):
    model = _three_conv_model()
    legend_fills = []
    original = utils.Box.draw

    def record(box, draw, draw_reversed=False):
        if getattr(box, "layer", None) is None:  # legend swatches carry no layer
            legend_fills.append(utils.get_rgba_tuple(box.fill)[:3])
        return original(box, draw, draw_reversed)

    monkeypatch.setattr(layered.Box, "draw", record)
    visualkeras.layered_view(
        model,
        type_ignore=[tf.keras.layers.InputLayer],
        styles={tf.keras.layers.Conv2D: {"fill": "#00ff00"}, "c1": {"fill": "#000000"}},
        legend=True,
    )
    assert legend_fills == [(0, 255, 0)]


def _multi_io_model():
    a = tf.keras.Input((16,), name="in_a")
    b = tf.keras.Input((8,), name="in_b")
    merged = tf.keras.layers.Concatenate(name="cat")(
        [tf.keras.layers.Dense(12, name="da")(a), tf.keras.layers.Dense(12, name="db")(b)]
    )
    return tf.keras.Model([a, b], tf.keras.layers.Dense(3, name="out")(merged))


def test_functional_inside_labels_fit_their_boxes(monkeypatch):
    drawn = []
    original = functional._draw_box_text_in_rect

    def record(base, rect, text, **kwargs):
        drawn.append((rect, text, kwargs["style"], kwargs["fallback_font"], kwargs["fallback_spacing"]))
        return original(base, rect, text, **kwargs)

    monkeypatch.setattr(functional, "_draw_box_text_in_rect", record)
    visualkeras.functional_view(
        _multi_io_model(), simple_text_visualization=True, simple_text_label_mode="inside",
    )

    assert {text for _, text, *_ in drawn} >= {"Concatenate", "Dense"}
    for (x1, y1, x2, y2), text, style, font, spacing in drawn:
        need_w, need_h = functional._box_text_required_size(text, style, font, spacing)
        assert x2 - x1 >= need_w and y2 - y1 >= need_h, (text, (x2 - x1, y2 - y1), (need_w, need_h))


def test_functional_group_outline_encloses_member_captions(monkeypatch):
    inputs = tf.keras.Input((16, 16, 4), name="x")
    y = tf.keras.layers.Conv2D(4, 3, padding="same", name="c1")(inputs)
    y = tf.keras.layers.Conv2D(4, 3, padding="same", name="c2")(y)
    model = tf.keras.Model(inputs, tf.keras.layers.Add(name="skip")([inputs, y]))
    group = {"name": "Residual", "layers": ["c1", "c2"]}

    label_rects, captured = [], {}
    original_labels = functional._external_label_rects
    original_boxes = functional._draw_group_boxes

    def record_labels(labels):
        result = original_labels(labels)
        label_rects.append(result)
        return result

    def record_boxes(draw, graph, groups, group_rects=None):
        captured.update({"rect": group_rects[id(group)], "graph": graph})
        return original_boxes(draw, graph, groups, group_rects=group_rects)

    monkeypatch.setattr(functional, "_external_label_rects", record_labels)
    monkeypatch.setattr(functional, "_draw_group_boxes", record_boxes)
    visualkeras.functional_view(model, draw_volume=True, text_callable="name", layered_groups=[group])

    g_left, g_top, g_right, g_bottom = captured["rect"]
    final_labels = label_rects[-1]
    members = [node for node in captured["graph"].nodes.values() if node.name in ("c1", "c2")]
    assert len(members) == 2
    for node in members:
        left, top, right, bottom = final_labels[node.node_id]
        assert g_left <= left and right <= g_right and g_top <= top and bottom <= g_bottom


def test_layered_presets_keep_v020_definitions():
    assert LAYERED_PRESETS["flat"].to_kwargs() == LayeredOptions(
        draw_volume=False, draw_funnel=False, shade_step=0,
    ).to_kwargs()
    assert LAYERED_PRESETS["presentation"].to_kwargs() == LayeredOptions(
        draw_volume=False, draw_funnel=True, spacing=40, padding=30, scale_xy=2.5, scale_z=1.0,
        legend=True, show_dimension=True, sizing_mode="balanced",
        text_callable=LAYERED_TEXT_CALLABLES["name_shape"],
    ).to_kwargs()
    image = visualkeras.layered_view(_three_conv_model(), preset="flat")
    assert image.width > 0 and image.height > 0


def test_default_palette_matches_v020():
    assert utils.ColorWheel().colors == [
        "#ffd166", "#ef476f", "#118ab2", "#073b4c", "#842da1",
        "#ffbad4", "#fe9775", "#83d483", "#06d6a0", "#0cb0a9",
    ]

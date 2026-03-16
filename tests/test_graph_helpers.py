from types import SimpleNamespace

import aggdraw
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import visualkeras.graph as graph


def test_dummy_layer_sets_name_and_optional_units():
    basic = graph._DummyLayer("out")
    assert basic.name == "out"
    assert not hasattr(basic, "units")

    with_units = graph._DummyLayer("out2", units=9)
    assert with_units.name == "out2"
    assert with_units.units == 9


def test_draw_connector_uses_style_overrides(monkeypatch):
    calls = {}

    class FakeDraw:
        @staticmethod
        def line(coords, pen):
            calls["coords"] = coords
            calls["pen"] = pen

    def fake_pen(color, width):
        calls["color"] = color
        calls["width"] = width
        return ("pen", color, width)

    monkeypatch.setattr(graph.aggdraw, "Pen", fake_pen)

    start = SimpleNamespace(
        x1=0,
        y1=0,
        x2=10,
        y2=20,
        style={"connector_fill": "red", "connector_width": 3},
    )
    end = SimpleNamespace(x1=40, y1=10, x2=50, y2=30)
    graph._draw_connector(FakeDraw(), start, end, color="gray", width=1)

    assert calls["color"] == graph.get_rgba_tuple("red")
    assert calls["width"] == 3
    assert calls["coords"][0] == 10
    assert calls["coords"][2] == 40


def test_get_graph_group_nodes():
    l1 = SimpleNamespace(name="conv")
    l2 = SimpleNamespace(name="dense")
    model = SimpleNamespace(layers=[l1, l2])
    mapping = {id(l1): [SimpleNamespace(x1=0, y1=0, x2=10, y2=10)]}

    nodes = graph._get_graph_group_nodes(mapping, {"layers": ["conv", "missing"]}, model)
    assert len(nodes) == 1
    assert graph._get_graph_group_nodes(mapping, {"layers": []}, model) == []


def test_draw_graph_group_boxes_and_captions():
    l1 = SimpleNamespace(name="a")
    l2 = SimpleNamespace(name="b")
    model = SimpleNamespace(layers=[l1, l2])
    n1 = SimpleNamespace(x1=10, y1=10, x2=30, y2=30)
    n2 = SimpleNamespace(x1=35, y1=12, x2=55, y2=28)
    mapping = {id(l1): [n1], id(l2): [n2]}
    groups = [
        {
            "name": "Encoder",
            "layers": ["a", "b"],
            "padding": 4,
            "fill": (210, 210, 210, 120),
            "outline": "black",
            "font": ImageFont.load_default(),
            "font_color": "black",
            "text_spacing": 2,
        }
    ]

    img = Image.new("RGBA", (90, 70), "white")
    draw = aggdraw.Draw(img)
    graph._draw_graph_group_boxes(draw, mapping, groups, model)
    draw.flush()
    graph._draw_graph_group_captions(img, mapping, groups, model)

    assert img.getbbox() is not None


def test_font_and_measure_helpers():
    explicit_font = ImageFont.ImageFont()
    assert graph._get_font({"font": explicit_font}) == explicit_font

    fallback_font = graph._get_font({"font": "/definitely/missing/font.ttf"})
    assert fallback_font is not None

    draw = ImageDraw.Draw(Image.new("RGBA", (40, 20), "white"))
    w, h = graph._measure_text(draw, "abc", ImageFont.load_default())
    assert w > 0 and h > 0

    class FakeFont:
        pass

    class FakeDraw:
        @staticmethod
        def textsize(text, font=None):
            return (len(text) * 2, 9)

    assert graph._measure_text(FakeDraw(), "abcd", FakeFont()) == (8, 9)


def test_graph_view_options_validation_paths():
    with pytest.raises(TypeError):
        graph.graph_view(object(), options=object())

    with pytest.raises(ValueError, match="Unknown graph preset"):
        graph.graph_view(object(), options={}, preset="does-not-exist")


def test_graph_view_new_output_fallback_and_internal_input_path(monkeypatch):
    class _Layer:
        def __init__(self, name, *, input_shape, output):
            self.name = name
            self.input_shape = input_shape
            self.output = output

    out_a = SimpleNamespace()
    out_b = SimpleNamespace(name="named_output_tensor")
    layer_a = _Layer("in_a", input_shape=[(None, 4)], output=out_a)
    layer_b = _Layer("dense_b", input_shape=(None, 3), output=out_b)
    out_a._keras_history = (layer_a, None, None)

    model = SimpleNamespace(
        outputs=[out_a, out_b],
        layers=[layer_a, layer_b],
        output_shape=[(None, 4), (None, 3)],
    )
    id_map = {id(layer_a): 0, id(layer_b): 1}
    adj = np.zeros((2, 2))

    monkeypatch.setattr(graph, "model_to_adj_matrix", lambda m: (dict(id_map), adj.copy()))
    monkeypatch.setattr(graph, "model_to_hierarchy_lists", lambda m, *_: [[layer_a], [layer_b]])

    def fake_augment(_model, new_layers, mapping, matrix):
        mapping = dict(mapping)
        base = matrix.shape[0]
        expanded = np.zeros((base + len(new_layers), base + len(new_layers)))
        expanded[:base, :base] = matrix
        for i, layer in enumerate(new_layers):
            mapping[id(layer)] = base + i
        return mapping, expanded

    monkeypatch.setattr(graph, "augment_output_layers", fake_augment)
    monkeypatch.setattr(graph, "is_internal_input", lambda layer: layer is layer_a)

    img = graph.graph_view(
        model,
        show_neurons=True,
        inout_as_tensor=False,
        styles={"in_a": {"ellipsize_after": 0}},
    )
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0


def test_graph_view_raises_on_unsupported_internal_input_shape(monkeypatch):
    output = SimpleNamespace(name="y")

    class _Layer:
        pass

    layer = _Layer()
    layer.name = "bad_input"
    layer.input_shape = [(None, 4), (None, 2)]
    layer.output = output
    model = SimpleNamespace(
        output_names=["bad_input"],
        outputs=[output],
        output_shape=(None, 4),
        layers=[layer],
    )

    monkeypatch.setattr(graph, "model_to_adj_matrix", lambda m: ({id(layer): 0}, np.zeros((1, 1))))
    monkeypatch.setattr(graph, "model_to_hierarchy_lists", lambda m, *_: [[layer]])
    monkeypatch.setattr(graph, "is_internal_input", lambda _layer: True)

    with pytest.raises(RuntimeError, match="not supported input shape"):
        graph.graph_view(model, show_neurons=True, inout_as_tensor=False)


def test_graph_get_font_default_fallback_path(monkeypatch):
    orig_truetype = graph.ImageFont.truetype

    def fake_truetype(font, *args, **kwargs):
        if font == "arial.ttf":
            raise IOError("x")
        return orig_truetype(font, *args, **kwargs)

    monkeypatch.setattr(graph.ImageFont, "truetype", fake_truetype)
    font = graph._get_font({})
    assert font is not None


def test_graph_view_shift_image_warning_and_fallback_layer_name(monkeypatch, tmp_path):
    class _Layer:
        pass

    layer = _Layer()
    layer.input_shape = (None, 3)
    layer.output = SimpleNamespace(name="tensor_in")

    layer2 = SimpleNamespace(name="dense_out", units=2, output=SimpleNamespace(name="tensor_out"))

    model = SimpleNamespace(
        output_names=["out"],
        outputs=[layer2.output],
        layers=[layer, layer2],
        output_shape=(None, 3),
    )

    monkeypatch.setattr(
        graph,
        "model_to_adj_matrix",
        lambda m: ({id(layer): 0, id(layer2): 1}, np.array([[0, 1], [0, 0]])),
    )
    monkeypatch.setattr(graph, "model_to_hierarchy_lists", lambda m, *_: [[layer], [layer2]])
    monkeypatch.setattr(graph, "augment_output_layers", lambda m, l, mapping, adj: (mapping, adj))
    monkeypatch.setattr(graph, "is_internal_input", lambda l: l is layer)

    img = graph.graph_view(
        model,
        inout_as_tensor=False,
        show_neurons=True,
        styles={_Layer: {"image": str(tmp_path / "missing.png")}},
        layered_groups=[{"name": "HugePad", "layers": [layer], "padding": 120, "text_spacing": 6}],
    )
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0

from types import SimpleNamespace

import aggdraw
from PIL import Image, ImageDraw, ImageFont

import visualkeras.layered as layered


class _TensorShape:
    def __init__(self, values, raise_on_as_list=False):
        self._values = values
        self._raise_on_as_list = raise_on_as_list

    def as_list(self):
        if self._raise_on_as_list:
            raise RuntimeError("boom")
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


def test_shape_to_tuple_variants():
    assert layered._shape_to_tuple(None) is None
    assert layered._shape_to_tuple((1, 2)) == (1, 2)
    assert layered._shape_to_tuple([1, 2]) == (1, 2)
    assert layered._shape_to_tuple(_TensorShape([1, 2])) == (1, 2)
    assert layered._shape_to_tuple(_TensorShape([3, 4], raise_on_as_list=True)) == (3, 4)


def test_resolve_layer_output_shape_priority():
    explicit = SimpleNamespace(output_shape=(None, 8), output=None)
    assert layered._resolve_layer_output_shape(explicit) == (None, 8)

    fallback_output = SimpleNamespace(
        output_shape=None,
        output=SimpleNamespace(shape=_TensorShape([None, 16])),
    )
    assert layered._resolve_layer_output_shape(fallback_output) == (None, 16)

    class Computed:
        output_shape = None
        output = None
        input_shape = (None, 9)

        @staticmethod
        def compute_output_shape(shape):
            return (shape[0], 3)

    assert layered._resolve_layer_output_shape(Computed()) == (None, 3)

    class Broken:
        output_shape = None
        output = None
        input_shape = (None, 9)

        @staticmethod
        def compute_output_shape(shape):
            raise RuntimeError("bad shape")

    assert layered._resolve_layer_output_shape(Broken()) is None


def _box(layer, x1, y1, x2, y2, de):
    return SimpleNamespace(layer=layer, x1=x1, y1=y1, x2=x2, y2=y2, de=de)


def test_get_group_boxes_by_object_and_name():
    layer_a = SimpleNamespace(name="conv_a")
    layer_b = SimpleNamespace(name="conv_b")
    boxes = [
        _box(layer_a, 0, 0, 10, 10, 2),
        _box(layer_b, 20, 0, 30, 10, 2),
        SimpleNamespace(x1=99, y1=99, x2=100, y2=100),  # no layer attr
    ]

    group = {"layers": [layer_a, "conv_b", "missing"]}
    selected = layered._get_group_boxes(boxes, group)
    assert selected == boxes[:2]
    assert layered._get_group_boxes(boxes, {"layers": []}) == []


def test_get_logo_boxes_by_name_and_type():
    class Conv:
        def __init__(self, name):
            self.name = name

    class Dense:
        def __init__(self, name):
            self.name = name

    conv = Conv("conv")
    dense = Dense("dense")
    boxes = [_box(conv, 0, 0, 10, 10, 1), _box(dense, 20, 0, 30, 10, 1)]

    result = layered._get_logo_boxes(boxes, {"layers": ["conv", Dense]})
    assert boxes[0] in result
    assert boxes[1] in result


def test_draw_group_boxes_and_captions():
    layer_a = SimpleNamespace(name="a")
    layer_b = SimpleNamespace(name="b")
    boxes = [_box(layer_a, 10, 10, 30, 30, 4), _box(layer_b, 40, 20, 60, 40, 4)]
    groups = [
        {
            "name": "Group 1",
            "layers": [layer_a, "b"],
            "padding": 3,
            "fill": (220, 220, 220, 120),
            "outline": "black",
            "font": ImageFont.load_default(),
            "font_color": "black",
            "text_spacing": 2,
        }
    ]

    img = Image.new("RGBA", (90, 90), "white")
    draw = aggdraw.Draw(img)
    layered._draw_layered_group_boxes(draw, boxes, groups, draw_reversed=False)
    draw.flush()
    layered._draw_layered_group_captions(img, boxes, groups, draw_reversed=False)
    layered._draw_layered_group_boxes(draw, boxes, groups, draw_reversed=True)

    assert img.getbbox() is not None


def test_font_and_measure_helpers():
    explicit_font = ImageFont.ImageFont()
    assert layered._get_font({"font": explicit_font}) == explicit_font

    fallback_font = layered._get_font({"font": "/definitely/missing/font.ttf", "font_size": 12})
    assert fallback_font is not None

    draw = ImageDraw.Draw(Image.new("RGBA", (40, 20), "white"))
    w, h = layered._measure_text(draw, "abc", ImageFont.load_default())
    assert w > 0 and h > 0

    class FakeFont:
        pass

    class FakeDraw:
        @staticmethod
        def textsize(text, font=None):
            return (len(text) * 2, 7)

    assert layered._measure_text(FakeDraw(), "abcd", FakeFont()) == (8, 7)

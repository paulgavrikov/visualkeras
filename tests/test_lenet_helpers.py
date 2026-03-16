from types import SimpleNamespace

import aggdraw
import pytest
from PIL import Image, ImageDraw, ImageFont

import visualkeras.lenet as lenet
from visualkeras.lenet import (
    FeatureMapStack,
    PyramidConnection,
    _adjust_wh_for_image_aspect,
    _apply_face_images,
    _choose_y_from_ranges,
    _as_tuple2,
    _canonicalize_shape,
    _clamp_int,
    _clamp_rect_to_face,
    _enforce_in_out_patch_separation,
    _effective_patch_alpha_for_layer,
    _fit_image_to_rect,
    _get_multiline_text_size,
    _get_text_size,
    _load_face_image,
    _parse_face_image_style,
    _resolve_layer_output_shape,
    _shape_to_tuple,
    _stable_seed,
    _with_alpha,
)


def test_as_tuple2_variants():
    assert _as_tuple2(None) == (1, 1)
    assert _as_tuple2(3) == (3, 3)
    assert _as_tuple2((2, 5)) == (2, 5)
    assert _as_tuple2([7]) == (1, 7)
    assert _as_tuple2("x") == (1, 1)


def test_clamp_int():
    assert _clamp_int(-2, 0, 10) == 0
    assert _clamp_int(22, 0, 10) == 10
    assert _clamp_int(5.4, 0, 10) == 5


def test_clamp_rect_to_face():
    rect = _clamp_rect_to_face(5, 5, 8, 8, (0, 0, 10, 10), margin=1)
    assert rect == (1.0, 1.0, 9.0, 9.0)

    collapsed = _clamp_rect_to_face(5, 5, 8, 8, (0, 0, 1, 1), margin=2)
    assert collapsed == (0.5, 0.5, 0.5, 0.5)


def test_with_alpha_clamps():
    assert _with_alpha((10, 20, 30, 200), 40) == (10, 20, 30, 40)
    assert _with_alpha((10, 20, 30, 200), -10) == (10, 20, 30, 0)
    assert _with_alpha((10, 20, 30, 200), 999) == (10, 20, 30, 255)


def test_effective_patch_alpha_for_layer():
    style = {"patch_fill_alpha": 111}
    assert _effective_patch_alpha_for_layer(style, has_face_image=True, base_alpha=200, default_on_image=140) == 111

    style = {"patch_fill_alpha_on_image": 90}
    assert _effective_patch_alpha_for_layer(style, has_face_image=True, base_alpha=200, default_on_image=140) == 90

    style = {}
    assert _effective_patch_alpha_for_layer(style, has_face_image=True, base_alpha=200, default_on_image=140) == 140
    assert _effective_patch_alpha_for_layer(style, has_face_image=False, base_alpha=200, default_on_image=140) == 200


def test_stable_seed_deterministic():
    a = _stable_seed(42, "conv", 1)
    b = _stable_seed(42, "conv", 1)
    c = _stable_seed(42, "conv", 2)
    assert a == b
    assert a != c


class _TensorShape:
    def __init__(self, values):
        self._values = values

    def as_list(self):
        return list(self._values)


def test_shape_to_tuple_variants():
    assert _shape_to_tuple((1, 2, 3)) == (1, 2, 3)
    assert _shape_to_tuple([1, 2, 3]) == (1, 2, 3)
    assert _shape_to_tuple(_TensorShape([1, 2])) == (1, 2)
    assert _shape_to_tuple(None) is None


def test_resolve_layer_output_shape_priority():
    layer = SimpleNamespace(output_shape=(None, 12, 12, 3), output=None)
    assert _resolve_layer_output_shape(layer) == (None, 12, 12, 3)

    layer2 = SimpleNamespace(output_shape=None, output=SimpleNamespace(shape=_TensorShape([None, 8])))
    assert _resolve_layer_output_shape(layer2) == (None, 8)

    class _Layer:
        output_shape = None
        output = None
        input_shape = (None, 4)

        @staticmethod
        def compute_output_shape(shape):
            return (shape[0], 2)

    assert _resolve_layer_output_shape(_Layer()) == (None, 2)


def test_load_face_image_from_object_and_path(tmp_path):
    img = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
    assert _load_face_image(img).size == (4, 3)

    path = tmp_path / "face.png"
    img.save(path)
    loaded = _load_face_image(str(path))
    assert loaded is not None
    assert loaded.size == (4, 3)
    assert _load_face_image("does-not-exist.png") is None


def test_parse_face_image_style():
    style = {
        "face_image": {"path": "/tmp/logo.png", "fit": "contain", "alpha": 180, "inset": 2},
        "face_image_fit": "fill",
        "face_image_alpha": 255,
    }
    spec, fit, alpha, inset = _parse_face_image_style(style)
    assert spec == "/tmp/logo.png"
    assert fit == "contain"
    assert alpha == 180
    assert inset == 2

    spec, fit, alpha, inset = _parse_face_image_style({"face_image": "x.png", "face_image_fit": "unknown"})
    assert spec == "x.png"
    assert fit == "cover"
    assert alpha == 255
    assert inset is None


def test_adjust_wh_for_image_aspect():
    img = Image.new("RGBA", (40, 20))
    assert _adjust_wh_for_image_aspect(10, 10, img, min_xy=4, max_xy=100) == (20, 10)

    weird = SimpleNamespace(size=(0, 0))
    assert _adjust_wh_for_image_aspect(10, 8, weird, min_xy=4, max_xy=100) == (10, 8)


def test_fit_image_to_rect_modes():
    img = Image.new("RGBA", (10, 5), (255, 0, 0, 255))
    contain = _fit_image_to_rect(img, 20, 20, fit="contain", background=(0, 0, 0, 0))
    cover = _fit_image_to_rect(img, 20, 20, fit="cover", background=(0, 0, 0, 0))
    fill = _fit_image_to_rect(img, 20, 20, fit="fill", background=(0, 0, 0, 0))
    assert contain.size == (20, 20)
    assert cover.size == (20, 20)
    assert fill.size == (20, 20)


def test_apply_face_images_with_cache_and_fallback(tmp_path, monkeypatch):
    base = Image.new("RGBA", (80, 60), (255, 255, 255, 255))
    face_path = tmp_path / "face.png"
    Image.new("RGBA", (24, 12), (220, 60, 40, 255)).save(face_path)

    st = FeatureMapStack(
        10,
        12,
        30,
        20,
        4,
        map_spacing=2,
        max_visual_channels=4,
        fill="white",
        outline="black",
        line_width=1,
        shade_step=4,
    )

    stacks = [
        {
            "style": {
                "face_image": str(face_path),
                "face_image_fit": "contain",
                "face_image_alpha": 170,
                "_face_image_obj": "not-an-image",
            },
            "stack": st,
        },
        {
            "style": {
                "face_image": str(tmp_path / "missing.png"),
            },
            "stack": st,
        },
    ]

    monkeypatch.setattr(Image.Image, "alpha_composite", lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    _apply_face_images(base, stacks)

    assert isinstance(stacks[0]["style"].get("_face_image_obj"), Image.Image)
    assert base.getbbox() is not None


def test_canonicalize_shape_branches():
    class Conv1DLayer:
        data_format = None

    class DenseLike:
        data_format = None

    conv1d_layer = Conv1DLayer()
    rs_conv1d = _canonicalize_shape(conv1d_layer, (None, 12, 7))
    assert rs_conv1d.kind == "spatial"
    assert rs_conv1d.w == 12 and rs_conv1d.c == 7

    plain = DenseLike()
    rs_vector = _canonicalize_shape(plain, (None, 3, 5))
    assert rs_vector.kind == "vector"
    assert rs_vector.c == 15

    rs_multi = _canonicalize_shape(plain, [(None, 6, 4, 2), (None, 1)])
    assert rs_multi.kind == "spatial"

    rs_bad = _canonicalize_shape(plain, "not-a-shape")
    assert rs_bad.kind == "vector"

    rs_none = _canonicalize_shape(plain, None)
    assert rs_none.kind == "vector"


def test_stack_connection_and_text_helpers():
    stack = FeatureMapStack(
        20,
        15,
        18,
        12,
        3,
        map_spacing=2,
        max_visual_channels=3,
        fill="white",
        outline="black",
        line_width=1,
        shade_step=0,
    )
    assert stack.front_anchor()[0] > 20
    assert stack.left_mid()[0] == stack.front_rect()[0]
    assert stack.right_mid()[0] == stack.front_rect()[2]

    img = Image.new("RGBA", (120, 80), (255, 255, 255, 255))
    draw = aggdraw.Draw(img)
    stack.draw(draw)

    dst = FeatureMapStack(
        70,
        20,
        20,
        14,
        2,
        map_spacing=2,
        max_visual_channels=2,
        fill="white",
        outline="black",
        line_width=1,
        shade_step=0,
    )
    conn = PyramidConnection(
        stack,
        dst,
        src_patch=(30, 20, 36, 26),
        dst_patch=(70, 24, 74, 28),
        connector_fill="gray",
        connector_width=1,
        src_patch_fill=(120, 120, 200, 120),
        dst_patch_fill=(120, 120, 200, 120),
        patch_outline="black",
        draw_patches=False,
    )
    conn.draw(draw)
    draw.flush()
    assert img.getbbox() is not None

    class _NoTextBbox:
        def textsize(self, text, font=None):
            return (len(text) * 3, 7)

    font = ImageFont.load_default()
    w, h = _get_text_size(_NoTextBbox(), "abc", font)
    assert w > 0 and h > 0
    assert _get_multiline_text_size(ImageDraw.Draw(Image.new("RGBA", (10, 10))), "", font) == (0, 0)


def test_lenet_view_options_and_empty_canvas(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        lenet.lenet_view(object(), preset="unknown-preset")
    with pytest.raises(TypeError):
        lenet.lenet_view(object(), options=123)

    monkeypatch.setattr(lenet, "get_layers", lambda model: [])
    out = tmp_path / "empty.png"
    img = lenet.lenet_view(object(), to_file=str(out), options={"padding": 5})
    assert img.size == (10, 10)
    assert out.exists() and out.stat().st_size > 0


def test_shape_resolution_and_alpha_invalid_inputs():
    class _BadTensorShape:
        @staticmethod
        def as_list():
            raise RuntimeError("bad")

    bad_shape = _shape_to_tuple(_BadTensorShape())
    assert isinstance(bad_shape, _BadTensorShape)

    class _BrokenLayer:
        output_shape = None
        output = None
        input_shape = (None, 4)

        @staticmethod
        def compute_output_shape(shape):
            raise RuntimeError("x")

    assert _resolve_layer_output_shape(_BrokenLayer()) is None

    style = {"patch_fill_alpha": "not-an-int", "patch_alpha_on_image": "bad"}
    alpha = _effective_patch_alpha_for_layer(style, has_face_image=True, base_alpha=220, default_on_image=140)
    assert alpha == 140


def test_load_and_parse_face_image_corner_cases(tmp_path, monkeypatch):
    assert _load_face_image(None) is None
    assert _load_face_image(123) is None

    p = tmp_path / "cached.png"
    Image.new("RGBA", (5, 5), (10, 20, 30, 255)).save(p)
    first = _load_face_image(str(p))
    second = _load_face_image(str(p))
    assert first is not None and second is not None
    assert first.size == second.size == (5, 5)

    monkeypatch.setattr(Image.Image, "convert", lambda self, mode: (_ for _ in ()).throw(RuntimeError("bad convert")))
    assert _load_face_image(Image.new("RGBA", (2, 2))) is None

    spec, fit, alpha, inset = _parse_face_image_style(
        {
            "face_image": {"path": str(p), "fit": "invalid", "alpha": "oops", "inset": "oops"},
            "face_image_inset": "not-int",
        }
    )
    assert spec == str(p)
    assert fit == "cover"
    assert alpha == 255
    assert inset is None


def test_adjust_fit_and_patch_separation_corner_cases():
    img = Image.new("RGBA", (20, 10))
    # Deliberately inverted bounds to force fallback branches.
    adjusted = _adjust_wh_for_image_aspect(10, 8, img, min_xy=12, max_xy=6)
    assert isinstance(adjusted, tuple) and len(adjusted) == 2

    filled = _fit_image_to_rect(img, 0, -5, fit="match_aspect", background=(0, 0, 0, 0))
    assert filled.size[0] >= 1 and filled.size[1] >= 1

    class _BadImage:
        @property
        def size(self):
            raise RuntimeError("bad size")

    blank = _fit_image_to_rect(_BadImage(), 8, 6, fit="contain", background=(1, 2, 3, 4))
    assert blank.size == (8, 6)

    rng = __import__("random").Random(0)
    y = _choose_y_from_ranges(rng, prefer=10, ranges=[(0, 20)])
    assert y is not None

    face = (0.0, 0.0, 10.0, 10.0)
    incoming = (9.0, 1.0, 10.0, 2.0)
    outgoing = (1.0, 1.0, 3.0, 3.0)
    inc, out = _enforce_in_out_patch_separation(face, incoming, outgoing, rng=rng, v_gap=5.0, margin=1.0)
    assert inc[0] <= out[0]


def test_lenet_view_face_image_style_and_text_font_fallback(tmp_path, monkeypatch):
    class _Layer:
        def __init__(self, name):
            self.name = name
            self.output_shape = (None, 8)
            self.data_format = None

    layer = _Layer("dense_1")
    icon = tmp_path / "face_style.png"
    Image.new("RGBA", (30, 18), (0, 100, 220, 255)).save(icon)

    monkeypatch.setattr(lenet, "get_layers", lambda model: [layer])
    monkeypatch.setattr(lenet.ImageFont, "load_default", lambda: (_ for _ in ()).throw(RuntimeError("no font")))

    img = lenet.lenet_view(
        object(),
        font=None,
        top_label=False,
        bottom_label=False,
        styles={"dense_1": {"face_image": str(icon), "face_image_fit": "match_aspect"}},
    )
    assert isinstance(img, Image.Image)

from types import SimpleNamespace

from PIL import Image

from visualkeras.lenet import (
    _adjust_wh_for_image_aspect,
    _as_tuple2,
    _clamp_int,
    _clamp_rect_to_face,
    _effective_patch_alpha_for_layer,
    _fit_image_to_rect,
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

from pathlib import Path

import pytest
from PIL import Image

import visualkeras


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


def _build_dense_model():
    inputs = tf.keras.layers.Input(shape=(8,), name="input")
    x = tf.keras.layers.Dense(12, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(10, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_3")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="out")(x)
    model = tf.keras.Model(inputs, outputs, name="dense_model_advanced")
    model.build((None, 8))
    return model


def _build_conv_model():
    inputs = tf.keras.layers.Input((12, 12, 3), name="input")
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same", name="conv_a")(inputs)
    x = tf.keras.layers.MaxPool2D(2, name="pool")(x)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv_b")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="dense")(x)
    model = tf.keras.Model(inputs, outputs, name="conv_model_advanced")
    model.build((None, 12, 12, 3))
    return model


def _make_icon(path: Path):
    Image.new("RGBA", (32, 20), (255, 90, 40, 255)).save(path)


def test_spacing_layer_can_be_used_in_a_real_model():
    inputs = tf.keras.Input((8,), name="spacing_input")
    hidden = tf.keras.layers.Dense(8, name="before_gap")(inputs)
    spaced = visualkeras.SpacingDummyLayer(spacing=32)(hidden)
    outputs = tf.keras.layers.Dense(3, name="after_gap")(spaced)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.layered_view(model, draw_volume=False, legend_text_spacing_offset=0)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("reversed_view", [False, True])
def test_spacing_subclasses_keep_gap_and_stay_out_of_legend(reversed_view):
    class CustomGap(visualkeras.SpacingDummyLayer):
        pass

    def build(gap_type):
        inputs = tf.keras.Input((8,), name="input")
        before = tf.keras.layers.Dense(8, name="pre")(inputs)
        gap = gap_type(spacing=45, name="gap")(before)
        outputs = tf.keras.layers.Dense(3, name="post")(gap)
        return tf.keras.Model(inputs, outputs)

    settings = {"draw_reversed": reversed_view, "legend": True, "text_callable": "name", "legend_text_spacing_offset": 0}
    reference = visualkeras.show(build(visualkeras.SpacingDummyLayer), mode="layered", **settings)
    image = visualkeras.show(build(CustomGap), mode="layered", **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["graph", "functional", "lenet"])
def test_public_model_adapter_renders_without_private_layer_containers(mode):
    class PublicModelAdapter:
        def __init__(self, model):
            self.layers = model.layers
            self.outputs = model.outputs
            self.output_names = model.output_names
            self.output_shape = model.output_shape
            self._model = model

        def get_layer(self, name):
            return self._model.get_layer(name=name)

    inputs = tf.keras.Input((8,), name="adapter_input")
    outputs = tf.keras.layers.Dense(3, name="adapter_output")(inputs)
    model = PublicModelAdapter(tf.keras.Model(inputs, outputs))
    image = visualkeras.show(model, mode=mode)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_numpy_shape_metadata_preserves_rendered_dimensions(mode):
    import numpy as np

    class ShapeMetadata(tf.keras.layers.Layer):
        def __init__(self, numpy_dimensions, **kwargs):
            super().__init__(**kwargs)
            self.numpy_dimensions = numpy_dimensions

        def call(self, inputs):
            return inputs

        @property
        def output_shape(self):
            integer = np.int64 if self.numpy_dimensions else int
            return (None, integer(6), integer(8), integer(12))

    def build(numpy_dimensions):
        inputs = tf.keras.Input((6, 8, 12), name="numeric_input")
        outputs = ShapeMetadata(numpy_dimensions, name="numeric_output")(inputs)
        return tf.keras.Model(inputs, outputs)

    options = {"draw_volume": False, "min_xy": 2, "min_z": 2, "scale_xy": 1, "scale_z": 1}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    reference = visualkeras.show(build(False), mode=mode, options=options)
    image = visualkeras.show(build(True), mode=mode, options=options)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_dynamic_spatial_axis_does_not_shift_known_height(mode):
    inputs = tf.keras.Input((None, 12, 3), name="dynamic_input")
    outputs = tf.keras.layers.Conv2D(8, 1, name="dynamic_conv")(inputs)
    model = tf.keras.Model(inputs, outputs)
    options = {"padding": 0, "draw_volume": False, "min_xy": 2, "min_z": 2, "scale_xy": 1, "scale_z": 1}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    image = visualkeras.show(model, mode=mode, options=options)
    assert image.height == 12


@pytest.mark.parametrize("mode", ["layered", "functional"])
@pytest.mark.parametrize("sizing_mode", ["balanced", "logarithmic"])
def test_high_rank_tensor_uses_all_channel_axes(mode, sizing_mode):
    from visualkeras.layer_utils import calculate_layer_dimensions

    inputs = tf.keras.Input((4, 5, 2, 40), name="volume_input")
    outputs = tf.keras.layers.Activation("linear", name="volume_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    dimensions = calculate_layer_dimensions(
        (None, 4, 5, 2, 40), 1, 1, 500, 500, 2, 2, sizing_mode=sizing_mode,
    )
    assert dimensions[2] == (48 if sizing_mode == "balanced" else 38)
    options = {"draw_volume": False, "min_xy": 2, "min_z": 2, "scale_xy": 1, "scale_z": 1, "sizing_mode": sizing_mode}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    image = visualkeras.show(model, mode=mode, options=options)
    assert image.width > 0 and image.height > 0


def test_layered_subclass_style_overrides_base_class():
    class StyledDense(tf.keras.layers.Dense):
        pass

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,), name="style_input"),
        StyledDense(4, name="styled_dense"),
    ])
    image = visualkeras.layered_view(
        model,
        styles={tf.keras.layers.Dense: {"fill": "red"}, StyledDense: {"fill": "blue"}},
        draw_volume=False,
        legend_text_spacing_offset=0,
    )
    assert (0, 0, 255, 255) in image.getdata()
    assert (255, 0, 0, 255) not in image.getdata()


@pytest.mark.parametrize("via_show", [False, True])
def test_functional_subclass_style_overrides_base_class(via_show):
    class StyledDense(tf.keras.layers.Dense):
        pass

    inputs = tf.keras.layers.Input(shape=(8,), name="style_input")
    outputs = StyledDense(4, name="styled_dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {"styles": {
        tf.keras.layers.Dense: {"fill": "red"}, StyledDense: {"fill": "blue"}
    }, "draw_volume": False}
    image = visualkeras.show(model, mode="functional", **settings) if via_show else visualkeras.functional_view(model, **settings)
    assert (0, 0, 255, 255) in image.getdata()
    assert (255, 0, 0, 255) not in image.getdata()


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_capped_spatial_model_renders(mode):
    inputs = tf.keras.layers.Input(shape=(12, 12, 3), name="capped_input")
    outputs = tf.keras.layers.Conv2D(8, 3, name="capped_conv")(inputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.show(
        model, mode=mode, sizing_mode="capped", draw_volume=False,
        dimension_caps={"sequence": 24, "channels": 12},
    )
    assert image.width > 0 and image.height > 0
    assert image.getbbox() is not None


@pytest.mark.parametrize("mode", ["layered", "functional", "lenet"])
def test_modern_multioutput_layer_uses_primary_tensor_shape(mode):
    import importlib
    from visualkeras.layer_utils import extract_primary_shape

    inputs = tf.keras.Input((8,), name="split_input")
    split = tf.keras.layers.Lambda(lambda tensor: (tensor[:, :6], tensor[:, :2]), name="split")
    outputs = split(inputs)
    model = tf.keras.Model(inputs, outputs[0])
    module = importlib.import_module("visualkeras." + mode)
    primary = extract_primary_shape(module._resolve_layer_output_shape(split))
    assert primary == (None, 6)
    image = visualkeras.show(model, mode=mode)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_transparent_logos_preserve_opaque_canvas_alpha(tmp_path, mode):
    icon = tmp_path / "transparent_logo.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8,), name="logo_input")
    outputs = tf.keras.layers.Dense(8, name="logo_dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.show(
        model, mode=mode, draw_volume=False, background_fill="white",
        logo_groups=[{"file": str(icon), "layers": ["logo_dense"], "size": 0.8}],
    )
    assert image.getextrema()[3] == (255, 255)


@pytest.mark.parametrize("circular_crop", [False, True])
@pytest.mark.parametrize("transparent_background", [False, True])
def test_graph_embedded_image_preserves_alpha_and_color(tmp_path, circular_crop, transparent_background):
    icon = tmp_path / "node_image.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((3,), name="image_input")
    outputs = tf.keras.layers.Dense(2, name="image_dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    options = {
        "node_size": 32, "background_fill": (0, 0, 0, 0) if transparent_background else "white",
        "circular_crop": circular_crop,
        "styles": {tf.keras.layers.Dense: {"image": str(icon), "image_fit": "fill"}},
    }
    image = visualkeras.show(model, mode="graph", options=options)
    expected_pixel = (255, 0, 0, 128) if transparent_background else (255, 127, 127, 255)
    assert expected_pixel in image.getdata()
    if not transparent_background:
        assert image.getextrema()[3] == (255, 255)


@pytest.mark.parametrize("reversed_view", [False, True])
def test_layered_front_image_preserves_opaque_canvas(tmp_path, reversed_view):
    icon = tmp_path / "front_image.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8,), name="front_input")
    outputs = tf.keras.layers.Dense(8, name="front_dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    image = visualkeras.show(model, mode="layered", options={
        "draw_reversed": reversed_view, "background_fill": "white", "legend_text_spacing_offset": 0,
        "styles": {"front_dense": {"image": str(icon), "image_axis": "z", "fill": "white"}},
    })
    assert (255, 127, 127, 255) in image.getdata()
    assert image.getextrema()[3] == (255, 255)


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_transparent_legend_icons_preserve_opaque_canvas(tmp_path, mode):
    icon = tmp_path / "legend_icon.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8,), name="legend_input")
    outputs = tf.keras.layers.Dense(8, name="legend_dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    options = {"draw_volume": False, "background_fill": "white", "logos_legend": True,
               "logo_groups": [{"name": "Icon", "file": str(icon), "layers": ["legend_dense"]}]}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    image = visualkeras.show(model, mode=mode, options=options)
    assert image.getextrema()[3] == (255, 255)


def test_graph_view_advanced_paths(tmp_path):
    model = _build_dense_model()
    icon_path = tmp_path / "icon.png"
    out_path = tmp_path / "graph_advanced.png"
    _make_icon(icon_path)

    styles = {
        "dense_1": {"image": str(icon_path), "image_fit": "contain", "image_indices": [0], "node_size": 40},
        "dense_2": {"image": str(icon_path), "image_fit": "cover", "circular_crop": True},
    }
    layered_groups = [
        {"name": "Dense Group", "layers": ["dense_1", "dense_2"], "padding": 8, "text_spacing": 4}
    ]

    img = visualkeras.graph_view(
        model,
        show_neurons=True,
        inout_as_tensor=True,
        ellipsize_after=4,
        styles=styles,
        layered_groups=layered_groups,
        to_file=str(out_path),
    )

    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0
    assert out_path.exists() and out_path.stat().st_size > 0


def test_layered_view_advanced_paths(tmp_path):
    model = _build_conv_model()
    icon_path = tmp_path / "layered_icon.png"
    out_path = tmp_path / "layered_advanced.png"
    _make_icon(icon_path)

    styles = {
        "conv_a": {"image": str(icon_path), "image_fit": "match_aspect", "image_axis": "z", "scale_image": 1.1},
        "conv_b": {"image": str(icon_path), "image_fit": "contain", "image_axis": "x"},
        "dense": {"image": str(icon_path), "image_fit": "cover", "image_axis": "y"},
    }
    groups = [{"name": "Conv Stack", "layers": ["conv_a", "conv_b"], "padding": 8, "text_spacing": 3}]
    logos = [{"name": "Icon", "file": str(icon_path), "layers": ["conv_a"], "axis": "z", "size": 0.45}]

    img = visualkeras.layered_view(
        model,
        draw_volume=True,
        styles=styles,
        layered_groups=groups,
        logo_groups=logos,
        logos_legend=True,
        legend=True,
        text_callable=lambda i, layer: (layer.name, i % 2 == 0),
        to_file=str(out_path),
        legend_text_spacing_offset=0,
    )

    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0
    assert out_path.exists() and out_path.stat().st_size > 0


def test_layered_view_reversed_image_axes(tmp_path):
    model = _build_conv_model()
    icon_path = tmp_path / "layered_rev_icon.png"
    out_path = tmp_path / "layered_reversed.png"
    _make_icon(icon_path)

    styles = {
        "conv_a": {"image": str(icon_path), "image_fit": "contain", "image_axis": "y"},
        "conv_b": {"image": str(icon_path), "image_fit": "cover", "image_axis": "x"},
        "dense": {"image": str(icon_path), "image_fit": "fill", "image_axis": "z"},
    }
    logos = [
        {"name": "MissingLogo", "file": str(tmp_path / "nope.png"), "layers": ["conv_a"]},
        {"name": "RealLogo", "file": str(icon_path), "layers": ["conv_b"], "axis": "x", "size": (16, 12)},
    ]

    img = visualkeras.layered_view(
        model,
        draw_volume=True,
        draw_reversed=True,
        draw_funnel=True,
        styles=styles,
        logo_groups=logos,
        logos_legend=True,
        to_file=str(out_path),
        legend_text_spacing_offset=0,
    )

    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0
    assert out_path.exists() and out_path.stat().st_size > 0


def test_functional_view_advanced_paths(tmp_path):
    model = _build_dense_model()
    icon_path = tmp_path / "functional_icon.png"
    out_path = tmp_path / "functional_advanced.png"
    _make_icon(icon_path)

    styles = {
        "dense_1": {"image": str(icon_path), "image_fit": "contain", "box_text": "Dense 1"},
        "dense_2": {"image": str(icon_path), "box_text_rotation": 270, "box_text_wrap": "chars"},
    }
    groups = [{"name": "Collapsed Block", "layers": ["dense_1", "dense_2"], "padding": 8, "text_spacing": 4}]
    logos = [
        {
            "name": "Node Logo",
            "file": str(icon_path),
            "layers": ["dense_3"],
            "axis": "z",
            "size": 0.5,
            "corner": "top-right",
        }
    ]

    img = visualkeras.functional_view(
        model,
        draw_volume=True,
        styles=styles,
        simple_text_visualization=True,
        collapse_enabled=True,
        collapse_rules=[
            {
                "kind": "block",
                "selector": [tf.keras.layers.Dense, tf.keras.layers.Dense],
                "repeat_count": 2,
                "label": "4x",
                "annotation_position": "below",
            }
        ],
        text_callable=lambda i, layer: (layer.name, i % 2 == 0),
        layered_groups=groups,
        logo_groups=logos,
        logos_legend=True,
        to_file=str(out_path),
    )

    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.parametrize("mode", ["layered", "graph", "functional"])
def test_group_caption_accepts_freetype_font_objects(mode):
    from PIL import ImageFont

    inputs = tf.keras.Input((4,), name="font_input")
    outputs = tf.keras.layers.Dense(3, name="font_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    group = {"name": "Block", "layers": ["font_output"], "padding": 4, "font_size": 24}
    options = {"padding": 60, "layered_groups": [{**group, "font": "DejaVuSans.ttf"}]}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    reference = visualkeras.show(model, mode=mode, options=options)
    options["layered_groups"] = [{**group, "font": font}]
    image = visualkeras.show(model, mode=mode, options=options)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered", "graph", "functional"])
def test_group_font_resolver_preserves_freetype_object(mode):
    from importlib import import_module
    from PIL import ImageFont

    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    module = import_module("visualkeras." + mode)
    assert module._get_font({"font": font}) is font


@pytest.mark.parametrize("draw_reversed", [False, True])
@pytest.mark.parametrize("font_size", [24, 36])
def test_layered_group_caption_preserves_full_glyph_height(draw_reversed, font_size):
    from PIL import ImageFont

    inputs = tf.keras.Input((4,), name="caption_input")
    outputs = tf.keras.layers.Dense(3, name="caption_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    image = visualkeras.layered_view(
        model, draw_reversed=draw_reversed, legend_text_spacing_offset=0,
        layered_groups=[{"name": "Blockgy", "layers": ["caption_output"],
                         "font": font, "font_color": "magenta"}],
    )
    rows = [index // image.width for index, (r, g, b, a) in enumerate(image.getdata())
            if r == 255 and b == 255 and g < 255]
    bounds = font.getbbox("Blockgy")
    assert max(rows) - min(rows) + 1 == bounds[3] - bounds[1]


@pytest.mark.parametrize("mode", ["graph", "functional"])
@pytest.mark.parametrize("font_size", [24, 36])
def test_group_captions_preserve_full_glyph_height_without_outer_padding(mode, font_size):
    from PIL import ImageFont

    inputs = tf.keras.Input((4,), name="caption_input")
    outputs = tf.keras.layers.Dense(3, name="caption_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    image = visualkeras.show(
        model, mode=mode, padding=0,
        layered_groups=[{"name": "Blockgy", "layers": ["caption_output"],
                         "font": font, "font_color": "magenta"}],
    )
    rows = [i // image.width for i, (r, g, b, a) in enumerate(image.getdata())
            if r == 255 and b == 255 and g < 255]
    bounds = font.getbbox("Blockgy")
    assert max(rows) - min(rows) + 1 == bounds[3] - bounds[1]


@pytest.mark.parametrize("mode", ["layered", "graph", "functional"])
def test_group_font_path_objects_match_string_paths(mode):
    from PIL import ImageFont

    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    path = Path(font.path)
    inputs = tf.keras.Input((4,), name="path_input")
    outputs = tf.keras.layers.Dense(3, name="path_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    group = {"name": "Block", "layers": ["path_output"], "font_size": 24}
    options = {"padding": 60, "layered_groups": [{**group, "font": path}]}
    if mode == "layered":
        options["legend_text_spacing_offset"] = 0
    image = visualkeras.show(model, mode=mode, options=options)
    options["layered_groups"] = [{**group, "font": str(path)}]
    reference = visualkeras.show(model, mode=mode, options=options)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_functional_box_font_path_objects_match_string_paths():
    from PIL import ImageFont

    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    path = Path(font.path)
    inputs = tf.keras.Input((4,), name="path_input")
    outputs = tf.keras.layers.Dense(3, name="path_output")(inputs)
    model = tf.keras.Model(inputs, outputs)
    style = {"box_size": (140, 80), "box_text": "Classify", "box_text_rotation": 0,
             "box_text_font_size": 24, "box_text_font": path}
    settings = {"simple_text_visualization": True, "simple_text_label_mode": "inside"}
    image = visualkeras.functional_view(
        model, **settings, styles={"path_input": {"box_size": (80, 80)}, "path_output": style},
    )
    reference = visualkeras.functional_view(
        model, **settings, styles={"path_input": {"box_size": (80, 80)},
                                  "path_output": {**style, "box_text_font": str(path)}},
    )
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered", "functional", "lenet"])
def test_dictionary_output_shape_metadata_renders(mode):
    class DictionarySplit(tf.keras.layers.Layer):
        def call(self, inputs):
            return {"main": inputs[:, :6], "aux": inputs[:, 6:]}

        @property
        def output_shape(self):
            return {"main": (None, 6), "aux": (None, 2)}

    inputs = tf.keras.Input((8,), name="in")
    outputs = DictionarySplit(name="split")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {}
    if mode in {"layered", "functional"}:
        settings.update(text_callable="name_shape", draw_volume=False)
    if mode == "layered":
        settings.update(spacing=40, legend_text_spacing_offset=0)
    image = visualkeras.show(model, mode=mode, **settings)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("mode", ["layered", "functional"])
@pytest.mark.parametrize("size", [(1000, 1), (1, 1000)])
def test_extreme_aspect_node_logos_fit_the_requested_fraction(tmp_path, mode, size):
    icon = tmp_path / "thin.png"
    Image.new("RGBA", size, "red").save(icon)
    inputs = tf.keras.Input((8,), name="in")
    outputs = tf.keras.layers.Dense(8, name="out")(inputs)
    model = tf.keras.Model(inputs, outputs)
    settings = {"draw_volume": False, "min_xy": 60, "min_z": 60,
                "logo_groups": [{"file": str(icon), "layers": ["out"], "size": 0.5, "corner": "center"}]}
    if mode == "layered":
        settings["legend_text_spacing_offset"] = 0
    image = visualkeras.show(model, mode=mode, **settings)
    points = [(i % image.width, i // image.width) for i, pixel in enumerate(image.getdata()) if pixel[:3] == (255, 0, 0)]
    assert max(x for x, y in points) - min(x for x, y in points) + 1 <= 30
    assert max(y for x, y in points) - min(y for x, y in points) + 1 <= 30


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_overlapping_logo_selectors_paint_each_layer_once(tmp_path, mode):
    icon = tmp_path / "logo.png"
    Image.new("RGBA", (12, 12), (255, 0, 0, 128)).save(icon)
    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(8, name="out")(inputs))
    settings = {"draw_volume": False, "min_xy": 60, "min_z": 60}
    if mode == "layered":
        settings["legend_text_spacing_offset"] = 0
    group = {"file": str(icon), "layers": ["out", tf.keras.layers.Dense, "out"], "corner": "center"}
    image = visualkeras.show(model, mode=mode, logo_groups=[group], **settings)
    reference = visualkeras.show(model, mode=mode, logo_groups=[{**group, "layers": ["out"]}], **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_logo_selectors_accept_layer_instances(tmp_path, mode):
    icon = tmp_path / "logo.png"
    Image.new("RGBA", (12, 12), "red").save(icon)
    inputs = tf.keras.Input((8,), name="in")
    output_layer = tf.keras.layers.Dense(8, name="out")
    model = tf.keras.Model(inputs, output_layer(inputs))
    settings = {"draw_volume": False, "min_xy": 60, "min_z": 60}
    if mode == "layered":
        settings["legend_text_spacing_offset"] = 0
    group = {"file": str(icon), "layers": [output_layer], "corner": "center"}
    image = visualkeras.show(model, mode=mode, logo_groups=[group], **settings)
    reference = visualkeras.show(model, mode=mode, logo_groups=[{**group, "layers": ["out"]}], **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered", "functional"])
def test_logo_class_selectors_include_subclasses(tmp_path, mode):
    class CustomDense(tf.keras.layers.Dense):
        pass

    icon = tmp_path / "logo.png"
    Image.new("RGBA", (12, 12), "red").save(icon)
    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, CustomDense(8, name="out")(inputs))
    settings = {"draw_volume": False, "min_xy": 60, "min_z": 60}
    if mode == "layered":
        settings["legend_text_spacing_offset"] = 0
    group = {"file": str(icon), "layers": [tf.keras.layers.Dense], "corner": "center"}
    image = visualkeras.show(model, mode=mode, logo_groups=[group], **settings)
    reference = visualkeras.show(model, mode=mode, logo_groups=[{**group, "layers": ["out"]}], **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("ignore_hidden", [False, True])
def test_layered_caption_callback_runs_once_per_visible_layer(ignore_hidden):
    inputs = tf.keras.Input((4,), name="in")
    hidden = tf.keras.layers.Dense(3, name="hidden")(inputs)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, name="out")(hidden))
    calls = []

    def caption(index, layer):
        repeated = layer.name in [name for i, name in calls]
        calls.append((index, layer.name))
        return ("Once" if not repeated else "Unexpected second invocation", False)

    settings = {"draw_volume": False, "legend_text_spacing_offset": 0,
                "index_ignore": [1] if ignore_hidden else []}
    image = visualkeras.layered_view(model, text_callable=caption, **settings)
    reference = visualkeras.layered_view(model, text_callable=lambda i, layer: ("Once", False), **settings)
    names = [layer.name for i, layer in enumerate(model.layers) if i not in settings["index_ignore"]]
    assert calls == list(enumerate(names))
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("mode", ["layered"])
@pytest.mark.parametrize("caption", ["Wide\nGroup", "A\n\ng"])
def test_group_captions_preserve_multiline_glyphs(mode, caption):
    from PIL import ImageDraw, ImageFont

    inputs = tf.keras.Input((4,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(3, name="out")(inputs))
    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    settings = {"padding": 2, "layered_groups": [{"name": caption, "layers": ["out"], "font": font, "font_color": "magenta"}]}
    if mode == "layered":
        settings.update(legend_text_spacing_offset=0, draw_volume=False)
    elif mode == "functional":
        settings["draw_volume"] = False
    else:
        settings.update(node_size=20, layer_spacing=40, show_neurons=False)
    image = visualkeras.show(model, mode=mode, **settings)
    reference = Image.new("RGBA", (240, 160), "white")
    ImageDraw.Draw(reference).multiline_text((10, 10), caption, font=font, fill="magenta")

    def glyphs(im):
        mask = Image.new("L", im.size)
        mask.putdata([255 - g if r == b == 255 else 0 for r, g, b, a in im.getdata()])
        return mask.crop(mask.getbbox())

    actual, expected = glyphs(image), glyphs(reference)
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()

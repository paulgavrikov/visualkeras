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

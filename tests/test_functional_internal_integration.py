import pytest
from PIL import Image

import visualkeras.functional as functional


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


def _build_model():
    inputs = tf.keras.layers.Input(shape=(8,), name="input")
    x = tf.keras.layers.Dense(12, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(10, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dense(6, activation="relu", name="dense_3")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="out")(x)
    model = tf.keras.Model(inputs, outputs, name="internal_build_graph_model")
    model.build((None, 8))
    return model


def test_build_graph_internal_branches(tmp_path):
    model = _build_model()
    icon = tmp_path / "icon.png"
    Image.new("RGBA", (30, 18), (0, 160, 220, 255)).save(icon)

    styles = {
        tf.keras.layers.Dense: {"draw_volume": True},
        "dense_1": {
            "image": str(icon),
            "image_fit": "match_aspect",
            "image_axis": "z",
            "scale_image": "bad",
            "box_size": (26, 14),
            "box_min_size": (20, 20),
        },
        "dense_2": {
            "image": str(icon),
            "image_fit": "match_aspect",
            "image_axis": "y",
            "scale_image": 1.1,
        },
        "dense_3": {
            "image": str(icon),
            "image_fit": "match_aspect",
            "image_axis": "x",
            "scale_image": 0.7,
        },
        "out": {
            "image": str(tmp_path / "missing.png"),  # trigger graceful image-load fallback
            "image_fit": "match_aspect",
            "image_axis": "z",
        },
    }

    graph = functional._build_graph(
        model,
        styles=styles,
        global_defaults={"fill": None, "outline": "black"},
        min_z=20,
        min_xy=20,
        max_z=400,
        max_xy=2000,
        scale_z=1.5,
        scale_xy=4.0,
        one_dim_orientation="z",
        sizing_mode="balanced",
        dimension_caps=None,
        relative_base_size=20,
        add_output_nodes=True,
        virtual_node_size=12,
        draw_volume=True,
        shade_step=10,
        image_fit="fill",
        image_axis="z",
        simple_text_visualization=True,
    )

    assert len(graph.nodes) >= len(model.layers)
    assert any(node.kind == "output" for node in graph.nodes.values())
    assert len(graph.edges) > 0
    dense1_node = next(node for node in graph.nodes.values() if node.name == "dense_1")
    assert dense1_node.width >= 20
    assert dense1_node.height >= 20
    assert dense1_node.style.get("draw_volume") is False

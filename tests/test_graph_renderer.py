import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import visualkeras
from visualkeras.options import GraphOptions


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration


DATA_PATH = Path(__file__).resolve().parent / "data" / "renderer_metrics.json"


def _load_all_metrics():
    if not DATA_PATH.exists():
        pytest.skip(
            "Renderer reference metrics missing. Run "
            "`python3 tests/renderer_baselines/generate_renderer_references.py` to regenerate."
        )
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_reference_model():
    inputs = tf.keras.layers.Input((12, 12, 3), name="input")
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same", name="conv")(inputs)
    x = tf.keras.layers.MaxPool2D(2, name="pool")(x)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="dense")(x)
    model = tf.keras.Model(inputs, outputs, name="reference_model")
    model.build((None, 12, 12, 3))
    return model


@pytest.mark.parametrize("via_show", [False, True])
@pytest.mark.parametrize(
    "shape, batch_size, expected_neurons",
    [((3,), None, 3), ((3,), 4, 3), ((2, 2, 3), None, 12)],
)
def test_graph_input_neurons_support_modern_keras(shape, batch_size, expected_neurons, via_show):
    inputs = tf.keras.layers.Input(shape=shape, batch_size=batch_size, name="neuron_input")
    flattened = tf.keras.layers.Flatten(name="neuron_flatten")(inputs)
    outputs = tf.keras.layers.Dense(2, name="neuron_output")(flattened)
    model = tf.keras.Model(inputs, outputs, name="input_neuron_model")
    settings = {
        "inout_as_tensor": False,
        "node_size": 12,
        "node_spacing": 5,
        "layer_spacing": 30,
        "padding": 8,
        "ellipsize_after": 0,
        "styles": {"neuron_input": {"fill": "magenta", "outline": "magenta"}},
    }
    image = visualkeras.show(model, mode="graph", **settings) if via_show else visualkeras.graph_view(model, **settings)
    column = [image.getpixel((14, y))[:3] == (255, 0, 255) for y in range(image.height)]
    runs = sum(value and (index == 0 or not column[index - 1]) for index, value in enumerate(column))
    assert runs == expected_neurons


def _downsample_metrics(image: Image.Image) -> dict[str, float]:
    resized = image.resize((32, 32), Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float64)
    return {
        "checksum": float(arr.sum()),
        "mean": float(arr.mean()),
        "pixels": {
            "0-0": [int(v) for v in arr[0, 0]],
            "15-15": [int(v) for v in arr[15, 15]],
            "31-31": [int(v) for v in arr[31, 31]],
        },
    }


@pytest.mark.parametrize("mode", ["graph", "functional"])
def test_reused_layer_renders_only_current_model_connections(mode):
    first = tf.keras.Input((3,), name="foreign_input")
    shared = tf.keras.layers.Dense(2, name="reused_dense")
    tf.keras.Model(first, shared(first))
    inputs = tf.keras.Input((3,), name="current_input")
    model = tf.keras.Model(inputs, shared(inputs))
    image = visualkeras.show(model, mode=mode, draw_volume=False) if mode == "functional" else visualkeras.show(model, mode=mode, show_neurons=False)
    assert image.width > 0 and image.height > 0


def test_skip_merge_connectors_always_point_to_a_later_column(monkeypatch):
    import visualkeras.graph as graph

    inputs = tf.keras.Input((3,), name="skip_input")
    first = tf.keras.layers.Dense(3, name="long_first")(inputs)
    short = tf.keras.layers.Dense(3, name="short_branch")(inputs)
    second = tf.keras.layers.Dense(3, name="long_second")(first)
    outputs = tf.keras.layers.Add(name="skip_merge")([short, second])
    model = tf.keras.Model(inputs, outputs)
    connections = []
    original = graph._draw_connector

    def record(draw, start, end, *args, **kwargs):
        connections.append((start.x2, end.x1))
        return original(draw, start, end, *args, **kwargs)

    monkeypatch.setattr(graph, "_draw_connector", record)
    image = visualkeras.graph_view(model, show_neurons=False, layer_spacing=50)
    assert len(connections) == 6
    assert all(start < end for start, end in connections)
    assert image.width > 0


def _assert_metrics_close(actual: dict, expected: dict):
    assert actual["checksum"] == pytest.approx(expected["checksum"], rel=1e-4, abs=25000.0)
    assert actual["mean"] == pytest.approx(expected["mean"], rel=1e-4, abs=6.0)
    for key, expected_pixel in expected["pixels"].items():
        actual_pixel = actual["pixels"][key]
        diffs = [abs(a - b) for a, b in zip(actual_pixel, expected_pixel)]
        assert max(diffs) <= 25, f"Pixel {key} differs too much: {actual_pixel} vs {expected_pixel}"


SCENARIOS = [
    (
        "graph_reference",
        partial(
            visualkeras.graph_view,
            show_neurons=False,
            background_fill="white",
        ),
    ),
    (
        "graph_presentation_preset",
        lambda model: visualkeras.show(
            model,
            mode="graph",
            preset="presentation",
            connector_fill="black",
        ),
    ),
    (
        "graph_tensor_nodes",
        partial(
            visualkeras.graph_view,
            show_neurons=True,
            ellipsize_after=6,
            background_fill="white",
        ),
    ),
]


@pytest.mark.parametrize("metric_key, renderer", SCENARIOS)
def test_graph_renderer_variants(metric_key, renderer):
    metrics = _load_all_metrics()
    expected = metrics[metric_key]
    model = _build_reference_model()

    image = renderer(model)

    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0

    actual = _downsample_metrics(image)
    _assert_metrics_close(actual, expected)

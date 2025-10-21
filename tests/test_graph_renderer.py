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


def _assert_metrics_close(actual: dict, expected: dict):
    assert actual["checksum"] == pytest.approx(expected["checksum"], rel=1e-4, abs=150.0)
    assert actual["mean"] == pytest.approx(expected["mean"], rel=1e-4, abs=0.7)
    for key, expected_pixel in expected["pixels"].items():
        actual_pixel = actual["pixels"][key]
        diffs = [abs(a - b) for a, b in zip(actual_pixel, expected_pixel)]
        assert max(diffs) <= 6, f"Pixel {key} differs too much: {actual_pixel} vs {expected_pixel}"


SCENARIOS = [
    (
        "graph_reference",
        partial(
            visualkeras.graph_view,
            show_neurons=False,
            background_fill="white",
        ),
        {"expect_warning": False},
    ),
    (
        "graph_detailed_preset",
        lambda model: visualkeras.show(
            model,
            mode="graph",
            preset="detailed",
            connector_fill="black",
        ),
        {"expect_warning": True, "warning_match": None},
    ),
    (
        "graph_tensor_nodes",
        partial(
            visualkeras.graph_view,
            show_neurons=True,
            ellipsize_after=6,
            background_fill="white",
        ),
        {"expect_warning": False},
    ),
]


@pytest.mark.parametrize("metric_key, renderer, expectations", SCENARIOS)
def test_graph_renderer_variants(metric_key, renderer, expectations):
    metrics = _load_all_metrics()
    expected = metrics[metric_key]
    model = _build_reference_model()

    if expectations.get("expect_warning"):
        match = expectations.get("warning_match")
        with pytest.warns(UserWarning, match=match):
            image = renderer(model)
    else:
        image = renderer(model)

    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0

    actual = _downsample_metrics(image)
    _assert_metrics_close(actual, expected)

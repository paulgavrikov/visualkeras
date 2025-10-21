import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import visualkeras
from visualkeras.options import LayeredOptions


tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")


DATA_PATH = Path(__file__).resolve().parent / "data" / "renderer_metrics.json"

CONV_COLOR_MAP = {
    tf.keras.layers.Conv2D: {"fill": "#ffb347", "outline": "#8c4500"},
    tf.keras.layers.Dense: {"fill": "#8ecae6", "outline": "#023047"},
}


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
    assert actual["checksum"] == pytest.approx(expected["checksum"], rel=1e-4, abs=120.0)
    assert actual["mean"] == pytest.approx(expected["mean"], rel=1e-4, abs=0.5)
    for key, expected_pixel in expected["pixels"].items():
        actual_pixel = actual["pixels"][key]
        diffs = [abs(a - b) for a, b in zip(actual_pixel, expected_pixel)]
        assert max(diffs) <= 5, f"Pixel {key} differs too much: {actual_pixel} vs {expected_pixel}"


SCENARIOS = [
    (
        "layered_reference",
        partial(
            visualkeras.layered_view,
            draw_funnel=False,
            draw_volume=False,
            legend=False,
            background_fill="white",
            type_ignore=[tf.keras.layers.InputLayer],
            legend_text_spacing_offset=0,
        ),
        {"expect_warning": False},
    ),
    (
        "layered_flat_preset",
        lambda model: visualkeras.show(
            model,
            preset="flat",
            spacing=20,
        ),
        {"expect_warning": True, "warning_match": "(legend_text_spacing_offset|many custom)"},
    ),
    (
        "layered_text_name",
        lambda model: visualkeras.show(
            model,
            options=LayeredOptions(draw_volume=False, legend=False, legend_text_spacing_offset=0),
            text_callable="name",
            spacing=15,
        ),
        {"expect_warning": True, "warning_match": "many custom"},
    ),
    (
        "layered_3d_default",
        partial(
            visualkeras.layered_view,
            draw_funnel=True,
            draw_volume=True,
            draw_reversed=False,
            background_fill="white",
            type_ignore=[tf.keras.layers.InputLayer],
            shade_step=14,
            legend=False,
            legend_text_spacing_offset=0,
        ),
        {"expect_warning": False},
    ),
    (
        "layered_3d_reversed",
        partial(
            visualkeras.layered_view,
            draw_funnel=True,
            draw_volume=True,
            draw_reversed=True,
            background_fill="white",
            padding=25,
            type_ignore=[tf.keras.layers.InputLayer],
            shade_step=18,
            legend=False,
            legend_text_spacing_offset=0,
        ),
        {"expect_warning": True, "warning_match": "many custom"},
    ),
    (
        "layered_index2d_relative",
        partial(
            visualkeras.layered_view,
            draw_volume=True,
            index_2D=[2],
            sizing_mode="relative",
            relative_base_size=6,
            dimension_caps={"channels": 140, "sequence": 220, "general": 260},
            background_fill="white",
            type_ignore=[tf.keras.layers.InputLayer],
            legend=False,
            legend_text_spacing_offset=0,
        ),
        {"expect_warning": True, "warning_match": "many custom"},
    ),
    (
        "layered_legend_dimensions",
        lambda model: visualkeras.show(
            model,
            options=LayeredOptions(
                legend=True,
                show_dimension=True,
                draw_volume=True,
                color_map=CONV_COLOR_MAP,
                legend_text_spacing_offset=0,
                type_ignore=(tf.keras.layers.InputLayer,),
            ),
            padding=20,
        ),
        {"expect_warning": True, "warning_match": "many custom"},
    ),
    (
        "layered_ignore_colors",
        partial(
            visualkeras.layered_view,
            draw_volume=True,
            background_fill="white",
            type_ignore=[tf.keras.layers.MaxPool2D],
            index_ignore=[4],
            color_map=CONV_COLOR_MAP,
            one_dim_orientation="y",
            spacing=18,
            shade_step=16,
            legend=False,
            legend_text_spacing_offset=0,
        ),
        {"expect_warning": True, "warning_match": "many custom"},
    ),
]


@pytest.mark.parametrize("metric_key, renderer, expectations", SCENARIOS)
def test_layered_renderer_variants(metric_key, renderer, expectations):
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

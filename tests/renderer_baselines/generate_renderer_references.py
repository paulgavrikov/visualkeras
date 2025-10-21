"""
Helper script to regenerate renderer reference artifacts for tests.

Run with:
    python3 tests/renderer_baselines/generate_renderer_references.py

This will:
    * Build a small convolutional model using tf.keras
    * Render layered and graph visualizations with visualkeras
    * Save the resulting PNG images under tests/renderer_baselines/reference_images/
    * Update tests/data/renderer_metrics.json with downsampled metrics

The renderer tests load these metrics and assert that the current output stays
within a small tolerance. Regenerate the references whenever visuals change.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import visualkeras
from visualkeras.options import LayeredOptions

REF_DIR = Path(__file__).resolve().parent / "reference_images"
DATA_PATH = ROOT / "tests" / "data" / "renderer_metrics.json"

SAMPLE_POINTS = [
    (0, 0),
    (15, 15),
    (31, 31),
]


def build_reference_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input((12, 12, 3), name="input")
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same", name="conv")(inputs)
    x = tf.keras.layers.MaxPool2D(2, name="pool")(x)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same", name="conv2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="dense")(x)
    model = tf.keras.Model(inputs, outputs, name="reference_model")
    model.build((None, 12, 12, 3))
    return model


def compute_metrics(image: Image.Image, tag: str) -> dict[str, float]:
    resized = image.resize((32, 32), Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float64)
    pixels = {
        f"{x}-{y}": [int(channel) for channel in arr[y, x]]
        for (x, y) in SAMPLE_POINTS
    }
    return {
        "checksum": float(arr.sum()),
        "mean": float(arr.mean()),
        "pixels": pixels,
        "tag": tag,
    }


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = build_reference_model()
    for layer in model.layers:
        if not hasattr(layer, "output_shape") or getattr(layer, "output_shape", None) is None:
            try:
                inferred = layer.compute_output_shape(layer.input_shape)
            except Exception:  # noqa: BLE001
                inferred = None
            if inferred is not None:
                setattr(layer, "output_shape", inferred)

    conv_color_map = {
        tf.keras.layers.Conv2D: {"fill": "#ffb347", "outline": "#8c4500"},
        tf.keras.layers.Dense: {"fill": "#8ecae6", "outline": "#023047"},
    }

    scenarios = {
        "layered_reference": lambda: visualkeras.layered_view(
            model,
            draw_funnel=False,
            draw_volume=False,
            legend=False,
            background_fill="white",
            type_ignore=[tf.keras.layers.InputLayer],
            legend_text_spacing_offset=0,
        ),
        "layered_flat_preset": lambda: visualkeras.show(
            model,
            preset="flat",
            spacing=20,
        ),
        "layered_text_name": lambda: visualkeras.show(
            model,
            options=LayeredOptions(legend=False, draw_volume=False, legend_text_spacing_offset=0),
            text_callable="name",
            spacing=15,
        ),
        "layered_3d_default": lambda: visualkeras.layered_view(
            model,
            draw_funnel=True,
            draw_volume=True,
            draw_reversed=False,
            background_fill="white",
            type_ignore=[tf.keras.layers.InputLayer],
            shade_step=14,
            legend=False,
            legend_text_spacing_offset=0,
        ),
        "layered_3d_reversed": lambda: visualkeras.layered_view(
            model,
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
        "layered_index2d_relative": lambda: visualkeras.layered_view(
            model,
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
        "layered_legend_dimensions": lambda: visualkeras.show(
            model,
            options=LayeredOptions(
                legend=True,
                show_dimension=True,
                draw_volume=True,
                color_map=conv_color_map,
                legend_text_spacing_offset=0,
                type_ignore=(tf.keras.layers.InputLayer,),
            ),
            padding=20,
        ),
        "layered_ignore_colors": lambda: visualkeras.layered_view(
            model,
            draw_volume=True,
            background_fill="white",
            type_ignore=[tf.keras.layers.MaxPool2D],
            index_ignore=[4],
            color_map=conv_color_map,
            one_dim_orientation="y",
            spacing=18,
            shade_step=16,
            legend=False,
            legend_text_spacing_offset=0,
        ),
        "graph_reference": lambda: visualkeras.graph_view(
            model,
            show_neurons=False,
            background_fill="white",
        ),
        "graph_detailed_preset": lambda: visualkeras.show(
            model,
            mode="graph",
            preset="detailed",
            connector_fill="black",
        ),
        "graph_tensor_nodes": lambda: visualkeras.graph_view(
            model,
            show_neurons=True,
            ellipsize_after=6,
            background_fill="white",
        ),
    }

    metrics = {}
    for name, render in scenarios.items():
        image = render()
        target_path = REF_DIR / f"{name}.png"
        image.save(target_path)
        metrics[name] = compute_metrics(image, name)

    with DATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    for name in scenarios:
        print(f"{name} image saved to {REF_DIR / f'{name}.png'}")
    print(f"Metrics written to {DATA_PATH}")


if __name__ == "__main__":
    main()

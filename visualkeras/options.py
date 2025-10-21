"""Typed configuration objects and reusable presets for visualkeras renderers.

These dataclasses mirror the existing keyword arguments accepted by
``layered_view`` and ``graph_view`` while providing a typed, documented surface
that is easier to compose, reason about, and share.

The module intentionally avoids importing TensorFlow/Keras at import time so
that configuration can be created even in environments where those heavy
dependencies are unavailable. Any callable or type hints that depend on the
backend therefore use ``typing.Any`` rather than backend-specific types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class LayeredOptions:
    """Configuration bundle for ``layered_view`` rendering."""

    to_file: Optional[str] = None
    min_z: int = 20
    min_xy: int = 20
    max_z: int = 400
    max_xy: int = 2000
    scale_z: float = 1.5
    scale_xy: float = 4.0
    type_ignore: tuple[type, ...] = field(default_factory=tuple)
    index_ignore: tuple[int, ...] = field(default_factory=tuple)
    color_map: Mapping[type, Mapping[str, Any]] = field(default_factory=dict)
    one_dim_orientation: str = "z"
    index_2D: tuple[int, ...] = field(default_factory=tuple)
    background_fill: Any = "white"
    draw_volume: bool = True
    draw_reversed: bool = False
    padding: int = 10
    text_callable: Optional[Callable[[int, Any], Tuple[str, bool]]] = None
    text_vspacing: int = 4
    spacing: int = 10
    draw_funnel: bool = True
    shade_step: int = 10
    legend: bool = False
    legend_text_spacing_offset: int = 15
    font: Any = None
    font_color: Any = "black"
    show_dimension: bool = False
    sizing_mode: str = "accurate"
    dimension_caps: Optional[Mapping[str, int]] = None
    relative_base_size: int = 20

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a shallow dict compatible with ``layered_view``."""
        return {
            "to_file": self.to_file,
            "min_z": self.min_z,
            "min_xy": self.min_xy,
            "max_z": self.max_z,
            "max_xy": self.max_xy,
            "scale_z": self.scale_z,
            "scale_xy": self.scale_xy,
            "type_ignore": tuple(self.type_ignore),
            "index_ignore": tuple(self.index_ignore),
            "color_map": self.color_map,
            "one_dim_orientation": self.one_dim_orientation,
            "index_2D": tuple(self.index_2D),
            "background_fill": self.background_fill,
            "draw_volume": self.draw_volume,
            "draw_reversed": self.draw_reversed,
            "padding": self.padding,
            "text_callable": self.text_callable,
            "text_vspacing": self.text_vspacing,
            "spacing": self.spacing,
            "draw_funnel": self.draw_funnel,
            "shade_step": self.shade_step,
            "legend": self.legend,
            "legend_text_spacing_offset": self.legend_text_spacing_offset,
            "font": self.font,
            "font_color": self.font_color,
            "show_dimension": self.show_dimension,
            "sizing_mode": self.sizing_mode,
            "dimension_caps": self.dimension_caps,
            "relative_base_size": self.relative_base_size,
        }


@dataclass(frozen=True)
class GraphOptions:
    """Configuration bundle for ``graph_view`` rendering."""

    to_file: Optional[str] = None
    color_map: Mapping[type, Mapping[str, Any]] = field(default_factory=dict)
    node_size: int = 50
    background_fill: Any = "white"
    padding: int = 10
    layer_spacing: int = 250
    node_spacing: int = 10
    connector_fill: Any = "gray"
    connector_width: int = 1
    ellipsize_after: int = 10
    inout_as_tensor: bool = True
    show_neurons: bool = True

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a shallow dict compatible with ``graph_view``."""
        return {
            "to_file": self.to_file,
            "color_map": self.color_map,
            "node_size": self.node_size,
            "background_fill": self.background_fill,
            "padding": self.padding,
            "layer_spacing": self.layer_spacing,
            "node_spacing": self.node_spacing,
            "connector_fill": self.connector_fill,
            "connector_width": self.connector_width,
            "ellipsize_after": self.ellipsize_after,
            "inout_as_tensor": self.inout_as_tensor,
            "show_neurons": self.show_neurons,
        }


# --- Text callable templates ----------------------------------------------- #

def _layer_name(index: int, layer: Any) -> str:
    """Return a human-friendly layer name with a fallback when missing."""
    name = getattr(layer, "name", None)
    if not name:
        name = f"layer_{index}"
    return str(name)


def _layer_class(layer: Any) -> str:
    """Return the class name of a layer with a sensible default."""
    try:
        return layer.__class__.__name__
    except AttributeError:
        return "Layer"


def _format_shape_text(layer: Any) -> str:
    """Format the primary output shape of a layer for display."""
    shape = getattr(layer, "output_shape", None)
    if shape is None:
        return "shape: ?"

    try:
        from .layer_utils import extract_primary_shape  # Local import to avoid cycles

        primary_shape = extract_primary_shape(shape, getattr(layer, "name", None))
    except Exception:  # noqa: BLE001 - best-effort formatting
        primary_shape = shape

    return f"shape: {primary_shape}"


LayeredTextCallable = Callable[[int, Any], Tuple[str, bool]]


def text_name_only(index: int, layer: Any) -> Tuple[str, bool]:
    """Show just the layer name beneath the block."""
    return (_layer_name(index, layer), False)


def text_type_and_name(index: int, layer: Any) -> Tuple[str, bool]:
    """Show the layer type and name above the block."""
    layer_type = _layer_class(layer)
    name = _layer_name(index, layer)
    label = f"{layer_type}\n{name}"
    return (label, True)


def text_name_and_shape(index: int, layer: Any) -> Tuple[str, bool]:
    """Show the layer name and formatted shape beneath the block."""
    name = _layer_name(index, layer)
    shape_text = _format_shape_text(layer)
    label = f"{name}\n{shape_text}"
    return (label, False)


LAYERED_TEXT_CALLABLES: Dict[str, LayeredTextCallable] = {
    "name": text_name_only,
    "type_name": text_type_and_name,
    "name_shape": text_name_and_shape,
}
"""Reusable caption generators keyed by a human-friendly identifier."""


# --- Built-in presets ----------------------------------------------------- #

LAYERED_PRESETS: Dict[str, LayeredOptions] = {
    # Mirrors the function defaults used across the README's quick-start.
    "default": LayeredOptions(),
    # Matches the "flat style" example showcased in the documentation.
    "flat": LayeredOptions(draw_volume=False, draw_funnel=False, shade_step=0),
    # Emphasizes clean spacing and 2D presentation for print-ready figures.
    "presentation": LayeredOptions(
        draw_volume=False,
        draw_funnel=True,
        spacing=40,
        padding=30,
        scale_xy=2.5,
        scale_z=1.0,
        legend=True,
        show_dimension=True,
        sizing_mode="balanced",
        text_callable=LAYERED_TEXT_CALLABLES["name_shape"],
    ),
}
"""Curated presets for layered renderings keyed by human-friendly names."""


GRAPH_PRESETS: Dict[str, GraphOptions] = {
    # Mirrors the defaults highlighted in README and usage examples.
    "default": GraphOptions(),
    # Keeps diagrams compact for notebooks or reports with limited space.
    "compact": GraphOptions(layer_spacing=180, node_spacing=8, node_size=40),
    # Reflects the advanced customization showcased in usage_examples.md.
    "detailed": GraphOptions(
        node_size=60,
        connector_width=2,
        layer_spacing=180,
        node_spacing=40,
        padding=40,
        ellipsize_after=8,
    ),
}
"""Curated presets for graph renderings keyed by human-friendly names."""

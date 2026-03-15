"""Typed configuration objects and reusable presets for visualkeras renderers.

These dataclasses mirror the keyword arguments accepted by ``layered_view``,
``graph_view``, ``functional_view``, and ``lenet_view`` while providing a typed,
documented surface that is easier to compose, reason about, and share.

The renderer functions continue to accept plain keyword arguments; options and
presets are an opt-in convenience layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union


StyleMap = Mapping[Union[str, type], Mapping[str, Any]]
TextCallable = Callable[[int, Any], Tuple[str, bool]]


# ---------------------------------------------------------------------------
# Text callable templates
# ---------------------------------------------------------------------------

def _safe_shape(layer: Any) -> Any:
    shape = getattr(layer, "output_shape", None)
    if shape is not None:
        return shape
    output = getattr(layer, "output", None)
    tensor_shape = getattr(output, "shape", None)
    if tensor_shape is not None:
        try:
            return tuple(tensor_shape.as_list())  # TF TensorShape
        except Exception:  # noqa: BLE001
            try:
                return tuple(tensor_shape)
            except Exception:  # noqa: BLE001
                return tensor_shape
    return None


def _format_shape(shape: Any) -> str:
    if shape is None:
        return "?"
    # Multi-output shapes: pick first (best effort).
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if hasattr(shape, "as_list"):
        try:
            shape = tuple(shape.as_list())
        except Exception:  # noqa: BLE001
            pass
    if not isinstance(shape, (list, tuple)):
        return str(shape)
    dims = [d for d in shape[1:] if d is not None]
    if not dims:
        dims = [d for d in shape[1:]]
    if not dims:
        return "?"
    return "×".join(str(d) if d is not None else "?" for d in dims)


def _layer_name(index: int, layer: Any) -> str:
    """Return a human-friendly layer name with a fallback when missing."""
    name = getattr(layer, "name", None)
    if not name:
        name = f"layer_{index}"
    return str(name)


def _layer_type(index: int, layer: Any) -> str:
    return type(layer).__name__


def _layer_shape(index: int, layer: Any) -> str:
    return _format_shape(_safe_shape(layer))


def _layer_name_shape(index: int, layer: Any) -> str:
    return f"{_layer_name(index, layer)}\n{_layer_shape(index, layer)}"


LAYERED_TEXT_CALLABLES: Dict[str, TextCallable] = {
    # (text, above)
    "name": lambda i, layer: (_layer_name(i, layer), False),
    "type": lambda i, layer: (_layer_type(i, layer), False),
    "shape": lambda i, layer: (_layer_shape(i, layer), False),
    "name_shape": lambda i, layer: (_layer_name_shape(i, layer), False),
}


# ---------------------------------------------------------------------------
# Options dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayeredOptions:
    # Mirrors layered_view kwargs (excluding `model`)
    to_file: Optional[str] = None
    min_z: int = 20
    min_xy: int = 20
    max_z: int = 400
    max_xy: int = 2000
    scale_z: float = 1.5
    scale_xy: float = 4.0
    type_ignore: Optional[Sequence[type]] = None
    index_ignore: Optional[Sequence[int]] = None
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None
    one_dim_orientation: str = "z"
    index_2D: Sequence[int] = field(default_factory=tuple)
    background_fill: Any = "white"
    draw_volume: bool = True
    draw_reversed: bool = False
    padding: int = 10
    text_callable: Optional[TextCallable] = None
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
    connector_fill: Any = "gray"
    connector_width: int = 1
    image_fit: str = "fill"
    image_axis: str = "z"
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None
    logo_groups: Optional[Sequence[Dict[str, Any]]] = None
    logos_legend: Union[bool, Dict[str, Any]] = False
    styles: Optional[StyleMap] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GraphOptions:
    to_file: Optional[str] = None
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None
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
    styles: Optional[StyleMap] = None
    image_fit: str = "contain"
    circular_crop: bool = True
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FunctionalOptions:
    to_file: Optional[str] = None
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None
    background_fill: Any = "white"
    padding: int = 20
    column_spacing: int = 80
    row_spacing: int = 40
    component_spacing: int = 80
    connector_fill: Any = "gray"
    connector_width: int = 2
    connector_arrow: bool = False
    connector_padding: int = 5
    min_z: int = 20
    min_xy: int = 20
    max_z: int = 400
    max_xy: int = 2000
    scale_z: float = 1.5
    scale_xy: float = 4.0
    one_dim_orientation: str = "z"
    sizing_mode: str = "balanced"
    dimension_caps: Optional[Mapping[str, int]] = None
    relative_base_size: int = 20
    text_callable: Optional[TextCallable] = None
    text_vspacing: int = 4
    font: Any = None
    font_color: Any = "black"
    add_output_nodes: bool = False
    layout_iterations: int = 4
    virtual_node_size: int = 12
    render_virtual_nodes: bool = False
    draw_volume: bool = False
    orientation_rotation: Optional[float] = None
    shade_step: int = 10
    image_fit: str = "fill"
    image_axis: str = "z"
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None
    logo_groups: Optional[Sequence[Dict[str, Any]]] = None
    logos_legend: Union[bool, Dict[str, Any]] = False
    simple_text_visualization: bool = False
    styles: Optional[StyleMap] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LenetOptions:
    """Configuration for :func:`visualkeras.lenet_view`."""
    to_file: Optional[str] = None
    min_xy: int = 20
    max_xy: int = 220
    scale_xy: float = 4.0
    type_ignore: Optional[Sequence[type]] = None
    index_ignore: Optional[Sequence[int]] = None
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None
    background_fill: Any = "black"
    padding: int = 20
    layer_spacing: int = 40
    map_spacing: int = 4
    max_visual_channels: int = 12
    connector_fill: Any = "gray"
    connector_width: int = 1
    patch_fill: Any = "#7db7ff"
    patch_outline: Any = "black"
    patch_scale: float = 1.0
    patch_alpha_on_image: int = 140
    seed: Optional[int] = None
    draw_connections: bool = True
    draw_patches: bool = True
    font: Any = None
    font_color: Any = "white"
    top_label_padding: int = 6
    bottom_label_padding: int = 6
    top_label: bool = True
    bottom_label: bool = True
    styles: Optional[StyleMap] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

LAYERED_PRESETS: Dict[str, LayeredOptions] = {
    "default": LayeredOptions(),
    "compact": LayeredOptions(spacing=6, padding=6, connector_width=1),
    "presentation": LayeredOptions(
        spacing=18,
        padding=20,
        connector_width=2,
        text_callable=LAYERED_TEXT_CALLABLES["name_shape"],
        legend=True,
    ),
}
"""Curated presets for layered renderings keyed by human-friendly names."""


GRAPH_PRESETS: Dict[str, GraphOptions] = {
    "default": GraphOptions(),
    "compact": GraphOptions(layer_spacing=180, node_size=40),
    "presentation": GraphOptions(layer_spacing=300, node_size=60, connector_width=2),
}
"""Curated presets for graph renderings keyed by human-friendly names."""


FUNCTIONAL_PRESETS: Dict[str, FunctionalOptions] = {
    "default": FunctionalOptions(),
    "compact": FunctionalOptions(column_spacing=60, row_spacing=30, connector_width=1, component_spacing=60),
    "presentation": FunctionalOptions(
        column_spacing=120,
        row_spacing=50,
        connector_width=2,
        component_spacing=100,
        sizing_mode="balanced",
        text_callable=LAYERED_TEXT_CALLABLES["name_shape"],
    ),
}
"""Curated presets for functional renderings keyed by human-friendly names."""


LENET_PRESETS: Dict[str, LenetOptions] = {
    "default": LenetOptions(),
    "compact": LenetOptions(layer_spacing=28, map_spacing=3, max_xy=180, padding=15),
    "presentation": LenetOptions(layer_spacing=55, map_spacing=5, max_xy=260, connector_width=2),
}
"""Curated presets for lenet-style renderings keyed by human-friendly names."""

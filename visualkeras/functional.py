"""Functional (graph-aware) layered renderer for Keras/TensorFlow models.

This renderer targets functional graphs with branches, merges, and multi-input/
output structures. The pipeline is:
1) Graph extraction
2) Rank assignment (longest path)
3) Edge normalization (virtual nodes for long edges)
4) Crossing reduction (barycentric ordering)
5) Component layout + rendering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from collections import deque
import warnings
import re

import aggdraw
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .layer_utils import (
    calculate_layer_dimensions,
    extract_primary_shape,
    find_output_layers,
    get_incoming_layers,
    get_layers,
)
from .options import FunctionalOptions, FUNCTIONAL_PRESETS, LAYERED_TEXT_CALLABLES
from .utils import (
    Box, 
    ColorWheel, 
    fade_color, 
    get_rgba_tuple, 
    resize_image_to_fit, 
    apply_affine_transform,
    draw_node_logo,
    draw_logos_legend
)



@dataclass
class FunctionalNode:
    layer: Any
    node_id: int
    name: str
    layer_type: type
    shape: Optional[Tuple[Any, ...]]
    dims: Tuple[int, int, int]
    width: int
    height: int
    order: int
    rank: int = 0
    rank_order: int = 0
    x: int = 0
    y: int = 0
    kind: str = "layer"  # layer, input, output, virtual, collapsed
    component: int = 0
    style: Dict[str, Any] = field(default_factory=dict)
    de: int = 0
    shade: int = 0
    image: Optional[Image.Image] = None


@dataclass(frozen=True)
class FunctionalEdge:
    src: int
    dst: int


@dataclass
class FunctionalGraph:
    nodes: Dict[int, FunctionalNode]
    edges: List[FunctionalEdge]
    inputs: List[int]
    outputs: List[int]


class _SyntheticLayer:
    """Lightweight placeholder for synthetic input/output/virtual anchors."""

    def __init__(self, name: str, output_shape: Optional[Tuple[Any, ...]] = None) -> None:
        self.name = name
        self.output_shape = output_shape


def _normalize_collapse_selector(kind: str, selector: Any, rule_index: int) -> Union[str, type, Tuple[Union[str, type], ...]]:
    if kind == "layer":
        if isinstance(selector, (str, type)):
            return selector
        raise TypeError(
            f"collapse_rules[{rule_index}]['selector'] must be a layer name (str) or layer type for kind='layer'."
        )

    if kind == "block":
        if not isinstance(selector, Sequence) or isinstance(selector, (str, bytes)):
            raise TypeError(
                f"collapse_rules[{rule_index}]['selector'] must be a sequence of layer names/types for kind='block'."
            )
        normalized: List[Union[str, type]] = []
        for item_index, item in enumerate(selector):
            if isinstance(item, (str, type)):
                normalized.append(item)
                continue
            raise TypeError(
                f"collapse_rules[{rule_index}]['selector'][{item_index}] must be a layer name (str) or layer type."
            )
        if len(normalized) < 2:
            raise ValueError(
                f"collapse_rules[{rule_index}]['selector'] for kind='block' must contain at least 2 entries."
            )
        return tuple(normalized)

    raise ValueError(f"Unsupported collapse rule kind '{kind}'.")


def _validate_and_normalize_collapse_rules(
    collapse_rules: Optional[Sequence[Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    """Validate and normalize explicit collapse rule definitions.

    Rules must be mappings with:
    - ``kind``: ``"layer"`` or ``"block"``
    - ``selector``: a layer name/type (for ``layer``) or sequence of names/types (for ``block``)
    - ``repeat_count``: integer >= 2

    Optional fields:
    - ``label``: string, defaults to ``"{repeat_count}x"``
    - ``annotation_position``: ``"above"`` or ``"below"``, defaults to ``"above"``

    Args:
        collapse_rules: User-provided collapse rules from ``functional_view`` options/kwargs.

    Returns:
        A normalized list of plain dict rules with canonical keys and validated values.

    Raises:
        TypeError: If rule container/items or typed fields have invalid types.
        ValueError: If required keys are missing or values are outside accepted ranges.
    """
    if collapse_rules is None:
        return []
    if not isinstance(collapse_rules, Sequence) or isinstance(collapse_rules, (str, bytes, Mapping)):
        raise TypeError("collapse_rules must be a sequence of mapping rules.")

    normalized_rules: List[Dict[str, Any]] = []
    for rule_index, raw_rule in enumerate(collapse_rules):
        if not isinstance(raw_rule, Mapping):
            raise TypeError(f"collapse_rules[{rule_index}] must be a mapping.")

        kind = str(raw_rule.get("kind", "")).strip().lower()
        if kind not in {"layer", "block"}:
            raise ValueError(
                f"collapse_rules[{rule_index}]['kind'] must be one of: 'layer', 'block'."
            )

        if "selector" not in raw_rule:
            raise ValueError(f"collapse_rules[{rule_index}] is missing required key 'selector'.")
        selector = _normalize_collapse_selector(kind, raw_rule["selector"], rule_index)

        repeat_count = raw_rule.get("repeat_count")
        if not isinstance(repeat_count, int) or repeat_count < 2:
            raise ValueError(
                f"collapse_rules[{rule_index}]['repeat_count'] must be an integer >= 2."
            )

        annotation_position = str(raw_rule.get("annotation_position", "above")).strip().lower()
        if annotation_position not in {"above", "below"}:
            raise ValueError(
                f"collapse_rules[{rule_index}]['annotation_position'] must be 'above' or 'below'."
            )

        label = raw_rule.get("label")
        if label is None:
            label = f"{repeat_count}x"
        elif not isinstance(label, str):
            raise TypeError(f"collapse_rules[{rule_index}]['label'] must be a string when provided.")

        normalized_rules.append(
            {
                "kind": kind,
                "selector": selector,
                "repeat_count": repeat_count,
                "label": label,
                "annotation_position": annotation_position,
            }
        )

    return normalized_rules


def functional_view(
    model,
    to_file: Optional[str] = None,
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None,
    background_fill: Any = "white",
    padding: int = 20,
    column_spacing: int = 80,
    row_spacing: int = 40,
    component_spacing: int = 80,
    connector_fill: Any = "gray",
    connector_width: int = 2,
    connector_arrow: bool = False,
    connector_padding: int = 5,
    min_z: int = 20,
    min_xy: int = 20,
    max_z: int = 400,
    max_xy: int = 2000,
    scale_z: float = 1.5,
    scale_xy: float = 4.0,
    one_dim_orientation: str = "z",
    sizing_mode: str = "balanced",
    dimension_caps: Optional[Mapping[str, int]] = None,
    relative_base_size: int = 20,
    text_callable: Optional[Callable[[int, Any], Tuple[str, bool]]] = None,
    text_vspacing: int = 4,
    font: Optional[ImageFont.ImageFont] = None,
    font_color: Any = "black",
    add_output_nodes: bool = False,
    layout_iterations: int = 4,
    virtual_node_size: int = 12,
    render_virtual_nodes: bool = False,
    draw_volume: bool = False,
    orientation_rotation: Optional[float] = None,
    shade_step: int = 10,
    image_fit: str = "fill",
    image_axis: str = "z",
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None,
    logo_groups: Optional[Sequence[Dict[str, Any]]] = None,
    logos_legend: Union[bool, Dict[str, Any]] = False,
    styles: Optional[Mapping[Union[str, type], Dict[str, Any]]] = None, 
    *,
    simple_text_visualization: bool = False,
    simple_text_label_mode: str = "below",
    collapse_enabled: bool = False,
    collapse_rules: Optional[Sequence[Mapping[str, Any]]] = None,
    collapse_annotations: bool = True,
    options: Union[FunctionalOptions, Mapping[str, Any], None] = None,
    preset: Union[str, None] = None,
) -> Image.Image:
    """Render a functional model using a multi-stream layered layout.
    
    :param layered_groups: List of dicts defining groups of layers to highlight with a background rectangle.
                           Each dict can contain:
                           - 'layers': List of layer names or objects.
                           - 'fill': Background color (default: semi-transparent gray).
                           - 'outline': Border color (default: black).
                           - 'width': Border thickness (default: 1).
                           - 'padding': Padding around the group (default: 10).
    :param logo_groups: List of dicts defining groups of layers to add logos to.
                        Each dict can contain:
                        - 'name': Name of the group (for legend).
                        - 'file': Path to the logo image.
                        - 'axis': Axis orientation ('x', 'y', 'z'). Default 'z'.
                        - 'size': Tuple (w, h) or float (scale relative to face min dimension).
                        - 'corner': Corner position ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center').
                        - 'layers': List of layer names or types.
    :param logos_legend: Boolean or dict to configure the logo legend.
    """
    using_presets = options is not None or preset is not None

    if not using_presets:
        defaults = FunctionalOptions().to_kwargs()
        current_params = {
            "to_file": to_file,
            "color_map": color_map,
            "background_fill": background_fill,
            "padding": padding,
            "column_spacing": column_spacing,
            "row_spacing": row_spacing,
            "component_spacing": component_spacing,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "connector_arrow": connector_arrow,
            "connector_padding": connector_padding,
            "min_z": min_z,
            "min_xy": min_xy,
            "max_z": max_z,
            "max_xy": max_xy,
            "scale_z": scale_z,
            "scale_xy": scale_xy,
            "one_dim_orientation": one_dim_orientation,
            "sizing_mode": sizing_mode,
            "dimension_caps": dimension_caps,
            "relative_base_size": relative_base_size,
            "text_callable": text_callable,
            "text_vspacing": text_vspacing,
            "font": font,
            "font_color": font_color,
            "add_output_nodes": add_output_nodes,
            "layout_iterations": layout_iterations,
            "virtual_node_size": virtual_node_size,
            "render_virtual_nodes": render_virtual_nodes,
            "draw_volume": draw_volume,
            "orientation_rotation": orientation_rotation,
            "shade_step": shade_step,
            "image_fit": image_fit,
            "image_axis": image_axis,
            "layered_groups": layered_groups,
            "logo_groups": logo_groups,
            "logos_legend": logos_legend,
            "simple_text_visualization": simple_text_visualization,
            "simple_text_label_mode": simple_text_label_mode,
            "collapse_enabled": collapse_enabled,
            "collapse_rules": collapse_rules,
            "collapse_annotations": collapse_annotations,
            "styles": styles,
        }
        custom_keys = [
            key for key, value in current_params.items()
            if key in defaults and value != defaults[key]
        ]
        if len(custom_keys) >= 5:
            warnings.warn(
                "functional_view received many custom keyword arguments. "
                "Consider using visualkeras.show(..., mode='functional', preset=...) "
                "and the FunctionalOptions dataclass for a simpler workflow.",
                UserWarning,
                stacklevel=2,
            )

    if preset is not None or options is not None:
        defaults = FunctionalOptions().to_kwargs()
        defaults["color_map"] = None
        defaults["dimension_caps"] = None
        defaults["font"] = None
        defaults["styles"] = None
        defaults["orientation_rotation"] = None
        defaults["image_fit"] = "fill"
        defaults["image_axis"] = "z"
        defaults["layered_groups"] = None
        defaults["logo_groups"] = None
        defaults["logos_legend"] = False
        defaults["simple_text_visualization"] = False

        resolved = dict(defaults)
        if preset is not None:
            try:
                resolved.update(FUNCTIONAL_PRESETS[preset].to_kwargs())
            except KeyError as exc:
                available = ", ".join(sorted(FUNCTIONAL_PRESETS.keys()))
                raise ValueError(
                    f"Unknown functional preset '{preset}'. Available presets: {available}"
                ) from exc

        if options is not None:
            if isinstance(options, FunctionalOptions):
                option_values = options.to_kwargs()
            elif isinstance(options, Mapping):
                option_values = dict(options)
            else:
                raise TypeError(
                    "options must be a FunctionalOptions instance or a mapping of keyword arguments."
                )
            resolved.update(option_values)

        explicit_values = {
            "to_file": to_file,
            "color_map": color_map,
            "background_fill": background_fill,
            "padding": padding,
            "column_spacing": column_spacing,
            "row_spacing": row_spacing,
            "component_spacing": component_spacing,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "connector_arrow": connector_arrow,
            "connector_padding": connector_padding,
            "min_z": min_z,
            "min_xy": min_xy,
            "max_z": max_z,
            "max_xy": max_xy,
            "scale_z": scale_z,
            "scale_xy": scale_xy,
            "one_dim_orientation": one_dim_orientation,
            "sizing_mode": sizing_mode,
            "dimension_caps": dimension_caps,
            "relative_base_size": relative_base_size,
            "text_callable": text_callable,
            "text_vspacing": text_vspacing,
            "font": font,
            "font_color": font_color,
            "add_output_nodes": add_output_nodes,
            "layout_iterations": layout_iterations,
            "virtual_node_size": virtual_node_size,
            "render_virtual_nodes": render_virtual_nodes,
            "draw_volume": draw_volume,
            "orientation_rotation": orientation_rotation,
            "shade_step": shade_step,
            "image_fit": image_fit,
            "image_axis": image_axis,
            "layered_groups": layered_groups,
            "logo_groups": logo_groups,
            "logos_legend": logos_legend,
            "simple_text_visualization": simple_text_visualization,
            "simple_text_label_mode": simple_text_label_mode,
            "collapse_enabled": collapse_enabled,
            "collapse_rules": collapse_rules,
            "collapse_annotations": collapse_annotations,
            "styles": styles,
        }

        for key, value in explicit_values.items():
            if key not in defaults:
                continue
            if value != defaults[key]:
                resolved[key] = value

        to_file = resolved["to_file"]
        color_map = resolved["color_map"]
        background_fill = resolved["background_fill"]
        padding = resolved["padding"]
        column_spacing = resolved["column_spacing"]
        row_spacing = resolved["row_spacing"]
        component_spacing = resolved["component_spacing"]
        connector_fill = resolved["connector_fill"]
        connector_width = resolved["connector_width"]
        connector_arrow = resolved["connector_arrow"]
        connector_padding = resolved["connector_padding"]
        min_z = resolved["min_z"]
        min_xy = resolved["min_xy"]
        max_z = resolved["max_z"]
        max_xy = resolved["max_xy"]
        scale_z = resolved["scale_z"]
        scale_xy = resolved["scale_xy"]
        one_dim_orientation = resolved["one_dim_orientation"]
        sizing_mode = resolved["sizing_mode"]
        dimension_caps = resolved["dimension_caps"]
        relative_base_size = resolved["relative_base_size"]
        text_callable = resolved["text_callable"]
        text_vspacing = resolved["text_vspacing"]
        font = resolved["font"]
        font_color = resolved["font_color"]
        add_output_nodes = resolved["add_output_nodes"]
        layout_iterations = resolved["layout_iterations"]
        virtual_node_size = resolved["virtual_node_size"]
        render_virtual_nodes = resolved["render_virtual_nodes"]
        draw_volume = resolved["draw_volume"]
        orientation_rotation = resolved.get("orientation_rotation", orientation_rotation)
        shade_step = resolved.get("shade_step", shade_step)
        image_fit = resolved.get("image_fit", image_fit)
        image_axis = resolved.get("image_axis", image_axis)
        layered_groups = resolved.get("layered_groups", layered_groups)
        logo_groups = resolved.get("logo_groups", logo_groups)
        logos_legend = resolved.get("logos_legend", logos_legend)
        simple_text_visualization = resolved.get("simple_text_visualization", simple_text_visualization)
        simple_text_label_mode = resolved.get("simple_text_label_mode", simple_text_label_mode)
        collapse_enabled = resolved.get("collapse_enabled", collapse_enabled)
        collapse_rules = resolved.get("collapse_rules", collapse_rules)
        collapse_annotations = resolved.get("collapse_annotations", collapse_annotations)
        styles = resolved["styles"]

        if simple_text_visualization:
            draw_volume = False
            orientation_rotation = None

        if color_map is not None and not isinstance(color_map, dict):
            color_map = dict(color_map)
        if dimension_caps is not None and not isinstance(dimension_caps, dict):
            dimension_caps = dict(dimension_caps)

    simple_text_label_mode = str(simple_text_label_mode or "below").strip().lower()
    if simple_text_label_mode not in {"inside", "below"}:
        raise ValueError(
            "simple_text_label_mode must be one of: 'inside', 'below'."
        )

    if isinstance(text_callable, str):
        try:
            text_callable = LAYERED_TEXT_CALLABLES[text_callable]
        except KeyError as exc:
            available = ", ".join(sorted(LAYERED_TEXT_CALLABLES))
            raise ValueError(
                f"Unknown text callable preset '{text_callable}'. "
                f"Available presets: {available}"
            ) from exc

    if color_map is None:
        color_map = {}

    if styles is not None and not isinstance(styles, dict):
        styles = dict(styles)

    if styles is None:
        styles = {}

    if simple_text_visualization and simple_text_label_mode == "below":
        if text_callable is None:
            text_callable = LAYERED_TEXT_CALLABLES["name_shape"]
        root_style = styles.get(object)
        if root_style is None:
            styles[object] = {"box_text_enabled": False}
        elif isinstance(root_style, Mapping):
            root_style_copy = dict(root_style)
            root_style_copy.setdefault("box_text_enabled", False)
            styles[object] = root_style_copy

    normalized_collapse_rules = _validate_and_normalize_collapse_rules(collapse_rules)

    global_defaults = {
        "connector_fill": connector_fill,
        "connector_width": connector_width,
        "connector_arrow": connector_arrow,
        "connector_padding": connector_padding,
        "draw_volume": draw_volume,
        "orientation_rotation": orientation_rotation,
        "shade_step": shade_step,
        "image_fit": image_fit,
        "image_axis": image_axis,
        "padding": 0,  # separate from global image padding

        "box_orientation": "vertical",
        "box_text_rotation": None,
        "box_text_color": font_color,
        "box_text_font": None,
        "box_text_font_size": 14,
        "box_text_padding": 8,
        "box_text_wrap": "words",
        "box_text_autoshrink": True,
        "box_text_min_font_size": 8,
        "box_text_align": "center",
        "box_text_valign": "middle",
        "box_outline_width": 2,
        "box_fill": None,
        "box_outline": None,
        "box_text_enabled": True,

        "collapse_badge_enabled": True,
        "collapse_badge_fill": "white",
        "collapse_badge_outline": "black",
        "collapse_badge_text_color": font_color,
        "collapse_badge_font": None,
        "collapse_badge_font_size": 12,
        "collapse_badge_padding": (4, 2),
        "collapse_annotation_color": connector_fill,
        "collapse_annotation_font": None,
        "collapse_annotation_font_size": 12,
        "collapse_annotation_offset": 10,
        "collapse_annotation_width": 2,
        "collapse_annotation_head_size": 6,
    }

    graph = _build_graph(
        model,
        styles=styles,
        global_defaults=global_defaults,
        min_z=min_z,
        min_xy=min_xy,
        max_z=max_z,
        max_xy=max_xy,
        scale_z=scale_z,
        scale_xy=scale_xy,
        one_dim_orientation=one_dim_orientation,
        sizing_mode=sizing_mode,
        dimension_caps=dimension_caps,
        relative_base_size=relative_base_size,
        add_output_nodes=add_output_nodes,
        virtual_node_size=virtual_node_size,
        draw_volume=draw_volume,
        shade_step=shade_step,
        image_fit=image_fit,
        image_axis=image_axis,
        simple_text_visualization=simple_text_visualization,
    )

    if collapse_enabled and normalized_collapse_rules:
        graph, rule_apply_counts = _collapse_graph_with_rules(
            graph,
            normalized_collapse_rules,
            collapse_annotations=collapse_annotations,
        )
        for rule_index, apply_count in enumerate(rule_apply_counts):
            if apply_count == 0:
                warnings.warn(
                    f"collapse_rules[{rule_index}] did not match any collapsible linear chain.",
                    UserWarning,
                    stacklevel=2,
                )

    ranks = _assign_ranks(graph.nodes, graph.edges)
    graph, ranks = _expand_long_edges(graph, ranks, virtual_node_size)

    if graph.nodes:
        _mark_inputs_outputs(graph)

    text_top_padding: Dict[int, int] = {}
    text_bottom_padding: Dict[int, int] = {}
    if simple_text_visualization and simple_text_label_mode == "below" and text_callable is not None:
        text_top_padding, text_bottom_padding = _compute_external_text_padding(
            graph,
            text_callable=text_callable,
            text_vspacing=text_vspacing,
            font=font,
        )

    components = _split_components(graph.nodes, graph.edges)
    components.sort(key=lambda comp: _component_sort_key(graph, comp))

    column_widths = _column_widths(graph.nodes, ranks)
    x_positions = _column_positions(column_widths, padding, column_spacing)

    y_offset = padding
    for component_index, node_ids in enumerate(components):
        rank_nodes = _order_by_barycenter(
            graph,
            node_ids,
            ranks,
            iterations=layout_iterations,
        )
        for node_id in node_ids:
            graph.nodes[node_id].component = component_index
        component_height = _assign_component_positions(
            graph,
            node_ids,
            rank_nodes,
            x_positions,
            column_widths,
            y_offset,
            row_spacing,
            node_top_padding=text_top_padding,
            node_bottom_padding=text_bottom_padding,
        )
        if component_height <= 0:
            continue
        y_offset += component_height + component_spacing

    _straighten_layout(
        graph,
        ranks,
        row_spacing,
        node_top_padding=text_top_padding,
        node_bottom_padding=text_bottom_padding,
    )

    img = _render_graph(
        graph,
        color_map=color_map,
        background_fill=background_fill,
        padding=padding,
        connector_fill=connector_fill,
        connector_width=connector_width,
        connector_arrow=connector_arrow,
        connector_padding=connector_padding,
        text_callable=text_callable,
        text_vspacing=text_vspacing,
        font=font,
        font_color=font_color,
        render_virtual_nodes=render_virtual_nodes,
        draw_volume=draw_volume,
        orientation_rotation=orientation_rotation,
        layered_groups=layered_groups,
        logo_groups=logo_groups,
        logos_legend=logos_legend,
        simple_text_visualization=simple_text_visualization,
        external_text_bottom_padding=text_bottom_padding,
    )

    if to_file is not None:
        img.save(to_file)
    return img


def _build_graph(
    model,
    *,
    styles: Mapping[Union[str, type], Dict[str, Any]],
    global_defaults: Dict[str, Any],
    min_z: int,
    min_xy: int,
    max_z: int,
    max_xy: int,
    scale_z: float,
    scale_xy: float,
    one_dim_orientation: str,
    sizing_mode: str,
    dimension_caps: Optional[Mapping[str, int]],
    relative_base_size: int,
    add_output_nodes: bool,
    virtual_node_size: int,
    draw_volume: bool,
    shade_step: int,
    image_fit: str,
    image_axis: str,
    simple_text_visualization: bool = False,
) -> FunctionalGraph:
    """Build the intermediate functional graph from a Keras model.

    This stage creates ``FunctionalNode`` objects for each model layer, applies
    style/default resolution, computes visual dimensions, optionally loads image
    textures, and collects directed graph edges from inbound layer relations.
    When requested, synthetic output nodes are appended.

    Args:
        model: Keras/TensorFlow model.
        styles: Style overrides keyed by layer type or layer name.
        global_defaults: Base style values merged into each node style.
        min_z: Minimum depth dimension used in dimension scaling.
        min_xy: Minimum width/height dimension used in scaling.
        max_z: Maximum depth dimension used in scaling.
        max_xy: Maximum width/height dimension used in scaling.
        scale_z: Depth scaling factor.
        scale_xy: Width/height scaling factor.
        one_dim_orientation: Orientation hint for one-dimensional shapes.
        sizing_mode: Dimension scaling strategy.
        dimension_caps: Optional dimension caps for capped/balanced sizing modes.
        relative_base_size: Base pixel size used by relative sizing mode.
        add_output_nodes: Whether to append synthetic output marker nodes.
        virtual_node_size: Size used for synthetic helper nodes.
        draw_volume: Global default for volumetric rendering depth.
        shade_step: Global shade delta for box rendering.
        image_fit: Default image fit mode for textured nodes.
        image_axis: Default box face axis for image projection.
        simple_text_visualization: If true, force flat 2D boxes.

    Returns:
        A ``FunctionalGraph`` containing nodes, edges, and inferred input/output ids.
    """
    
    def resolve_style(layer, name) -> Dict[str, Any]:
        final_style = global_defaults.copy()
        for cls in type(layer).__mro__:
            if cls in styles:
                final_style.update(styles[cls])
        if name in styles:
            final_style.update(styles[name])
        return final_style
    
    layers = list(get_layers(model))
    order_map = {id(layer): index for index, layer in enumerate(layers)}
    nodes: Dict[int, FunctionalNode] = {}

    for layer in layers:
        node_id = id(layer)
        name = getattr(layer, "name", None) or f"layer_{order_map[node_id]}"
        node_style = resolve_style(layer, name)

        if simple_text_visualization:
            node_style = dict(node_style)
            node_style["draw_volume"] = False

        image_path = node_style.get("image")
        node_image = None
        shape = extract_primary_shape(_resolve_layer_output_shape(layer), name)
        dims = calculate_layer_dimensions(
            shape,
            scale_z,
            scale_xy,
            max_z,
            max_xy,
            min_z,
            min_xy,
            one_dim_orientation=one_dim_orientation,
            sizing_mode=sizing_mode,
            dimension_caps=dimension_caps,
            relative_base_size=relative_base_size,
        )
        width = max(min_xy, int(dims[2]))
        height = max(min_xy, int(dims[1]))

        if simple_text_visualization:
            box_size = node_style.get("box_size")
            if isinstance(box_size, (tuple, list)) and len(box_size) == 2:
                try:
                    width = int(box_size[0])
                    height = int(box_size[1])
                except Exception:
                    pass
            box_min = node_style.get("box_min_size")
            if isinstance(box_min, (tuple, list)) and len(box_min) == 2:
                try:
                    width = max(width, int(box_min[0]))
                    height = max(height, int(box_min[1]))
                except Exception:
                    pass

        use_volume = node_style.get('draw_volume', draw_volume)
        if simple_text_visualization:
            use_volume = False
        de = 0
        if use_volume:
            de = int(width / 3)

        if image_path:
            try:
                node_image = Image.open(image_path).convert("RGBA")
                fit_mode = node_style.get("image_fit", image_fit)
                axis = ("z" if simple_text_visualization else node_style.get("image_axis", image_axis))
                if fit_mode == "match_aspect":
                    img_w, img_h = node_image.size
                    img_ratio = img_w / img_h

                    if axis == 'z':
                        surf_ratio = width / height
                        if img_ratio > surf_ratio:
                            width = int(height * img_ratio)
                        else:
                            height = int(width / img_ratio)

                    elif axis == 'y':
                        if img_ratio > 0:
                            de = int(width / img_ratio)

                    elif axis == 'x':
                        de = int(height * img_ratio)

                scale_factor = node_style.get("scale_image")
                if scale_factor is not None:
                    try:
                        scale_factor = float(scale_factor)
                        if scale_factor < 0:
                            scale_factor = 0.0
                    except (ValueError, TypeError):
                        scale_factor = 1.0
                    
                    if axis == 'z': # Front (Width x Height)
                        width = int(width * scale_factor)
                        height = int(height * scale_factor)
                    elif axis == 'y': # Top (Width x Depth)
                        width = int(width * scale_factor)
                        de = int(de * scale_factor)
                    elif axis == 'x': # Side (Depth x Height)
                        de = int(de * scale_factor)
                        height = int(height * scale_factor)

            except Exception as e:
                warnings.warn(f"Failed to load image for layer '{name}': {e}. Reverting to default visualization.")
                image_path = None # Fallback to standard logic below

        total_width = width + de
        total_height = height + de
        
        shade = node_style.get('shade_step', shade_step)

        nodes[node_id] = FunctionalNode(
            layer=layer,
            node_id=node_id,
            name=name,
            layer_type=type(layer),
            shape=extract_primary_shape(_resolve_layer_output_shape(layer), name) if not image_path else None,
            dims=(int(dims[0]), int(dims[1]), int(dims[2])),
            width=total_width,
            height=total_height,
            order=order_map[node_id],
            style=node_style,
            de=de,
            shade=shade,
            image=node_image
        )

    edges = _collect_edges(nodes)

    if add_output_nodes:
        nodes, edges = _attach_output_nodes(model, nodes, edges, virtual_node_size)

    inputs, outputs = _find_inputs_outputs(nodes, edges)
    return FunctionalGraph(nodes=nodes, edges=edges, inputs=inputs, outputs=outputs)


def _collect_edges(nodes: Dict[int, FunctionalNode]) -> List[FunctionalEdge]:
    edges: List[FunctionalEdge] = []
    seen = set()
    for node in nodes.values():
        for inbound in get_incoming_layers(node.layer):
            inbound_id = id(inbound)
            if inbound_id not in nodes:
                continue
            edge = (inbound_id, node.node_id)
            if edge in seen:
                continue
            edges.append(FunctionalEdge(*edge))
            seen.add(edge)
    return edges


def _attach_output_nodes(
    model,
    nodes: Dict[int, FunctionalNode],
    edges: List[FunctionalEdge],
    virtual_node_size: int,
) -> Tuple[Dict[int, FunctionalNode], List[FunctionalEdge]]:
    order_base = max((node.order for node in nodes.values()), default=0) + 1
    new_nodes = dict(nodes)
    new_edges = list(edges)
    outputs = list(find_output_layers(model))

    output_size = max(virtual_node_size * 2, 16)

    for index, layer in enumerate(outputs):
        name = f"output_{index}"
        synthetic = _SyntheticLayer(name=name, output_shape=getattr(layer, "output_shape", None))
        node_id = id(synthetic)
        new_nodes[node_id] = FunctionalNode(
            layer=synthetic,
            node_id=node_id,
            name=name,
            layer_type=type(synthetic),
            shape=extract_primary_shape(getattr(synthetic, "output_shape", None), name),
            dims=(output_size, output_size, output_size),
            width=output_size,
            height=output_size,
            order=order_base + index,
            kind="output",
        )
        if id(layer) in nodes:
            new_edges.append(FunctionalEdge(id(layer), node_id))

    return new_nodes, new_edges


def _find_inputs_outputs(
    nodes: Dict[int, FunctionalNode],
    edges: Sequence[FunctionalEdge],
) -> Tuple[List[int], List[int]]:
    incoming = {node_id: 0 for node_id in nodes}
    outgoing = {node_id: 0 for node_id in nodes}
    for edge in edges:
        incoming[edge.dst] += 1
        outgoing[edge.src] += 1
    inputs = [node_id for node_id, count in incoming.items() if count == 0]
    outputs = [node_id for node_id, count in outgoing.items() if count == 0]
    return inputs, outputs


def _build_edge_index(
    nodes: Mapping[int, FunctionalNode],
    edges: Sequence[FunctionalEdge],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    outgoing: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
    incoming: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        if edge.src not in nodes or edge.dst not in nodes:
            continue
        outgoing[edge.src].append(edge.dst)
        incoming[edge.dst].append(edge.src)
    return outgoing, incoming


def _node_matches_collapse_selector(node: FunctionalNode, selector: Union[str, type]) -> bool:
    if isinstance(selector, str):
        return node.name == selector
    try:
        return issubclass(node.layer_type, selector)
    except TypeError:
        return False


def _find_first_collapse_sequence(
    graph: FunctionalGraph,
    rule: Mapping[str, Any],
) -> Optional[List[int]]:
    """Find the first collapsible linear node sequence matching one rule.

    Matching is strict and linear: each internal hop must traverse a node with
    exactly one outgoing edge and a successor with exactly one incoming edge
    from that predecessor. This avoids collapsing ambiguous branch/merge paths.

    Args:
        graph: Current graph state.
        rule: One normalized collapse rule.

    Returns:
        A list of node ids for the first valid match in node order, or ``None``
        if no sequence satisfies the rule.
    """
    kind = str(rule["kind"])
    repeat_count = int(rule["repeat_count"])
    if kind == "layer":
        selector_pattern: List[Union[str, type]] = [rule["selector"]] * repeat_count
    else:
        block = list(rule["selector"])
        selector_pattern = block * repeat_count

    if len(selector_pattern) < 2:
        return None

    outgoing, incoming = _build_edge_index(graph.nodes, graph.edges)
    candidate_ids = sorted(graph.nodes, key=lambda node_id: graph.nodes[node_id].order)

    for start_id in candidate_ids:
        start_node = graph.nodes[start_id]
        if start_node.kind != "layer":
            continue
        if not _node_matches_collapse_selector(start_node, selector_pattern[0]):
            continue

        sequence = [start_id]
        cursor = start_id
        valid = True

        for expected_selector in selector_pattern[1:]:
            out_nodes = outgoing.get(cursor, [])
            if len(out_nodes) != 1:
                valid = False
                break
            next_id = out_nodes[0]
            if next_id in sequence:
                valid = False
                break
            next_node = graph.nodes.get(next_id)
            if next_node is None or next_node.kind != "layer":
                valid = False
                break
            in_nodes = incoming.get(next_id, [])
            if len(in_nodes) != 1 or in_nodes[0] != cursor:
                valid = False
                break
            if not _node_matches_collapse_selector(next_node, expected_selector):
                valid = False
                break
            sequence.append(next_id)
            cursor = next_id

        if valid:
            return sequence

    return None


def _collapse_node_sequence(
    graph: FunctionalGraph,
    sequence: Sequence[int],
    *,
    rule: Mapping[str, Any],
    collapse_annotations: bool,
) -> FunctionalGraph:
    """Collapse one matched node sequence into a synthetic collapsed node.

    The function removes all sequence members, inserts one synthetic node with
    merged metadata/style markers, and rewires boundary edges so incoming edges
    target the collapsed node and outgoing edges originate from it.

    Args:
        graph: Source graph.
        sequence: Ordered node ids to collapse.
        rule: Normalized rule that produced the match.
        collapse_annotations: Whether block-level annotation rendering is enabled.

    Returns:
        A new ``FunctionalGraph`` with collapsed topology and recomputed
        input/output sets.
    """
    if not sequence:
        return graph

    seq_ids = list(sequence)
    seq_set = set(seq_ids)
    old_nodes = graph.nodes
    member_nodes = [old_nodes[node_id] for node_id in seq_ids]
    first_node = member_nodes[0]
    last_node = member_nodes[-1]

    collapse_label = str(rule.get("label", f"{rule['repeat_count']}x"))
    collapse_kind = str(rule["kind"])
    synthetic_name = f"collapsed_{first_node.name}_{collapse_label}"
    synthetic_layer = _SyntheticLayer(
        name=synthetic_name,
        output_shape=getattr(last_node.layer, "output_shape", None),
    )
    collapsed_node_id = id(synthetic_layer)
    while collapsed_node_id in old_nodes:
        synthetic_layer = _SyntheticLayer(
            name=f"{synthetic_name}_{collapsed_node_id}",
            output_shape=getattr(last_node.layer, "output_shape", None),
        )
        collapsed_node_id = id(synthetic_layer)

    collapsed_style = dict(first_node.style or {})
    collapsed_style["collapsed"] = True
    collapsed_style["collapse_kind"] = collapse_kind
    collapsed_style["collapse_repeat_count"] = int(rule["repeat_count"])
    collapsed_style["collapse_label"] = collapse_label
    collapsed_style["collapse_annotation_position"] = rule.get("annotation_position", "above")
    collapsed_style["collapse_annotation_enabled"] = bool(collapse_annotations)
    collapsed_style["collapse_members"] = tuple(node.name for node in member_nodes)
    if collapse_kind == "block":
        selector = rule.get("selector", ())
        collapsed_style["collapse_block_size"] = len(selector) if isinstance(selector, tuple) else 0

    collapsed_node = FunctionalNode(
        layer=synthetic_layer,
        node_id=collapsed_node_id,
        name=synthetic_name,
        layer_type=first_node.layer_type,
        shape=last_node.shape,
        dims=(
            max(node.dims[0] for node in member_nodes),
            max(node.dims[1] for node in member_nodes),
            max(node.dims[2] for node in member_nodes),
        ),
        width=max(node.width for node in member_nodes),
        height=max(node.height for node in member_nodes),
        order=first_node.order,
        rank=first_node.rank,
        rank_order=first_node.rank_order,
        kind="collapsed",
        component=first_node.component,
        style=collapsed_style,
        de=max(node.de for node in member_nodes),
        shade=first_node.shade,
        image=first_node.image if collapse_kind == "layer" else None,
    )

    new_nodes = {node_id: node for node_id, node in old_nodes.items() if node_id not in seq_set}
    new_nodes[collapsed_node_id] = collapsed_node

    first_id = seq_ids[0]
    last_id = seq_ids[-1]
    seen_edges = set()
    new_edges: List[FunctionalEdge] = []
    for edge in graph.edges:
        src = edge.src
        dst = edge.dst

        if src in seq_set and dst in seq_set:
            continue
        if src in seq_set:
            if src != last_id:
                continue
            src = collapsed_node_id
        if dst in seq_set:
            if dst != first_id:
                continue
            dst = collapsed_node_id
        if src == dst:
            continue
        edge_key = (src, dst)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        new_edges.append(FunctionalEdge(src, dst))

    inputs, outputs = _find_inputs_outputs(new_nodes, new_edges)
    return FunctionalGraph(nodes=new_nodes, edges=new_edges, inputs=inputs, outputs=outputs)


def _collapse_graph_with_rules(
    graph: FunctionalGraph,
    rules: Sequence[Mapping[str, Any]],
    *,
    collapse_annotations: bool,
) -> Tuple[FunctionalGraph, List[int]]:
    """Apply collapse rules repeatedly until no more matches remain.

    Rules are processed in order. For each rule, the first valid match is
    collapsed repeatedly until that rule has no additional matches in the
    updated graph.

    Args:
        graph: Source graph before collapse.
        rules: Normalized collapse rules.
        collapse_annotations: Whether collapsed block annotations are enabled.

    Returns:
        Tuple of ``(collapsed_graph, applied_counts)`` where ``applied_counts``
        tracks how many collapses each rule produced.
    """
    if not rules:
        return graph, []

    collapsed_graph = graph
    applied_counts = [0 for _ in rules]
    for rule_index, rule in enumerate(rules):
        while True:
            sequence = _find_first_collapse_sequence(collapsed_graph, rule)
            if not sequence:
                break
            collapsed_graph = _collapse_node_sequence(
                collapsed_graph,
                sequence,
                rule=rule,
                collapse_annotations=collapse_annotations,
            )
            applied_counts[rule_index] += 1
    return collapsed_graph, applied_counts


def _assign_ranks(
    nodes: Dict[int, FunctionalNode],
    edges: Sequence[FunctionalEdge],
) -> Dict[int, int]:
    outgoing: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
    incoming_count = {node_id: 0 for node_id in nodes}
    for edge in edges:
        outgoing[edge.src].append(edge.dst)
        incoming_count[edge.dst] += 1

    queue = deque(sorted(
        (node_id for node_id, count in incoming_count.items() if count == 0),
        key=lambda node_id: nodes[node_id].order,
    ))
    ranks = {node_id: 0 for node_id in queue}

    while queue:
        node_id = queue.popleft()
        for child_id in outgoing[node_id]:
            ranks[child_id] = max(ranks.get(child_id, 0), ranks[node_id] + 1)
            incoming_count[child_id] -= 1
            if incoming_count[child_id] == 0:
                queue.append(child_id)

    if len(ranks) != len(nodes):
        missing = [node_id for node_id in nodes if node_id not in ranks]
        for node_id in missing:
            ranks[node_id] = 0
        warnings.warn(
            "Functional graph contains cycles or disconnected nodes. "
            "Assigning rank 0 to unprocessed nodes.",
            UserWarning,
            stacklevel=2,
        )

    for node_id, rank in ranks.items():
        nodes[node_id].rank = rank
    return ranks


def _expand_long_edges(
    graph: FunctionalGraph,
    ranks: Dict[int, int],
    virtual_node_size: int,
) -> Tuple[FunctionalGraph, Dict[int, int]]:
    if not graph.edges:
        return graph, ranks

    new_nodes = dict(graph.nodes)
    new_edges: List[FunctionalEdge] = []
    order_base = max((node.order for node in new_nodes.values()), default=0) + 1

    for edge in graph.edges:
        src_rank = ranks.get(edge.src, 0)
        dst_rank = ranks.get(edge.dst, src_rank + 1)
        rank_delta = dst_rank - src_rank
        if rank_delta <= 1:
            new_edges.append(edge)
            continue

        prev_id = edge.src
        for step in range(1, rank_delta):
            rank = src_rank + step
            name = f"virtual_{order_base}"
            synthetic = _SyntheticLayer(name=name)
            node_id = id(synthetic)
            new_nodes[node_id] = FunctionalNode(
                layer=synthetic,
                node_id=node_id,
                name=name,
                layer_type=type(synthetic),
                shape=None,
                dims=(virtual_node_size, virtual_node_size, virtual_node_size),
                width=virtual_node_size,
                height=virtual_node_size,
                order=order_base,
                rank=rank,
                kind="virtual",
                de=0,  # Virtual nodes are always flat
                shade=0
            )
            ranks[node_id] = rank
            new_edges.append(FunctionalEdge(prev_id, node_id))
            prev_id = node_id
            order_base += 1

        new_edges.append(FunctionalEdge(prev_id, edge.dst))

    inputs, outputs = _find_inputs_outputs(new_nodes, new_edges)
    return FunctionalGraph(nodes=new_nodes, edges=new_edges, inputs=inputs, outputs=outputs), ranks


def _mark_inputs_outputs(graph: FunctionalGraph) -> None:
    for node_id in graph.inputs:
        node = graph.nodes.get(node_id)
        if node and node.kind == "layer":
            node.kind = "input"
    for node_id in graph.outputs:
        node = graph.nodes.get(node_id)
        if node and node.kind == "layer":
            node.kind = "output"


def _split_components(
    nodes: Mapping[int, FunctionalNode],
    edges: Sequence[FunctionalEdge],
) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        adjacency[edge.src].append(edge.dst)
        adjacency[edge.dst].append(edge.src)

    components: List[List[int]] = []
    visited = set()

    for node_id in nodes:
        if node_id in visited:
            continue
        queue = deque([node_id])
        visited.add(node_id)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    return components


def _component_sort_key(graph: FunctionalGraph, node_ids: Sequence[int]) -> Tuple[int, int]:
    orders = [graph.nodes[node_id].order for node_id in node_ids]
    ranks = [graph.nodes[node_id].rank for node_id in node_ids]
    return (min(ranks or [0]), min(orders or [0]))


def _order_by_barycenter(
    graph: FunctionalGraph,
    node_ids: Sequence[int],
    ranks: Dict[int, int],
    *,
    iterations: int,
) -> Dict[int, List[int]]:
    node_set = set(node_ids)
    sub_edges = [
        edge
        for edge in graph.edges
        if edge.src in node_set
        and edge.dst in node_set
        and abs(ranks.get(edge.dst, 0) - ranks.get(edge.src, 0)) == 1
    ]

    max_rank = max((ranks.get(node_id, 0) for node_id in node_ids), default=0)
    rank_nodes: Dict[int, List[int]] = {rank: [] for rank in range(max_rank + 1)}
    for node_id in node_ids:
        rank_nodes[ranks.get(node_id, 0)].append(node_id)
    for node_list in rank_nodes.values():
        node_list.sort(key=lambda node_id: graph.nodes[node_id].order)

    positions = _positions_from_rank_nodes(rank_nodes)
    incoming = _incoming_map(sub_edges)
    outgoing = _outgoing_map(sub_edges)

    if iterations <= 0:
        for rank, node_list in rank_nodes.items():
            for index, node_id in enumerate(node_list):
                graph.nodes[node_id].rank_order = index
        return rank_nodes

    for _ in range(iterations):
        for rank in range(1, max_rank + 1):
            rank_nodes[rank].sort(
                key=lambda node_id: _barycenter_key(node_id, incoming, positions, graph)
            )
            positions = _positions_from_rank_nodes(rank_nodes)

        for rank in range(max_rank - 1, -1, -1):
            rank_nodes[rank].sort(
                key=lambda node_id: _barycenter_key(node_id, outgoing, positions, graph)
            )
            positions = _positions_from_rank_nodes(rank_nodes)

    for rank, node_list in rank_nodes.items():
        for index, node_id in enumerate(node_list):
            graph.nodes[node_id].rank_order = index

    return rank_nodes


def _barycenter_key(
    node_id: int,
    neighbor_map: Mapping[int, List[int]],
    positions: Mapping[int, int],
    graph: FunctionalGraph,
) -> Tuple[float, int]:
    neighbors = neighbor_map.get(node_id, [])
    values = [positions[n] for n in neighbors if n in positions]
    if values:
        barycenter = sum(values) / len(values)
    else:
        barycenter = float(positions.get(node_id, 0))
    return (barycenter, graph.nodes[node_id].order)


def _compute_external_text_padding(
    graph: FunctionalGraph,
    *,
    text_callable: Callable[[int, Any], Tuple[str, bool]],
    text_vspacing: int,
    font: Optional[ImageFont.ImageFont],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Estimate vertical label extents for external text labels.

    The returned maps are keyed by node id and contain pixel extents that should
    be reserved above or below node rectangles to avoid overlap with neighboring
    rows when labels are drawn outside boxes.
    """
    text_top_padding: Dict[int, int] = {}
    text_bottom_padding: Dict[int, int] = {}
    active_font = font or ImageFont.load_default()

    visible_nodes = [
        node
        for node in sorted(graph.nodes.values(), key=lambda n: n.order)
        if node.kind != "virtual"
    ]
    for index, node in enumerate(visible_nodes):
        text, above = text_callable(index, node.layer)
        text_value = "" if text is None else str(text)
        if not text_value:
            continue

        text_height = 0
        for line in text_value.split("\n"):
            if hasattr(active_font, "getsize"):
                text_height += active_font.getsize(line)[1]
            else:
                text_height += active_font.getbbox(line)[3]
        text_height += (len(text_value.split("\n")) - 1) * text_vspacing

        extent = max(0, int(text_height) + 4)
        if above:
            text_top_padding[node.node_id] = extent
        else:
            text_bottom_padding[node.node_id] = extent

    return text_top_padding, text_bottom_padding


def _resolve_external_label_x_collisions(
    labels: List[Dict[str, Any]],
    *,
    image_width: int,
    edge_padding: int,
    min_gap: int = 8,
    y_tolerance: int = 2,
) -> None:
    """Shift external labels horizontally to reduce same-row overlap.

    Labels are grouped by overlapping y-ranges (with a small tolerance). Within
    each group, labels are ordered by preferred x and nudged to satisfy a
    minimum horizontal gap where possible.
    """
    if len(labels) < 2:
        return

    sorted_indices = sorted(range(len(labels)), key=lambda idx: labels[idx]["y"])
    groups: List[List[int]] = []
    current_group = [sorted_indices[0]]
    current_bottom = labels[sorted_indices[0]]["y"] + labels[sorted_indices[0]]["h"]

    for idx in sorted_indices[1:]:
        label = labels[idx]
        y1 = label["y"]
        y2 = y1 + label["h"]
        if y1 <= current_bottom + y_tolerance:
            current_group.append(idx)
            current_bottom = max(current_bottom, y2)
        else:
            groups.append(current_group)
            current_group = [idx]
            current_bottom = y2
    groups.append(current_group)

    left_bound = float(edge_padding)
    right_bound = float(max(edge_padding, image_width - edge_padding))
    available = max(1.0, right_bound - left_bound)

    for group in groups:
        if len(group) < 2:
            continue

        ordered = sorted(group, key=lambda idx: labels[idx]["x_pref"])
        widths = [float(max(1, labels[idx]["w"])) for idx in ordered]
        required = sum(widths) + float(min_gap * (len(ordered) - 1))

        xs = [float(labels[idx]["x_pref"]) for idx in ordered]
        if required > available:
            x_cursor = left_bound
            for i in range(len(xs)):
                xs[i] = x_cursor
                x_cursor += widths[i] + min_gap
        else:
            xs[0] = max(xs[0], left_bound)
            for i in range(1, len(xs)):
                xs[i] = max(xs[i], xs[i - 1] + widths[i - 1] + min_gap)

            overflow = (xs[-1] + widths[-1]) - right_bound
            if overflow > 0:
                xs = [x - overflow for x in xs]

            if xs[0] < left_bound:
                x_cursor = left_bound
                for i in range(len(xs)):
                    xs[i] = x_cursor
                    x_cursor += widths[i] + min_gap

        for i, idx in enumerate(ordered):
            labels[idx]["x"] = int(round(xs[i]))


def _positions_from_rank_nodes(rank_nodes: Mapping[int, List[int]]) -> Dict[int, int]:
    positions: Dict[int, int] = {}
    for node_list in rank_nodes.values():
        for index, node_id in enumerate(node_list):
            positions[node_id] = index
    return positions


def _assign_component_positions(
    graph: FunctionalGraph,
    node_ids: Sequence[int],
    rank_nodes: Mapping[int, List[int]],
    x_positions: Mapping[int, int],
    column_widths: Mapping[int, int],
    base_y: int,
    row_spacing: int,
    *,
    node_top_padding: Optional[Mapping[int, int]] = None,
    node_bottom_padding: Optional[Mapping[int, int]] = None,
) -> int:
    node_top_padding = node_top_padding or {}
    node_bottom_padding = node_bottom_padding or {}
    node_set = set(node_ids)
    max_height = 0
    column_heights: Dict[int, int] = {}

    for rank, ordered_ids in rank_nodes.items():
        filtered = [node_id for node_id in ordered_ids if node_id in node_set]
        if not filtered:
            continue
        column_width = column_widths.get(rank, 0)
        if column_width <= 0:
            column_width = max(graph.nodes[node_id].width for node_id in filtered)
        column_height = 0
        for node_id in filtered:
            node = graph.nodes[node_id]
            column_height += (
                node.height
                + int(node_top_padding.get(node_id, 0))
                + int(node_bottom_padding.get(node_id, 0))
            )
        if len(filtered) > 1:
            column_height += row_spacing * (len(filtered) - 1)
        column_heights[rank] = column_height
        max_height = max(max_height, column_height)

    for rank, ordered_ids in rank_nodes.items():
        filtered = [node_id for node_id in ordered_ids if node_id in node_set]
        if not filtered:
            continue
        column_width = column_widths.get(rank, 0)
        if column_width <= 0:
            column_width = max(graph.nodes[node_id].width for node_id in filtered)
        column_height = column_heights.get(rank, 0)
        y_cursor = base_y + int((max_height - column_height) / 2)
        for node_id in filtered:
            node = graph.nodes[node_id]
            top_pad = int(node_top_padding.get(node_id, 0))
            bottom_pad = int(node_bottom_padding.get(node_id, 0))
            node.x = x_positions.get(rank, 0) + int((column_width - node.width) / 2)
            node.y = y_cursor + top_pad
            y_cursor = node.y + node.height + bottom_pad + row_spacing
        if column_height:
            max_height = max(max_height, y_cursor - base_y - row_spacing)

    return max_height


def _column_widths(nodes: Mapping[int, FunctionalNode], ranks: Mapping[int, int]) -> Dict[int, int]:
    widths: Dict[int, int] = {}
    for node_id, node in nodes.items():
        rank = ranks.get(node_id, 0)
        widths[rank] = max(widths.get(rank, 0), node.width)
    return widths


def _column_positions(column_widths: Mapping[int, int], padding: int, column_spacing: int) -> Dict[int, int]:
    if not column_widths:
        return {}
    max_rank = max(column_widths)
    positions: Dict[int, int] = {}
    x_cursor = padding
    for rank in range(max_rank + 1):
        positions[rank] = x_cursor
        x_cursor += column_widths.get(rank, 0) + column_spacing
    return positions


def _get_font(group: Dict[str, Any]) -> ImageFont.ImageFont:
    font_src = group.get("font", None)
    font_size = group.get("font_size", 15)
    
    if font_src is None:
         try:
            return ImageFont.truetype("arial.ttf", font_size)
         except IOError:
            return ImageFont.load_default()
    elif isinstance(font_src, str):
         try:
            return ImageFont.truetype(font_src, font_size)
         except IOError:
            return ImageFont.load_default()
    elif isinstance(font_src, ImageFont.ImageFont):
        return font_src
    else:
        return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: Any) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    else:
        return draw.textsize(text, font=font)




def _prettify_layer_name(name: str) -> str:
    if not name:
        return ""
    name = name.replace("_", " ").strip()
    # CamelCase -> spaced
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _resolve_box_label(node: FunctionalNode) -> str:
    style = node.style or {}
    if style.get("box_text") is not None:
        return str(style.get("box_text"))
    cb = style.get("box_text_callable")
    if callable(cb):
        try:
            out = cb(node.layer)
            if out is not None:
                return str(out)
        except Exception:
            pass
    return _prettify_layer_name(getattr(node.layer_type, "__name__", node.name))


def _try_load_font(path: str, size: int) -> Optional[ImageFont.ImageFont]:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def _is_font_like(value: Any) -> bool:
    if value is None:
        return False
    return any(hasattr(value, attr) for attr in ("getbbox", "getsize", "getmask"))


def _resolve_box_font(style: Mapping[str, Any], fallback: Optional[Any]) -> Tuple[Any, Optional[str], int]:
    size = int(style.get("box_text_font_size", 14) or 14)
    src = style.get("box_text_font")
    if _is_font_like(src):
        return src, None, size
    if isinstance(src, str):
        f = _try_load_font(src, size)
        if f is not None:
            return f, src, size
    if _is_font_like(fallback):
        return fallback, None, size
    for cand in ("DejaVuSans.ttf", "arial.ttf"):
        f = _try_load_font(cand, size)
        if f is not None:
            return f, cand, size
    return ImageFont.load_default(), None, size


def _multiline_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, spacing: int) -> Tuple[int, int, int, int]:
    # Prefer PIL's bbox helper (handles glyph bearings correctly)
    if hasattr(draw, "multiline_textbbox"):
        return draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="left")
    # Fallback: union per-line bbox
    lefts, tops, rights, bottoms = [], [], [], []
    y = 0
    for line in text.split("\n"):
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, y), line, font=font)
        else:
            w, h = draw.textsize(line, font=font)
            l, t, r, b = 0, y, w, y + h
        lefts.append(l); tops.append(t); rights.append(r); bottoms.append(b)
        y += (b - t) + spacing
    if not lefts:
        return (0, 0, 0, 0)
    return (min(lefts), min(tops), max(rights), max(bottoms))


def _render_text_image_bbox(text: str, font: ImageFont.ImageFont, color: Any, spacing: int, margin: int = 2) -> Image.Image:
    # IMPORTANT: account for negative bearings by offsetting by -left/-top
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    l, t, r, b = _multiline_bbox(d, text, font, spacing)
    w = max(1, int(r - l))
    h = max(1, int(b - t))
    img = Image.new("RGBA", (w + 2 * margin, h + 2 * margin), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(img)
    d2.multiline_text((margin - l, margin - t), text, font=font, fill=color, spacing=spacing, align="center")
    return img


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, mode: str, max_lines: Optional[int]) -> str:
    mode = (mode or "words").lower()
    if mode == "none" or max_width <= 0:
        return text
    if mode not in {"words", "chars"}:
        mode = "words"
    tokens = text.split() if mode == "words" else list(text)
    lines = []
    cur = ""
    for tok in tokens:
        cand = tok if not cur else (cur + (" " if mode == "words" else "") + tok)
        l, t, r, b = _multiline_bbox(draw, cand, font, 0)
        if (r - l) <= max_width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
            if max_lines is not None and len(lines) >= max_lines:
                break
    if cur and (max_lines is None or len(lines) < max_lines):
        lines.append(cur)
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)


def _draw_box_text_in_rect(
    base: Image.Image,
    rect: Tuple[int, int, int, int],
    text: str,
    *,
    style: Mapping[str, Any],
    fallback_font: Optional[ImageFont.ImageFont],
    fallback_color: Any,
    fallback_spacing: int,
) -> None:
    """Render style-driven text inside a rectangular node region.

    The text is optionally wrapped, autoshrunk, rotated, and aligned according
    to ``box_text_*`` style keys. Rendering is performed via an intermediate
    RGBA text image to avoid clipping from glyph bearings.

    Args:
        base: Destination image.
        rect: Target rectangle ``(x1, y1, x2, y2)``.
        text: Raw label text.
        style: Node style mapping with text layout keys.
        fallback_font: Global fallback font when per-node font is absent.
        fallback_color: Global fallback text color.
        fallback_spacing: Global fallback line spacing.
    """
    if not text:
        return
    x1, y1, x2, y2 = rect
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    pad = style.get("box_text_padding", 8)
    if isinstance(pad, (tuple, list)) and len(pad) == 2:
        pad_x, pad_y = int(pad[0]), int(pad[1])
    else:
        pad_x = pad_y = int(pad)

    avail_w = max(1, w - 2 * pad_x)
    avail_h = max(1, h - 2 * pad_y)

    orientation = str(style.get("box_orientation", "vertical") or "vertical").lower()
    rot = style.get("box_text_rotation")
    if rot is None:
        rot = 90 if orientation == "vertical" else 0
    try:
        rot = int(rot) % 360
    except Exception:
        rot = 0
    if rot not in (0, 90, 180, 270):
        rot = 0

    # For 90/270, swap fit constraints pre-rotation
    pre_w, pre_h = (avail_h, avail_w) if rot in (90, 270) else (avail_w, avail_h)

    spacing = int(style.get("box_text_line_spacing", fallback_spacing) or fallback_spacing)
    color = style.get("box_text_color", fallback_color)

    wrap_mode = str(style.get("box_text_wrap", "words") or "words").lower()
    max_lines = style.get("box_text_max_lines")
    try:
        max_lines = int(max_lines) if max_lines is not None else None
    except Exception:
        max_lines = None

    autoshrink = bool(style.get("box_text_autoshrink", True))
    min_size = int(style.get("box_text_min_font_size", 8) or 8)

    font, font_path, size0 = _resolve_box_font(style, fallback_font)

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)

    def measure_fit(fnt: ImageFont.ImageFont) -> Tuple[str, int, int, bool]:
        wrapped = _wrap_text_to_width(d, text, fnt, pre_w, wrap_mode, max_lines)
        l, t, r, b = _multiline_bbox(d, wrapped, fnt, spacing)
        tw = int(r - l)
        th = int(b - t)
        return wrapped, tw, th, (tw <= pre_w and th <= pre_h)

    wrapped, tw, th, ok = measure_fit(font)
    if autoshrink and not ok and font_path:
        cur = size0
        while cur > min_size:
            cur -= 1
            f2 = _try_load_font(font_path, cur)
            if f2 is None:
                break
            wrapped2, tw2, th2, ok2 = measure_fit(f2)
            if ok2:
                font = f2
                wrapped, tw, th, ok = wrapped2, tw2, th2, ok2
                break

    txt_img = _render_text_image_bbox(wrapped, font, color, spacing, margin=2)
    if rot:
        txt_img = txt_img.rotate(rot, expand=True)

    tw, th = txt_img.size

    align = str(style.get("box_text_align", "center") or "center").lower()
    valign = str(style.get("box_text_valign", "middle") or "middle").lower()

    if align == "left":
        px = x1 + pad_x
    elif align == "right":
        px = x2 - pad_x - tw
    else:
        px = x1 + (w - tw) // 2

    if valign == "top":
        py = y1 + pad_y
    elif valign == "bottom":
        py = y2 - pad_y - th
    else:
        py = y1 + (h - th) // 2

    base.alpha_composite(txt_img, (int(px), int(py)))


def _resolve_annotation_font(
    style_font: Any,
    fallback_font: Optional[Any],
    size: int,
) -> Any:
    if _is_font_like(style_font):
        return style_font
    if isinstance(style_font, str):
        resolved = _try_load_font(style_font, size)
        if resolved is not None:
            return resolved
    if _is_font_like(fallback_font):
        return fallback_font
    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        resolved = _try_load_font(candidate, size)
        if resolved is not None:
            return resolved
    return ImageFont.load_default()


def _draw_collapse_badge(
    draw: ImageDraw.ImageDraw,
    *,
    rect: Tuple[int, int, int, int],
    label: str,
    font: Any,
    fill: Any,
    outline: Any,
    text_color: Any,
    padding: Union[int, Tuple[int, int], List[int]],
) -> None:
    """Draw a compact collapse-count badge (for example ``"4x"``) on a node.

    Args:
        draw: PIL drawing context.
        rect: Node bounds ``(x1, y1, x2, y2)``.
        label: Badge text.
        font: Font used for label text.
        fill: Badge background color.
        outline: Badge border color.
        text_color: Badge text color.
        padding: Horizontal/vertical text padding in the badge.
    """
    if not label:
        return

    x1, y1, x2, _ = rect
    text_w, text_h = _measure_text(draw, label, font)
    if isinstance(padding, (tuple, list)) and len(padding) == 2:
        pad_x, pad_y = int(padding[0]), int(padding[1])
    else:
        pad_x = pad_y = int(padding)

    badge_w = max(1, text_w + 2 * pad_x)
    badge_h = max(1, text_h + 2 * pad_y)
    badge_x2 = x2 - 4
    badge_x1 = max(x1 + 2, badge_x2 - badge_w)
    badge_y1 = y1 + 4
    badge_y2 = badge_y1 + badge_h

    if hasattr(draw, "rounded_rectangle"):
        try:
            draw.rounded_rectangle(
                (badge_x1, badge_y1, badge_x2, badge_y2),
                radius=4,
                fill=fill,
                outline=outline,
                width=1,
            )
        except TypeError:
            draw.rounded_rectangle(
                (badge_x1, badge_y1, badge_x2, badge_y2),
                radius=4,
                fill=fill,
                outline=outline,
            )
    else:
        draw.rectangle((badge_x1, badge_y1, badge_x2, badge_y2), fill=fill, outline=outline)
    draw.text((badge_x1 + pad_x, badge_y1 + pad_y), label, font=font, fill=text_color)


def _draw_collapse_block_annotation(
    draw: ImageDraw.ImageDraw,
    *,
    rect: Tuple[int, int, int, int],
    label: str,
    position: str,
    color: Any,
    line_width: int,
    head_size: int,
    offset: int,
    font: Any,
    image_size: Tuple[int, int],
) -> None:
    """Draw a block-collapse double-headed arrow with an optional count label.

    Args:
        draw: PIL drawing context.
        rect: Collapsed node bounds ``(x1, y1, x2, y2)``.
        label: Annotation text shown near the arrow.
        position: ``"above"`` or ``"below"`` relative to the node.
        color: Arrow and text color.
        line_width: Arrow line thickness.
        head_size: Arrowhead size.
        offset: Pixel distance from node edge to arrow baseline.
        font: Font used for annotation text.
        image_size: Destination image size used for clipping guards.
    """
    x1, y1, x2, y2 = rect
    if x2 <= x1:
        return

    left = int(x1 + 8)
    right = int(x2 - 8)
    if right - left < 12:
        left = x1 + 2
        right = x2 - 2
    if right - left < 8:
        return

    if position == "below":
        line_y = y2 + offset
    else:
        line_y = y1 - offset
    line_y = max(2, min(image_size[1] - 3, int(line_y)))

    draw.line((left, line_y, right, line_y), fill=color, width=max(1, line_width))
    head = max(3, head_size)
    draw.polygon(
        [(left, line_y), (left + head, line_y - head // 2), (left + head, line_y + head // 2)],
        fill=color,
    )
    draw.polygon(
        [(right, line_y), (right - head, line_y - head // 2), (right - head, line_y + head // 2)],
        fill=color,
    )

    if not label:
        return
    text_w, text_h = _measure_text(draw, label, font)
    text_x = int((left + right - text_w) / 2)
    if position == "below":
        text_y = line_y + 4
    else:
        text_y = line_y - text_h - 4
    text_y = max(0, min(image_size[1] - max(1, text_h), text_y))
    bg = (255, 255, 255, 220)
    draw.rectangle(
        (text_x - 2, text_y - 1, text_x + text_w + 2, text_y + text_h + 1),
        fill=bg,
    )
    draw.text((text_x, text_y), label, font=font, fill=color)


def _draw_collapsed_annotations(
    img: Image.Image,
    collapsed_nodes: Sequence[Tuple[FunctionalNode, Tuple[int, int, int, int]]],
    *,
    fallback_font: Optional[Any],
    fallback_font_color: Any,
    default_annotation_color: Any,
) -> None:
    """Render all collapsed-node overlays after the main node pass.

    For each collapsed node this can draw:
    - an ``Nx`` badge
    - an optional double-headed block annotation line with label

    Args:
        img: Destination RGBA image.
        collapsed_nodes: ``(node, rect)`` entries gathered during rendering.
        fallback_font: Global fallback font.
        fallback_font_color: Global fallback text color.
        default_annotation_color: Global fallback arrow/annotation color.
    """
    if not collapsed_nodes:
        return

    draw = ImageDraw.Draw(img)
    for node, rect in collapsed_nodes:
        style = node.style or {}
        if not bool(style.get("collapsed", False)):
            continue
        label = str(style.get("collapse_label", ""))

        if bool(style.get("collapse_badge_enabled", True)):
            badge_font_size = int(style.get("collapse_badge_font_size", 12) or 12)
            badge_font = _resolve_annotation_font(
                style.get("collapse_badge_font"),
                fallback_font,
                badge_font_size,
            )
            _draw_collapse_badge(
                draw,
                rect=rect,
                label=label,
                font=badge_font,
                fill=style.get("collapse_badge_fill", "white"),
                outline=style.get("collapse_badge_outline", "black"),
                text_color=style.get("collapse_badge_text_color", fallback_font_color),
                padding=style.get("collapse_badge_padding", (4, 2)),
            )

        if (
            style.get("collapse_kind") == "block"
            and bool(style.get("collapse_annotation_enabled", True))
        ):
            annotation_font_size = int(style.get("collapse_annotation_font_size", 12) or 12)
            annotation_font = _resolve_annotation_font(
                style.get("collapse_annotation_font"),
                fallback_font,
                annotation_font_size,
            )
            position = str(style.get("collapse_annotation_position", "above") or "above").lower()
            if position not in {"above", "below"}:
                position = "above"
            _draw_collapse_block_annotation(
                draw,
                rect=rect,
                label=label,
                position=position,
                color=style.get("collapse_annotation_color", default_annotation_color),
                line_width=int(style.get("collapse_annotation_width", 2) or 2),
                head_size=int(style.get("collapse_annotation_head_size", 6) or 6),
                offset=int(style.get("collapse_annotation_offset", 10) or 10),
                font=annotation_font,
                image_size=img.size,
            )


def _render_graph(
    graph: FunctionalGraph,
    *,
    color_map: Mapping[type, Mapping[str, Any]],
    background_fill: Any,
    padding: int,
    connector_fill: Any,
    connector_width: int,
    connector_arrow: bool,
    connector_padding: int,
    text_callable: Optional[Callable[[int, Any], Tuple[str, bool]]],
    text_vspacing: int,
    font: Optional[ImageFont.ImageFont],
    font_color: Any,
    render_virtual_nodes: bool,
    draw_volume: bool,
    orientation_rotation: Optional[float] = None,
    layered_groups: Optional[Sequence[Dict[str, Any]]] = None,
    logo_groups: Optional[Sequence[Dict[str, Any]]] = None,
    logos_legend: Union[bool, Dict[str, Any]] = False,
    simple_text_visualization: bool = False,
    external_text_bottom_padding: Optional[Mapping[int, int]] = None,
) -> Image.Image:
    """Render a positioned ``FunctionalGraph`` to a PIL image.

    Rendering order:
    1. optional group backgrounds
    2. connectors
    3. nodes (flat or volumetric), node images, and logos
    4. in-node text and external annotations (collapsed markers)
    5. optional group captions and logo legend

    Args:
        graph: Graph with already-computed node positions.
        color_map: Optional per-layer fill/outline overrides.
        background_fill: Canvas background color.
        padding: Outer image padding.
        connector_fill: Connector color.
        connector_width: Default connector width.
        connector_arrow: Whether connectors include arrowheads by default.
        connector_padding: Anchor offset from node faces.
        text_callable: Optional callable for above/below external labels.
        text_vspacing: Vertical spacing for multiline text labels.
        font: Optional global font for text labels.
        font_color: Global text color.
        render_virtual_nodes: Whether virtual routing nodes are visible.
        draw_volume: Whether default node rendering uses volumetric boxes.
        orientation_rotation: Optional rotation applied to volumetric boxes.
        layered_groups: Optional background highlight groups.
        logo_groups: Optional node-logo overlay groups.
        logos_legend: Legend toggle/config for logo groups.
        simple_text_visualization: Switch to flat 2D box rendering mode.

    Returns:
        A rendered ``PIL.Image.Image``.
    """
    max_right = padding
    max_bottom = padding
    for node in graph.nodes.values():
        max_right = max(max_right, node.x + node.width + padding)
        max_bottom = max(max_bottom, node.y + node.height + padding)
        if (
            node.kind == "collapsed"
            and node.style.get("collapse_kind") == "block"
            and bool(node.style.get("collapse_annotation_enabled", True))
            and str(node.style.get("collapse_annotation_position", "above")).lower() == "below"
        ):
            annotation_offset = int(node.style.get("collapse_annotation_offset", 10) or 10)
            max_bottom = max(max_bottom, node.y + node.height + annotation_offset + padding + 24)

    if external_text_bottom_padding:
        for node_id, extra in external_text_bottom_padding.items():
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            max_bottom = max(max_bottom, node.y + node.height + int(extra) + padding)

    if layered_groups:
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        for group in layered_groups:
            group_nodes = _get_group_nodes(graph, group)
            if not group_nodes:
                continue
            g_min_x = min(n.x for n in group_nodes)
            g_max_x = max(n.x + n.width for n in group_nodes)
            g_min_y = min(n.y for n in group_nodes)
            g_max_y = max(n.y + n.height for n in group_nodes)
            
            g_padding = group.get("padding", 10)
            g_min_x -= g_padding
            g_max_x += g_padding
            g_min_y -= g_padding
            g_max_y += g_padding
            
            max_right = max(max_right, g_max_x + padding)
            max_bottom = max(max_bottom, g_max_y + padding)
            caption = group.get("name", group.get("caption"))
            if caption:
                font = _get_font(group)
                text_w, text_h = _measure_text(dummy_draw, caption, font)
                
                center_x = (g_min_x + g_max_x) / 2
                text_x = center_x - text_w / 2
                gap = group.get("text_spacing", 5)
                text_y = g_max_y + gap
                
                max_right = max(max_right, text_x + text_w + padding)
                max_bottom = max(max_bottom, text_y + text_h + padding)

    node_logos = {} # node_id -> list of (group, image)
    if logo_groups:
        for group in logo_groups:
            path = group.get("file")
            if not path: continue
            try:
                logo_img = Image.open(path)
            except:
                continue
                
            target_nodes = _get_logo_nodes(graph, group)
            for node in target_nodes:
                if node.node_id not in node_logos:
                    node_logos[node.node_id] = []
                node_logos[node.node_id].append((group, logo_img))

    img = Image.new("RGBA", (int(max_right), int(max_bottom)), background_fill)
    draw = aggdraw.Draw(img)
    color_wheel = ColorWheel()

    if layered_groups:
        _draw_group_boxes(draw, graph, layered_groups)

    _draw_connectors(
        draw,
        graph.edges,
        graph.nodes,
        render_virtual_nodes,
        connector_fill,
        connector_width,
        connector_arrow,
        connector_padding,
    )
    draw.flush()

    pending_simple_text: List[Tuple[FunctionalNode, Tuple[int, int, int, int]]] = []
    pending_collapsed_annotations: List[Tuple[FunctionalNode, Tuple[int, int, int, int]]] = []

    for node in graph.nodes.values():
        if node.kind == "virtual" and not render_virtual_nodes:
            continue

        if simple_text_visualization:
            x1 = int(node.x)
            y1 = int(node.y)
            x2 = int(node.x + node.width)
            y2 = int(node.y + node.height)

            fill = node.style.get("box_fill")
            if fill is None:
                fill = node.style.get("fill")
            if fill is None:
                fill = color_map.get(node.layer_type, {}).get("fill")
            if fill is None:
                fill = color_wheel.get_color(node.layer_type)

            outline = node.style.get("box_outline")
            if outline is None:
                outline = node.style.get("outline")
            if outline is None:
                outline = color_map.get(node.layer_type, {}).get("outline")
            if outline is None:
                outline = "black"

            outline_w = int(node.style.get("box_outline_width", 2) or 2)
            pen = aggdraw.Pen(get_rgba_tuple(outline), outline_w)
            brush = aggdraw.Brush(get_rgba_tuple(fill))
            draw.rectangle((x1, y1, x2, y2), pen, brush)

            if node.image is not None:
                draw.flush()
                fit_mode = node.style.get("image_fit", "fill")
                quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                try:
                    apply_affine_transform(img, node.image, quad, fit_mode)
                except Exception:
                    pass
                draw = aggdraw.Draw(img)

            if node.node_id in node_logos:
                draw.flush()
                box_tmp = Box()
                box_tmp.de = 0
                box_tmp.shade = 0
                box_tmp.rotation = None
                box_tmp.x1 = x1
                box_tmp.y1 = y1
                box_tmp.x2 = x2
                box_tmp.y2 = y2
                for group, logo_img in node_logos[node.node_id]:
                    draw_node_logo(img, box_tmp, logo_img, group, draw_volume=False)
                draw = aggdraw.Draw(img)

            if node.kind != "virtual" and bool(node.style.get("box_text_enabled", True)):
                pending_simple_text.append((node, (x1, y1, x2, y2)))
            if node.kind == "collapsed":
                pending_collapsed_annotations.append((node, (x1, y1, x2, y2)))
            continue

        box = Box()
        box.de = getattr(node, 'de', 0)
        box.shade = getattr(node, 'shade', 0)
        box.rotation = orientation_rotation

        box.x1 = node.x
        box.y1 = node.y + box.de

        real_width = node.width - box.de
        real_height = node.height - box.de

        box.x2 = box.x1 + real_width
        box.y2 = box.y1 + real_height

        fill = color_map.get(node.layer_type, {}).get("fill")
        outline = color_map.get(node.layer_type, {}).get("outline")
        if node.kind == "virtual":
            bg = get_rgba_tuple(background_fill)
            fill = fade_color(bg, 10)
            outline = fade_color(get_rgba_tuple(connector_fill), 10)
            
        box.fill = fill if fill is not None else color_wheel.get_color(node.layer_type)
        box.outline = outline if outline is not None else "black"

        if node.style.get('fill'):
            box.fill = node.style.get('fill')
        if node.style.get('outline'):
            box.outline = node.style.get('outline')

        box.draw(draw, draw_reversed=False)

        if node.image is not None:
            draw.flush()
            
            fit_mode = node.style.get("image_fit", "fill")
            axis = node.style.get("image_axis", "z")

            target_face_idx = 0 # Front
            if axis == 'y': target_face_idx = 4 # Top
            elif axis == 'x': target_face_idx = 2 # Right / Side

            quad = box.get_face_quad(target_face_idx)
            
            if quad:
                apply_affine_transform(img, node.image, quad, fit_mode)

            draw = aggdraw.Draw(img)

        if node.node_id in node_logos:
            draw.flush()
            for group, logo_img in node_logos[node.node_id]:
                draw_node_logo(img, box, logo_img, group, draw_volume)
            draw = aggdraw.Draw(img)

        if node.kind == "collapsed":
            pending_collapsed_annotations.append(
                (node, (int(node.x), int(node.y), int(node.x + node.width), int(node.y + node.height)))
            )

    draw.flush()

    if simple_text_visualization and pending_simple_text:
        for node, rect in pending_simple_text:
            label = _resolve_box_label(node)
            _draw_box_text_in_rect(
                img,
                rect,
                label,
                style=node.style or {},
                fallback_font=font,
                fallback_color=font_color,
                fallback_spacing=text_vspacing,
            )

    if text_callable is not None:
        if font is None:
            font = ImageFont.load_default()
        draw_text = ImageDraw.Draw(img)
        external_labels: List[Dict[str, Any]] = []
        visible_nodes = [
            node
            for node in sorted(graph.nodes.values(), key=lambda n: n.order)
            if node.kind != "virtual"
        ]
        for index, node in enumerate(visible_nodes):
            text, above = text_callable(index, node.layer)
            text_value = "" if text is None else str(text)
            if not text_value:
                continue
            text_height = 0
            text_widths = []
            for line in text_value.split("\n"):
                if hasattr(font, "getsize"):
                    text_widths.append(font.getsize(line)[0])
                    text_height += font.getsize(line)[1]
                else:
                    bbox = font.getbbox(line)
                    text_widths.append(bbox[2])
                    text_height += bbox[3]
            text_height += (len(text_value.split("\n")) - 1) * text_vspacing

            de = getattr(node, 'de', 0)
            real_width = node.width - de
            real_height = node.height - de

            face_x = node.x
            face_y = node.y + de

            text_x = face_x + real_width / 2 - max(text_widths or [0]) / 2

            if above:
                text_y = face_y - text_height - 4
            else:
                text_y = face_y + real_height + 4

            text_width = max(text_widths or [0])
            external_labels.append(
                {
                    "x": float(text_x),
                    "x_pref": float(text_x),
                    "y": int(text_y),
                    "w": int(text_width),
                    "h": int(text_height),
                    "text": text_value,
                }
            )

        _resolve_external_label_x_collisions(
            external_labels,
            image_width=img.size[0],
            edge_padding=padding,
            min_gap=8,
        )

        for label in external_labels:
            draw_text.multiline_text(
                (label["x"], label["y"]),
                label["text"],
                font=font,
                fill=font_color,
                spacing=text_vspacing,
            )

    if pending_collapsed_annotations:
        _draw_collapsed_annotations(
            img,
            pending_collapsed_annotations,
            fallback_font=font,
            fallback_font_color=font_color,
            default_annotation_color=connector_fill,
        )

    if layered_groups:
        _draw_group_captions(img, graph, layered_groups)

    if logos_legend:
        if font is None:
            font = ImageFont.load_default()
        img = draw_logos_legend(img, logo_groups, logos_legend, background_fill, font, font_color)

    return img

def _straighten_layout(
    graph: FunctionalGraph,
    ranks: Dict[int, int],
    row_spacing: int,
    *,
    node_top_padding: Optional[Mapping[int, int]] = None,
    node_bottom_padding: Optional[Mapping[int, int]] = None,
) -> None:
    """
    Adjusts y positions to align linear connections straight. 
    Accounts for 3D depth (de) to ensure visual centers align.
    """
    nodes_by_rank = {}
    node_top_padding = node_top_padding or {}
    node_bottom_padding = node_bottom_padding or {}
    for node in graph.nodes.values():
        r = ranks.get(node.node_id, 0)
        nodes_by_rank.setdefault(r, []).append(node)

    sorted_ranks = sorted(nodes_by_rank.keys())
    
    outgoing = {n: [] for n in graph.nodes}
    incoming = {n: [] for n in graph.nodes}
    for edge in graph.edges:
        outgoing[edge.src].append(edge.dst)
        incoming[edge.dst].append(edge.src)

    def get_visual_center(node: FunctionalNode) -> float:
        de = getattr(node, 'de', 0)
        return node.y + (node.height + de) / 2.0

    def set_visual_center(node: FunctionalNode, center_y: float):
        de = getattr(node, 'de', 0)
        new_y = center_y - (node.height + de) / 2.0
        node.y = int(new_y)

    def resolve_collisions(col_nodes: List[FunctionalNode]):
        col_nodes.sort(key=lambda n: n.y - int(node_top_padding.get(n.node_id, 0)))
        if not col_nodes:
            return

        current_y_map = {n.node_id: n.y for n in col_nodes}

        for i in range(1, len(col_nodes)):
            prev_node = col_nodes[i-1]
            curr_node = col_nodes[i]
            prev_bottom = int(node_bottom_padding.get(prev_node.node_id, 0))
            curr_top = int(node_top_padding.get(curr_node.node_id, 0))
            min_y = (
                current_y_map[prev_node.node_id]
                + prev_node.height
                + prev_bottom
                + row_spacing
                + curr_top
            )
            if current_y_map[curr_node.node_id] < min_y:
                current_y_map[curr_node.node_id] = min_y

        for node in col_nodes:
            node.y = int(current_y_map[node.node_id])

    for rank in sorted_ranks:
        col_nodes = nodes_by_rank[rank]
        for node in col_nodes:
            parents = incoming[node.node_id]
            if len(parents) == 1:
                parent_id = parents[0]
                if len(outgoing[parent_id]) == 1:
                    parent = graph.nodes[parent_id]
                    target_center = get_visual_center(parent)
                    set_visual_center(node, target_center)
        
        resolve_collisions(col_nodes)

    for rank in reversed(sorted_ranks):
        col_nodes = nodes_by_rank[rank]
        for node in col_nodes:
            children = outgoing[node.node_id]
            if len(children) == 1:
                child_id = children[0]
                if len(incoming[child_id]) == 1:
                    child = graph.nodes[child_id]
                    target_center = get_visual_center(child)
                    set_visual_center(node, target_center)
        
        resolve_collisions(col_nodes)


def _get_group_nodes(graph: FunctionalGraph, group: Dict[str, Any]) -> List[FunctionalNode]:
    """Resolve all graph nodes referenced by one layered-group configuration.

    Group ``layers`` entries may reference layer objects directly or layer names.
    Name matching checks both the rendered node name and the underlying Keras
    layer ``name`` attribute.
    """
    layers = group.get("layers", [])
    if not layers:
        return []
        
    group_nodes = []
    for node in graph.nodes.values():
        for layer_ref in layers:
            if node.layer is layer_ref:
                group_nodes.append(node)
                break
            node_layer_name = getattr(node.layer, 'name', '')
            if isinstance(layer_ref, str) and (node.name == layer_ref or node_layer_name == layer_ref):
                group_nodes.append(node)
                break
    return group_nodes


def _draw_group_boxes(
    draw: aggdraw.Draw,
    graph: FunctionalGraph,
    groups: Sequence[Dict[str, Any]],
) -> None:
    """Draw configured group highlight rectangles behind matching nodes.

    Args:
        draw: Aggdraw context for the destination image.
        graph: Positioned graph.
        groups: Group style dictionaries containing layer selectors and colors.
    """
    for group in groups:
        group_nodes = _get_group_nodes(graph, group)
        if not group_nodes:
            continue

        min_x = min(n.x for n in group_nodes)
        max_x = max(n.x + n.width for n in group_nodes)
        min_y = min(n.y for n in group_nodes)
        max_y = max(n.y + n.height for n in group_nodes)

        padding = group.get("padding", 10)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        fill = group.get("fill", (200, 200, 200, 100)) 
        outline = group.get("outline", "black")
        width = group.get("width", 1)

        fill_rgba = get_rgba_tuple(fill)
        outline_rgba = get_rgba_tuple(outline)
        
        pen = aggdraw.Pen(outline_rgba, width)
        brush = aggdraw.Brush(fill_rgba)
        
        draw.rectangle([min_x, min_y, max_x, max_y], pen, brush)


def _draw_group_captions(
    img: Image.Image,
    graph: FunctionalGraph,
    groups: Sequence[Dict[str, Any]],
) -> None:
    """Draw text captions for configured layer groups.

    Captions are centered horizontally below each group's padded bounding box.

    Args:
        img: Destination image.
        graph: Positioned graph.
        groups: Group configuration dictionaries.
    """
    draw = ImageDraw.Draw(img)
    for group in groups:
        caption = group.get("name", group.get("caption"))
        if not caption:
            continue
            
        group_nodes = _get_group_nodes(graph, group)
        if not group_nodes:
            continue

        min_x = min(n.x for n in group_nodes)
        max_x = max(n.x + n.width for n in group_nodes)
        min_y = min(n.y for n in group_nodes)
        max_y = max(n.y + n.height for n in group_nodes)
        
        padding = group.get("padding", 10)
        box_min_x = min_x - padding
        box_max_x = max_x + padding
        box_max_y = max_y + padding

        font = _get_font(group)
        color = group.get("font_color", "black")
        gap = group.get("text_spacing", 5)

        text_w, text_h = _measure_text(draw, caption, font)

        center_x = (box_min_x + box_max_x) / 2
        text_x = center_x - text_w / 2
        text_y = box_max_y + gap
        
        draw.text((text_x, text_y), caption, fill=color, font=font)


def _draw_connectors(
    draw: aggdraw.Draw,
    edges: Iterable[FunctionalEdge],
    nodes: Mapping[int, FunctionalNode],
    render_virtual_nodes: bool,
    connector_fill: Any,
    connector_width: int,
    connector_arrow: bool,
    connector_padding: int,
) -> None:
    """Draw orthogonal connector polylines between graph nodes.

    The routing pass consolidates virtual-node chains, computes shared elbows
    for branch/merge readability, and supports per-node connector overrides for
    width, arrowheads, and padding.

    Args:
        draw: Aggdraw context.
        edges: Directed graph edges.
        nodes: Node lookup by id.
        render_virtual_nodes: Whether virtual nodes are visually rendered.
        connector_fill: Default connector color.
        connector_width: Default connector line width.
        connector_arrow: Default arrowhead toggle.
        connector_padding: Default anchor offset from node faces.
    """
    pen = aggdraw.Pen(connector_fill, connector_width)
    brush = aggdraw.Brush(connector_fill)
    
    outgoing = _outgoing_map(edges)
    incoming = _incoming_map(edges)
    visited: set[Tuple[int, int]] = set()
    paths_by_src: Dict[int, List[List[int]]] = {}

    def anchor(node: FunctionalNode, role: str) -> Tuple[int, int]:
        padding = node.style.get("connector_padding", connector_padding)
        de = getattr(node, 'de', 0)

        real_width = node.width - de
        real_height = node.height - de

        face_x = node.x
        face_y = node.y + de
        
        if node.kind == "virtual" and not render_virtual_nodes:
            x = face_x + real_width / 2
        elif role == "start":
            x = face_x + real_width + padding
        elif role == "end":
            x = face_x - padding
        else:
            x = face_x + real_width / 2
            
        y = face_y + real_height / 2
        return int(round(x)), int(round(y))

    shared_merge_x: Dict[int, int] = {}
    for dst_node_id, src_node_ids in incoming.items():
        if len(src_node_ids) > 1:
            max_start_x = -1
            valid_merge = True
            for src_id in src_node_ids:
                if src_id not in nodes: 
                    valid_merge = False
                    break
                s_node = nodes[src_id]
                sx, _ = anchor(s_node, "start")
                if sx > max_start_x:
                    max_start_x = sx
            
            if valid_merge and dst_node_id in nodes:
                d_node = nodes[dst_node_id]
                dx, _ = anchor(d_node, "end")
                shared_merge_x[dst_node_id] = int(round(max_start_x + (dx - max_start_x) / 2))

    def append_path(start_id: int, next_id: int) -> None:
        path = [start_id]
        prev_id = start_id
        current_id = next_id
        while True:
            path.append(current_id)
            visited.add((prev_id, current_id))
            node = nodes.get(current_id)
            if node is None:
                break
            if node.kind != "virtual" or render_virtual_nodes:
                break
            if len(outgoing.get(current_id, [])) != 1 or len(incoming.get(current_id, [])) != 1:
                break
            next_ids = outgoing.get(current_id, [])
            if not next_ids:
                break
            prev_id = current_id
            current_id = next_ids[0]
            if (prev_id, current_id) in visited:
                break
        paths_by_src.setdefault(start_id, []).append(path)

    for edge in edges:
        if edge.src not in nodes or edge.dst not in nodes:
            continue
        if (edge.src, edge.dst) in visited:
            continue
        src_node = nodes.get(edge.src)
        if src_node is None:
            continue
        if src_node.kind == "virtual" and not render_virtual_nodes and len(incoming.get(edge.src, [])) == 1:
            continue
        append_path(edge.src, edge.dst)

    def add_point(points: List[Tuple[int, int]], x: int, y: int) -> None:
        if not points or points[-1] != (x, y):
            points.append((x, y))

    for src_id, paths in paths_by_src.items():
        start_node = nodes.get(src_id)
        if start_node is None:
            continue
        paths.sort(key=lambda path: nodes[path[-1]].y + nodes[path[-1]].height / 2)
        count = len(paths)
        
        shared_branch_x: Optional[int] = None
        if count > 1:
            x_start, _ = anchor(start_node, "start")
            next_xs: List[int] = []
            for path in paths:
                if len(path) < 2: continue
                next_node = nodes.get(path[1])
                if next_node:
                    next_role = "end" if len(path) == 2 else "mid"
                    x_next, _ = anchor(next_node, next_role)
                    next_xs.append(x_next)
            if next_xs:
                min_x_next = min(next_xs)
                shared_branch_x = int(round(x_start + (min_x_next - x_start) / 2))
                min_mid = x_start + 2
                max_mid = min_x_next - 2
                if min_mid < max_mid:
                    shared_branch_x = max(min_mid, min(max_mid, shared_branch_x))
                else:
                    shared_branch_x = int(round((x_start + min_x_next) / 2))
                if shared_branch_x <= x_start + 1:
                    shared_branch_x = None

        for index, path in enumerate(paths):
            points: List[Tuple[int, int]] = []
            for idx, node_id in enumerate(path):
                node = nodes.get(node_id)
                if node is None: continue
                
                role = "mid"
                if idx == 0: role = "start"
                elif idx == len(path) - 1: role = "end"
                
                x, y = anchor(node, role)
                if not points:
                    add_point(points, x, y)
                    continue

                x1, y1 = points[-1]
                x2, y2 = x, y
                if x2 <= x1 + 1:
                    add_point(points, x2, y2)
                    continue

                mid_x = 0
                
                if idx == len(path) - 1 and node_id in shared_merge_x:
                     mid_x = shared_merge_x[node_id]

                elif idx == 1 and shared_branch_x is not None:
                     mid_x = shared_branch_x
                
                else:
                    mid_x = int(round(x1 + (x2 - x1) / 2))

                min_mid = x1 + 2
                max_mid = x2 - 2
                if min_mid < max_mid:
                    mid_x = max(min_mid, min(max_mid, mid_x))
                else:
                    mid_x = int(round((x1 + x2) / 2))

                add_point(points, mid_x, y1)
                add_point(points, mid_x, y2)
                add_point(points, x2, y2)
            
            src_node = nodes[path[0]]
            use_arrow = src_node.style.get("connector_arrow", connector_arrow)
            use_width = src_node.style.get("connector_width", connector_width)
            
            current_pen = aggdraw.Pen(connector_fill, use_width)

            if len(points) >= 2:
                draw.line([coord for point in points for coord in point], current_pen)

                if use_arrow:
                    end_x, end_y = points[-1]
                    prev_x, prev_y = points[-2]
                    
                    arrow_size = max(8, use_width * 3)
                    
                    if end_x > prev_x:   # Right
                        p1 = (end_x, end_y)
                        p2 = (end_x - arrow_size, end_y - arrow_size // 2)
                        p3 = (end_x - arrow_size, end_y + arrow_size // 2)
                    elif end_x < prev_x: # Left
                        p1 = (end_x, end_y)
                        p2 = (end_x + arrow_size, end_y - arrow_size // 2)
                        p3 = (end_x + arrow_size, end_y + arrow_size // 2)
                    elif end_y > prev_y: # Down
                        p1 = (end_x, end_y)
                        p2 = (end_x - arrow_size // 2, end_y - arrow_size)
                        p3 = (end_x + arrow_size // 2, end_y - arrow_size)
                    else:                # Up
                        p1 = (end_x, end_y)
                        p2 = (end_x - arrow_size // 2, end_y + arrow_size)
                        p3 = (end_x + arrow_size // 2, end_y + arrow_size)

                    draw.polygon([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], current_pen, brush)


def _incoming_map(edges: Sequence[FunctionalEdge]) -> Dict[int, List[int]]:
    incoming: Dict[int, List[int]] = {}
    for edge in edges:
        incoming.setdefault(edge.dst, []).append(edge.src)
    return incoming


def _outgoing_map(edges: Sequence[FunctionalEdge]) -> Dict[int, List[int]]:
    outgoing: Dict[int, List[int]] = {}
    for edge in edges:
        outgoing.setdefault(edge.src, []).append(edge.dst)
    return outgoing


def _resolve_layer_output_shape(layer: Any) -> Optional[Any]:
    shape = getattr(layer, "output_shape", None)
    if shape is not None:
        return _shape_to_tuple(shape)

    output = getattr(layer, "output", None)
    tensor_shape = getattr(output, "shape", None)
    if tensor_shape is not None:
        return _shape_to_tuple(tensor_shape)

    input_shape = getattr(layer, "input_shape", None)
    if input_shape is not None:
        return _shape_to_tuple(input_shape)

    compute_output_shape = getattr(layer, "compute_output_shape", None)
    if callable(compute_output_shape):
        if input_shape is not None:
            try:
                return _shape_to_tuple(compute_output_shape(input_shape))
            except Exception:  # noqa: BLE001
                pass

    return None


def _shape_to_tuple(shape: Any) -> Any:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return shape
    if hasattr(shape, "as_list"):
        try:
            return tuple(shape.as_list())
        except Exception:  # noqa: BLE001
            return tuple(shape)
    if isinstance(shape, list):
        return tuple(shape)
    return shape

def _get_logo_nodes(graph: FunctionalGraph, group: Dict[str, Any]) -> List[FunctionalNode]:
    """Resolve nodes targeted by one logo-group configuration.

    String entries in ``layers`` target layer names. Type entries target all
    nodes whose layer is an instance of that type.
    """
    layers_ref = group.get("layers", [])
    if not layers_ref:
        return []
    
    target_nodes = []
    name_to_nodes = {}
    type_to_nodes = {}
    
    for node in graph.nodes.values():
        if node.kind == "virtual": continue
        
        layer_name = getattr(node.layer, 'name', None)
        if layer_name:
            if layer_name not in name_to_nodes:
                name_to_nodes[layer_name] = []
            name_to_nodes[layer_name].append(node)
            
        layer_type = type(node.layer)
        if layer_type not in type_to_nodes:
            type_to_nodes[layer_type] = []
        type_to_nodes[layer_type].append(node)
        
    for ref in layers_ref:
        if isinstance(ref, str):
            if ref in name_to_nodes:
                target_nodes.extend(name_to_nodes[ref])
        elif isinstance(ref, type):
            if ref in type_to_nodes:
                target_nodes.extend(type_to_nodes[ref])
                
    return target_nodes

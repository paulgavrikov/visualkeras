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
    kind: str = "layer"  # layer, input, output, virtual
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
        styles = resolved["styles"]

        if simple_text_visualization:
            draw_volume = False
            orientation_rotation = None

        if color_map is not None and not isinstance(color_map, dict):
            color_map = dict(color_map)
        if dimension_caps is not None and not isinstance(dimension_caps, dict):
            dimension_caps = dict(dimension_caps)

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

        # Simple text visualization (2D diagram boxes) defaults
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

    ranks = _assign_ranks(graph.nodes, graph.edges)
    graph, ranks = _expand_long_edges(graph, ranks, virtual_node_size)

    if graph.nodes:
        _mark_inputs_outputs(graph)

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
        )
        if component_height <= 0:
            continue
        y_offset += component_height + component_spacing

    _straighten_layout(graph, ranks, row_spacing)

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
        
        # Resolve style early to check for image replacement
        node_style = resolve_style(layer, name)

        # Simple diagram mode uses 2D boxes; disable volume depth
        if simple_text_visualization:
            node_style = dict(node_style)
            node_style["draw_volume"] = False

        # DEBUG
        if "image" in node_style:
            print(f"DEBUG: Found image style for layer: {name} -> {node_style['image']}")
        
        image_path = node_style.get("image")
        node_image = None
        
        # Always calculate standard dimensions first
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

        # Calculate Depth (de)
        use_volume = node_style.get('draw_volume', draw_volume)
        if simple_text_visualization:
            use_volume = False
        de = 0
        if use_volume:
            de = int(width / 3)

        if image_path:
            try:
                # Load the image and convert to RGBA for transparency support
                node_image = Image.open(image_path).convert("RGBA")
                
                # Determine fit mode and axis
                fit_mode = node_style.get("image_fit", image_fit)
                axis = ("z" if simple_text_visualization else node_style.get("image_axis", image_axis))
                
                if fit_mode == "match_aspect":
                    # Resize surface to match image proportions
                    img_w, img_h = node_image.size
                    img_ratio = img_w / img_h
                    
                    if axis == 'z': # Front (Width x Height)
                        # Current surface ratio
                        surf_ratio = width / height
                        if img_ratio > surf_ratio:
                            # Image is wider than surface -> Increase width
                            width = int(height * img_ratio)
                        else:
                            # Image is taller than surface -> Increase height
                            height = int(width / img_ratio)
                            
                    elif axis == 'y': # Top (Width x Depth)
                        # Width x Depth
                        # We adjust 'de' to match aspect ratio relative to 'width'
                        # Ratio = Width / Depth
                        # Depth = Width / Ratio
                        if img_ratio > 0:
                            de = int(width / img_ratio)
                            
                    elif axis == 'x': # Side (Depth x Height)
                        # Depth x Height
                        # Ratio = Depth / Height
                        # Depth = Height * Ratio
                        de = int(height * img_ratio)

                # Apply scale_image factor if present
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

        # Inflate dimensions to reserve space for the 3D projection
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
) -> int:
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
        column_height = sum(graph.nodes[node_id].height for node_id in filtered)
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
            node.x = x_positions.get(rank, 0) + int((column_width - node.width) / 2)
            node.y = y_cursor
            y_cursor += node.height + row_spacing
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
) -> Image.Image:
    max_right = padding
    max_bottom = padding
    for node in graph.nodes.values():
        max_right = max(max_right, node.x + node.width + padding)
        max_bottom = max(max_bottom, node.y + node.height + padding)

    if layered_groups:
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        for group in layered_groups:
            group_nodes = _get_group_nodes(graph, group)
            if not group_nodes:
                continue
            
            # Group Box Bounds
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
            
            # Caption Bounds
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

    # Pre-process logos
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

    # Draw connectors FIRST so they appear behind 3D boxes
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

    # Flush connectors before pasting images
    draw.flush()

    pending_simple_text: List[Tuple[FunctionalNode, Tuple[int, int, int, int]]] = []

    for node in graph.nodes.values():
        if node.kind == "virtual" and not render_virtual_nodes:
            continue

        if simple_text_visualization:
            x1 = int(node.x)
            y1 = int(node.y)
            x2 = int(node.x + node.width)
            y2 = int(node.y + node.height)

            # Resolve fill/outline
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

            # Optional image fill
            if node.image is not None:
                draw.flush()
                fit_mode = node.style.get("image_fit", "fill")
                quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                try:
                    apply_affine_transform(img, node.image, quad, fit_mode)
                except Exception:
                    pass
                draw = aggdraw.Draw(img)

            # Optional logos
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
            continue

        box = Box()
        box.de = getattr(node, 'de', 0)
        box.shade = getattr(node, 'shade', 0)
        box.rotation = orientation_rotation

        # Calculate coordinates for the Front Face (Layout reference)
        box.x1 = node.x
        box.y1 = node.y + box.de
        
        # Recover the real width/height of the face.
        real_width = node.width - box.de
        real_height = node.height - box.de
        
        box.x2 = box.x1 + real_width
        box.y2 = box.y1 + real_height
        
        # Fill/Outline logic
        fill = color_map.get(node.layer_type, {}).get("fill")
        outline = color_map.get(node.layer_type, {}).get("outline")
        if node.kind == "virtual":
            bg = get_rgba_tuple(background_fill)
            fill = fade_color(bg, 10)
            outline = fade_color(get_rgba_tuple(connector_fill), 10)
            
        box.fill = fill if fill is not None else color_wheel.get_color(node.layer_type)
        box.outline = outline if outline is not None else "black"
        
        # Apply style overrides if present
        if node.style.get('fill'):
            box.fill = node.style.get('fill')
        if node.style.get('outline'):
            box.outline = node.style.get('outline')

        # Draw the box (background/volume)
        box.draw(draw, draw_reversed=False)

        if node.image is not None:
            # Apply image to the front face
            draw.flush()
            
        # Re-implement Image pasting using Box's projected faces
        if node.image is not None:
            draw.flush()
            
            fit_mode = node.style.get("image_fit", "fill")
            axis = node.style.get("image_axis", "z")
            
            # Retrieve the projected quad for the target face from the Box
            # Axis 'z' -> Front Face (Index 0)
            # Axis 'y' -> Top Face (Index 4 - assuming Box implementation details below)
            # Axis 'x' -> Right Face (Index 2)
            
            target_face_idx = 0 # Front
            if axis == 'y': target_face_idx = 4 # Top
            elif axis == 'x': target_face_idx = 2 # Right / Side

            # Get Quad from Box (list of tuples)
            quad = box.get_face_quad(target_face_idx)
            
            if quad:
                apply_affine_transform(img, node.image, quad, fit_mode)

            # Re-create aggdraw context
            draw = aggdraw.Draw(img)

        if node.node_id in node_logos:
            draw.flush()
            for group, logo_img in node_logos[node.node_id]:
                # Update draw_node_logo to accept the box object directly for quad extraction
                draw_node_logo(img, box, logo_img, group, draw_volume)
            draw = aggdraw.Draw(img)

    draw.flush()

    if simple_text_visualization and pending_simple_text:
        # Render inside-box labels with bbox-safe text images (avoids clipping)
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
        for index, node in enumerate(sorted(graph.nodes.values(), key=lambda n: n.order)):
            if node.kind == "virtual":
                continue
            text, above = text_callable(index, node.layer)
            text_height = 0
            text_widths = []
            for line in text.split("\n"):
                if hasattr(font, "getsize"):
                    text_widths.append(font.getsize(line)[0])
                    text_height += font.getsize(line)[1]
                else:
                    bbox = font.getbbox(line)
                    text_widths.append(bbox[2])
                    text_height += bbox[3]
            text_height += (len(text.split("\n")) - 1) * text_vspacing
            
            # Center text relative to the Front Face
            de = getattr(node, 'de', 0)
            real_width = node.width - de
            real_height = node.height - de
            
            # Center X = node.x + real_width / 2
            # Center Y = node.y + de + real_height / 2 (approx center of face)
            # Text Y is relative to top/bottom of the face
            
            face_x = node.x
            face_y = node.y + de
            
            text_x = face_x + real_width / 2 - max(text_widths or [0]) / 2
            
            if above:
                # Above the top of the front face
                text_y = face_y - text_height - 4
            else:
                # Below the bottom of the front face
                text_y = face_y + real_height + 4
                
            draw_text.multiline_text(
                (text_x, text_y),
                text,
                font=font,
                fill=font_color,
                spacing=text_vspacing,
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
    row_spacing: int
) -> None:
    """
    Adjusts y positions to align linear connections straight. 
    Accounts for 3D depth (de) to ensure visual centers align.
    """
    nodes_by_rank = {}
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
        # Visual Center = Top of Front Face + Half Height of Front Face
        # Top of Front Face = node.y + node.de
        # Height of Front Face = node.height - node.de
        # Center = (node.y + node.de) + (node.height - node.de) / 2
        # Simplified: node.y + (node.height + node.de) / 2
        de = getattr(node, 'de', 0)
        return node.y + (node.height + de) / 2.0

    def set_visual_center(node: FunctionalNode, center_y: float):
        de = getattr(node, 'de', 0)
        # Inverse of get_visual_center
        # node.y = center_y - (node.height + de) / 2
        new_y = center_y - (node.height + de) / 2.0
        node.y = int(new_y)

    def resolve_collisions(col_nodes: List[FunctionalNode]):
        col_nodes.sort(key=lambda n: n.y)
        if not col_nodes:
            return

        current_y_map = {n.node_id: n.y for n in col_nodes}

        # Forward sweep (push down)
        for i in range(1, len(col_nodes)):
            prev_node = col_nodes[i-1]
            curr_node = col_nodes[i]
            # Ensure spacing between Bounding Boxes
            min_y = current_y_map[prev_node.node_id] + prev_node.height + row_spacing
            if current_y_map[curr_node.node_id] < min_y:
                current_y_map[curr_node.node_id] = min_y

        for node in col_nodes:
            node.y = int(current_y_map[node.node_id])

    # Forward pass (Align Child to Parent)
    for rank in sorted_ranks:
        col_nodes = nodes_by_rank[rank]
        for node in col_nodes:
            parents = incoming[node.node_id]
            if len(parents) == 1:
                parent_id = parents[0]
                if len(outgoing[parent_id]) == 1:
                    parent = graph.nodes[parent_id]
                    # Align to Parent's Visual Center
                    target_center = get_visual_center(parent)
                    set_visual_center(node, target_center)
        
        resolve_collisions(col_nodes)

    # Backward pass (Align Parent to Child)
    for rank in reversed(sorted_ranks):
        col_nodes = nodes_by_rank[rank]
        for node in col_nodes:
            children = outgoing[node.node_id]
            if len(children) == 1:
                child_id = children[0]
                if len(incoming[child_id]) == 1:
                    child = graph.nodes[child_id]
                    # Align to Child's Visual Center
                    target_center = get_visual_center(child)
                    set_visual_center(node, target_center)
        
        resolve_collisions(col_nodes)


def _get_group_nodes(graph: FunctionalGraph, group: Dict[str, Any]) -> List[FunctionalNode]:
    layers = group.get("layers", [])
    if not layers:
        return []
        
    group_nodes = []
    for node in graph.nodes.values():
        # Check if node matches any layer in the group
        for layer_ref in layers:
            if node.layer is layer_ref:
                group_nodes.append(node)
                break
            # Check name match
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
    for group in groups:
        group_nodes = _get_group_nodes(graph, group)
        if not group_nodes:
            continue
            
        # Calculate bounding box
        # Min X: node.x
        # Max X: node.x + node.width
        # Min Y: node.y
        # Max Y: node.y + node.height
        
        min_x = min(n.x for n in group_nodes)
        max_x = max(n.x + n.width for n in group_nodes)
        min_y = min(n.y for n in group_nodes)
        max_y = max(n.y + n.height for n in group_nodes)
        
        # Apply padding
        padding = group.get("padding", 10)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Style
        fill = group.get("fill", (200, 200, 200, 100)) 
        outline = group.get("outline", "black")
        width = group.get("width", 1)
        
        # Convert colors
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
    draw = ImageDraw.Draw(img)
    for group in groups:
        caption = group.get("name", group.get("caption"))
        if not caption:
            continue
            
        group_nodes = _get_group_nodes(graph, group)
        if not group_nodes:
            continue
            
        # Calculate bounding box (same as boxes)
        min_x = min(n.x for n in group_nodes)
        max_x = max(n.x + n.width for n in group_nodes)
        min_y = min(n.y for n in group_nodes)
        max_y = max(n.y + n.height for n in group_nodes)
        
        padding = group.get("padding", 10)
        box_min_x = min_x - padding
        box_max_x = max_x + padding
        box_max_y = max_y + padding
        
        # Config
        font = _get_font(group)
        color = group.get("font_color", "black")
        gap = group.get("text_spacing", 5)
        
        # Calculate text size
        text_w, text_h = _measure_text(draw, caption, font)
        
        # Position: Centered horizontally, below box
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
    pen = aggdraw.Pen(connector_fill, connector_width)
    brush = aggdraw.Brush(connector_fill)
    
    outgoing = _outgoing_map(edges)
    incoming = _incoming_map(edges)
    visited: set[Tuple[int, int]] = set()
    paths_by_src: Dict[int, List[List[int]]] = {}

    def anchor(node: FunctionalNode, role: str) -> Tuple[int, int]:
        # Use the node's specific style, fallback to the global arg
        padding = node.style.get("connector_padding", connector_padding)
        de = getattr(node, 'de', 0)
        
        # Calculate "Real" dimensions (the front face)
        real_width = node.width - de
        real_height = node.height - de
        
        # Front face top-left
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
                # Calculate the shared elbow based on the widest source
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
                # Draw the Line
                draw.line([coord for point in points for coord in point], current_pen)

                # Draw the Arrowhead (if enabled)
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
    layers_ref = group.get("layers", [])
    if not layers_ref:
        return []
    
    target_nodes = []
    
    # Build lookup maps
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
            # Try name first
            if ref in name_to_nodes:
                target_nodes.extend(name_to_nodes[ref])
            # If not found by name, check if it matches a type name? 
            # The requirement says "name specification overriding type specification".
            # This implies we should check both or have a way to distinguish.
            # But usually type is passed as a class object in python.
            # If the user passes a string that happens to be a type name, we might want to support it?
            # For now, assume string is name, type is type.
        elif isinstance(ref, type):
            if ref in type_to_nodes:
                target_nodes.extend(type_to_nodes[ref])
                
    return target_nodes

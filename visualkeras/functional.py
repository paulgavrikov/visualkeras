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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from collections import deque
import warnings

import aggdraw
from PIL import Image, ImageDraw, ImageFont

from .layer_utils import (
    calculate_layer_dimensions,
    extract_primary_shape,
    find_output_layers,
    get_incoming_layers,
    get_layers,
)
from .options import FunctionalOptions, FUNCTIONAL_PRESETS, LAYERED_TEXT_CALLABLES
from .utils import Box, ColorWheel, fade_color, get_rgba_tuple


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
    *,
    options: Union[FunctionalOptions, Mapping[str, Any], None] = None,
    preset: Union[str, None] = None,
) -> Image.Image:
    """Render a functional model using a multi-stream layered layout."""
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

    graph = _build_graph(
        model,
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
        text_callable=text_callable,
        text_vspacing=text_vspacing,
        font=font,
        font_color=font_color,
        render_virtual_nodes=render_virtual_nodes,
    )

    if to_file is not None:
        img.save(to_file)
    return img


def _build_graph(
    model,
    *,
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
) -> FunctionalGraph:
    layers = list(get_layers(model))
    order_map = {id(layer): index for index, layer in enumerate(layers)}
    nodes: Dict[int, FunctionalNode] = {}

    for layer in layers:
        node_id = id(layer)
        name = getattr(layer, "name", None) or f"layer_{order_map[node_id]}"
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
        nodes[node_id] = FunctionalNode(
            layer=layer,
            node_id=node_id,
            name=name,
            layer_type=type(layer),
            shape=shape,
            dims=(int(dims[0]), int(dims[1]), int(dims[2])),
            width=width,
            height=height,
            order=order_map[node_id],
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


def _render_graph(
    graph: FunctionalGraph,
    *,
    color_map: Mapping[type, Mapping[str, Any]],
    background_fill: Any,
    padding: int,
    connector_fill: Any,
    connector_width: int,
    text_callable: Optional[Callable[[int, Any], Tuple[str, bool]]],
    text_vspacing: int,
    font: Optional[ImageFont.ImageFont],
    font_color: Any,
    render_virtual_nodes: bool,
) -> Image.Image:
    max_right = padding
    max_bottom = padding
    for node in graph.nodes.values():
        max_right = max(max_right, node.x + node.width + padding)
        max_bottom = max(max_bottom, node.y + node.height + padding)

    img = Image.new("RGBA", (int(max_right), int(max_bottom)), background_fill)
    draw = aggdraw.Draw(img)
    color_wheel = ColorWheel()

    boxes: Dict[int, Box] = {}
    for node in graph.nodes.values():
        box = Box()
        box.x1 = node.x
        box.y1 = node.y
        box.x2 = node.x + node.width
        box.y2 = node.y + node.height
        box.de = 0
        box.shade = 0
        fill = color_map.get(node.layer_type, {}).get("fill")
        outline = color_map.get(node.layer_type, {}).get("outline")
        if node.kind == "virtual":
            bg = get_rgba_tuple(background_fill)
            fill = fade_color(bg, 10)
            outline = fade_color(get_rgba_tuple(connector_fill), 10)
        box.fill = fill if fill is not None else color_wheel.get_color(node.layer_type)
        box.outline = outline if outline is not None else "black"
        boxes[node.node_id] = box

    _draw_connectors(
        draw,
        graph.edges,
        graph.nodes,
        render_virtual_nodes,
        connector_fill,
        connector_width,
    )

    for node_id, box in boxes.items():
        node = graph.nodes[node_id]
        if node.kind == "virtual" and not render_virtual_nodes:
            continue
        box.draw(draw, draw_reversed=False)

    draw.flush()

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
            text_x = node.x + node.width / 2 - max(text_widths or [0]) / 2
            text_y = node.y - text_height - 4 if above else node.y + node.height + 4
            draw_text.multiline_text(
                (text_x, text_y),
                text,
                font=font,
                fill=font_color,
                spacing=text_vspacing,
            )

    return img

def _straighten_layout(
    graph: FunctionalGraph,
    ranks: Dict[int, int],
    row_spacing: int
) -> None:
    """
    Adjusts y-positions to align linear connections straight.
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

    for rank in sorted_ranks:
        col_nodes = nodes_by_rank[rank]
        col_nodes.sort(key=lambda n: n.y)
        
        proposed_centers = {}
        for node in col_nodes:
            parents = incoming[node.node_id]

            if len(parents) == 1:
                parent_id = parents[0]
                if len(outgoing[parent_id]) == 1:
                    parent = graph.nodes[parent_id]
                    # Calculate center alignment
                    parent_center = parent.y + parent.height / 2
                    proposed_centers[node.node_id] = parent_center

        if not col_nodes:
            continue

        current_y_map = {n.node_id: n.y for n in col_nodes}
        
        for node in col_nodes:
            if node.node_id in proposed_centers:
                desired_center = proposed_centers[node.node_id]
                new_y = desired_center - node.height / 2
                current_y_map[node.node_id] = int(new_y)

        for i in range(1, len(col_nodes)):
            prev_node = col_nodes[i-1]
            curr_node = col_nodes[i]
            min_y = current_y_map[prev_node.node_id] + prev_node.height + row_spacing
            if current_y_map[curr_node.node_id] < min_y:
                current_y_map[curr_node.node_id] = min_y

        for node in col_nodes:
            node.y = int(current_y_map[node.node_id])


def _draw_connectors(
    draw: aggdraw.Draw,
    edges: Iterable[FunctionalEdge],
    nodes: Mapping[int, FunctionalNode],
    render_virtual_nodes: bool,
    connector_fill: Any,
    connector_width: int,
) -> None:
    pen = aggdraw.Pen(connector_fill, connector_width)
    outgoing = _outgoing_map(edges)
    incoming = _incoming_map(edges)
    visited: set[Tuple[int, int]] = set()
    paths_by_src: Dict[int, List[List[int]]] = {}

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

    def anchor(node: FunctionalNode, role: str) -> Tuple[int, int]:
        if node.kind == "virtual" and not render_virtual_nodes:
            x = node.x + node.width / 2
        elif role == "start":
            x = node.x + node.width
        elif role == "end":
            x = node.x
        else:
            x = node.x + node.width / 2
        y = node.y + node.height / 2
        return int(round(x)), int(round(y))

    def add_point(points: List[Tuple[int, int]], x: int, y: int) -> None:
        if not points or points[-1] != (x, y):
            points.append((x, y))

    for src_id, paths in paths_by_src.items():
        start_node = nodes.get(src_id)
        if start_node is None:
            continue
        paths.sort(key=lambda path: nodes[path[-1]].y + nodes[path[-1]].height / 2)
        count = len(paths)
        spread = max(4, connector_width * 2)

        # When a node fans out to multiple downstream nodes, using per-edge offsets
        # in the first elbow ("mid_x") causes forks to originate from different x
        # positions. Compute a shared first elbow so the branching looks consistent.
        shared_mid_x: Optional[int] = None
        if count > 1:
            x_start, _ = anchor(start_node, "start")
            next_xs: List[int] = []
            for path in paths:
                if len(path) < 2:
                    continue
                next_node = nodes.get(path[1])
                if next_node is None:
                    continue
                next_role = "end" if len(path) == 2 else "mid"
                x_next, _ = anchor(next_node, next_role)
                next_xs.append(x_next)
            if next_xs:
                min_x_next = min(next_xs)
                shared_mid_x = int(round(x_start + (min_x_next - x_start) / 2))
                min_mid = x_start + 2
                max_mid = min_x_next - 2
                if min_mid < max_mid:
                    shared_mid_x = max(min_mid, min(max_mid, shared_mid_x))
                else:
                    shared_mid_x = int(round((x_start + min_x_next) / 2))
                if shared_mid_x <= x_start + 1:
                    shared_mid_x = None

        for index, path in enumerate(paths):
            offset = 0
            if count > 1:
                offset = int((index - (count - 1) / 2) * spread)

            points: List[Tuple[int, int]] = []
            for idx, node_id in enumerate(path):
                node = nodes.get(node_id)
                if node is None:
                    continue
                role = "mid"
                if idx == 0:
                    role = "start"
                elif idx == len(path) - 1:
                    role = "end"
                x, y = anchor(node, role)
                if not points:
                    add_point(points, x, y)
                    continue

                x1, y1 = points[-1]
                x2, y2 = x, y
                if x2 <= x1 + 1:
                    add_point(points, x2, y2)
                    continue

                if shared_mid_x is not None:
                    mid_x = shared_mid_x
                else:
                    mid_x = int(round(x1 + (x2 - x1) / 2)) + offset

                min_mid = x1 + 2
                max_mid = x2 - 2
                if min_mid < max_mid:
                    mid_x = max(min_mid, min(max_mid, mid_x))
                else:
                    mid_x = int(round((x1 + x2) / 2))

                add_point(points, mid_x, y1)
                add_point(points, mid_x, y2)
                add_point(points, x2, y2)

            if len(points) >= 2:
                draw.line([coord for point in points for coord in point], pen)


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

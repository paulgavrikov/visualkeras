from typing import Any, Dict, Mapping, Optional, Union
import aggdraw
from PIL import Image, ImageDraw
from math import ceil
import warnings
from .utils import *
from .layer_utils import *
from .options import GraphOptions, GRAPH_PRESETS

class _DummyLayer:

    def __init__(self, name, units=None):
        if units:
            self.units = units
        self.name = name


def graph_view(model, to_file: str = None,
               color_map: dict = None, node_size: int = 50,
               background_fill: Any = 'white', padding: int = 10,
               layer_spacing: int = 250, node_spacing: int = 10, connector_fill: Any = 'gray',
               connector_width: int = 1, ellipsize_after: int = 10,
               inout_as_tensor: bool = True, show_neurons: bool = True,
               styles: Optional[Mapping[Union[str, type], Dict[str, Any]]] = None,
               *, options: Union[GraphOptions, Mapping[str, Any], None] = None,
               preset: Union[str, None] = None) -> Image:
    """
    Generates a architecture visualization for a given linear keras model (i.e. one input and output tensor for each
    layer) in graph style.

    :param model: A keras model that will be visualized.
    :param to_file: Path to the file to write the created image to. If the image does not exist yet it will be created,
    else overwritten. Image type is inferred from the file ending. Providing None will disable writing.
    :param color_map: Dict defining fill and outline for each layer by class type. Will fallback to default values for
    not specified classes.
    :param node_size: Size in pixel each node will have.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :param padding: Distance in pixel before the first and after the last layer.
    :param layer_spacing: Spacing in pixel between two layers
    :param node_spacing: Spacing in pixel between nodes
    :param connector_fill: Color for the connectors. Can be str or (R,G,B,A).
    :param connector_width: Line-width of the connectors in pixel.
    :param ellipsize_after: Maximum number of neurons per layer to draw. If a layer is exceeding this, the remaining
    neurons will be drawn as ellipses.
    :param inout_as_tensor: If True there will be one input and output node for each tensor, else the tensor will be
    flattened and one node for each scalar will be created (e.g. a (10, 10) shape will be represented by 100 nodes)
    :param show_neurons: If True a node for each neuron in supported layers is created (constrained by ellipsize_after),
    else each layer is represented by a node
    :param styles: Mapping keyed by layer class or layer name. Values override defaults on a per-layer basis.
        Supported keys include: fill, outline, node_size, node_spacing, layer_spacing,
        connector_fill, connector_width, ellipsize_after, show_neurons, box_scale.
    :param options: Optional ``GraphOptions`` (or mapping) providing a configuration bundle.
        Explicit keyword arguments override the bundle.
    :param preset: Name of a preset from ``visualkeras.GRAPH_PRESETS`` to use as the base style.

    :return: Generated architecture image.
    """
    using_presets = options is not None or preset is not None

    if not using_presets:
        defaults = GraphOptions().to_kwargs()
        defaults.update({
            "to_file": None,
            "color_map": None,
            "node_size": 50,
            "background_fill": 'white',
            "padding": 10,
            "layer_spacing": 250,
            "node_spacing": 10,
            "connector_fill": 'gray',
            "connector_width": 1,
            "ellipsize_after": 10,
            "inout_as_tensor": True,
            "show_neurons": True,
        })

        current_params = {
            "to_file": to_file,
            "color_map": color_map,
            "node_size": node_size,
            "background_fill": background_fill,
            "padding": padding,
            "layer_spacing": layer_spacing,
            "node_spacing": node_spacing,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "ellipsize_after": ellipsize_after,
            "inout_as_tensor": inout_as_tensor,
            "show_neurons": show_neurons,
        }

        custom_keys = [
            key for key, value in current_params.items()
            if key in defaults and value != defaults[key]
        ]

        if len(custom_keys) >= 4:
            warnings.warn(
                "graph_view received many custom keyword arguments. "
                "Consider using visualkeras.show(..., mode='graph', preset=...) and the GraphOptions dataclass for a simpler workflow.",
                UserWarning,
                stacklevel=2,
            )

    if preset is not None or options is not None:
        defaults = GraphOptions().to_kwargs()
        defaults["color_map"] = None

        resolved = dict(defaults)

        if preset is not None:
            try:
                resolved.update(GRAPH_PRESETS[preset].to_kwargs())
            except KeyError as exc:
                available = ", ".join(sorted(GRAPH_PRESETS.keys()))
                raise ValueError(
                    f"Unknown graph preset '{preset}'. Available presets: {available}"
                ) from exc

        if options is not None:
            if isinstance(options, GraphOptions):
                option_values = options.to_kwargs()
            elif isinstance(options, Mapping):
                option_values = dict(options)
            else:
                raise TypeError(
                    "options must be a GraphOptions instance or a mapping of keyword arguments."
                )
            resolved.update(option_values)

        explicit_values = {
            "to_file": to_file,
            "color_map": color_map,
            "node_size": node_size,
            "background_fill": background_fill,
            "padding": padding,
            "layer_spacing": layer_spacing,
            "node_spacing": node_spacing,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "ellipsize_after": ellipsize_after,
            "inout_as_tensor": inout_as_tensor,
            "show_neurons": show_neurons,
        }

        for key, value in explicit_values.items():
            if key not in defaults:
                continue
            if value != defaults[key]:
                resolved[key] = value

        to_file = resolved["to_file"]
        color_map = resolved["color_map"]
        node_size = resolved["node_size"]
        background_fill = resolved["background_fill"]
        padding = resolved["padding"]
        layer_spacing = resolved["layer_spacing"]
        node_spacing = resolved["node_spacing"]
        connector_fill = resolved["connector_fill"]
        connector_width = resolved["connector_width"]
        ellipsize_after = resolved["ellipsize_after"]
        inout_as_tensor = resolved["inout_as_tensor"]
        show_neurons = resolved["show_neurons"]

        if color_map is not None and not isinstance(color_map, dict):
            color_map = dict(color_map)

    if color_map is None:
        color_map = dict()

    if styles is None:
        styles = {}

    global_defaults = {
        "fill": None,
        "outline": "black",
        "node_size": node_size,
        "node_spacing": node_spacing,
        "layer_spacing": layer_spacing,
        "connector_fill": connector_fill,
        "connector_width": connector_width,
        "ellipsize_after": ellipsize_after,
        "show_neurons": show_neurons,
        "box_scale": 3,
    }

    # Iterate over the model to compute bounds and generate boxes

    layers = list()
    layer_y = list()

    # Determine output names compatible with both Keras versions
    if hasattr(model, 'output_names'):
        # Older versions of Keras
        output_names = model.output_names
    else:
        # Newer versions of Keras
        output_names = []
        for output in model.outputs:
            if hasattr(output, '_keras_history'):
                # Get the layer that produced the output
                layer = output._keras_history[0]
                output_names.append(layer.name)
            else:
                # Fallback
                # Use the tensor's name or a default name if keras_history is not available
                output_names.append(getattr(output, 'name', f'output_{len(output_names)}'))    # Attach helper layers

    id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    model_layers = model_to_hierarchy_lists(model, id_to_num_mapping, adj_matrix)
    
    # Add fake output layers only when needed
    # When inout_as_tensor=False, only add dummy layers if output-producing layers
    # are not in the last hierarchy level (to avoid duplication)
    should_add_dummy_outputs = inout_as_tensor
    
    if not inout_as_tensor:
        # Check if all output-producing layers are in the last hierarchy level
        last_level_layers = model_layers[-1] if model_layers else []
        layers_producing_outputs = []
        
        for output_tensor in model.outputs:
            for layer in model.layers:
                if hasattr(layer, 'output') and layer.output is output_tensor:
                    layers_producing_outputs.append(layer)
                    break
        
        # Only add dummy outputs if some output-producing layers are NOT in the last level
        should_add_dummy_outputs = not all(layer in last_level_layers for layer in layers_producing_outputs)
    
    if should_add_dummy_outputs:
        # Normalize output_shape to always be a list of tuples
        if isinstance(model.output_shape, tuple):
            # Single output model: output_shape is a tuple, convert to list of tuples
            output_shapes = [model.output_shape]
        else:
            # Multi-output model: output_shape is already a list of tuples
            output_shapes = model.output_shape
        
        model_layers.append([
            _DummyLayer(
                output_names[i],
                None if inout_as_tensor else self_multiply(output_shapes[i])
            )
            for i in range(len(model.outputs))
        ])
        id_to_num_mapping, adj_matrix = augment_output_layers(model, model_layers[-1], id_to_num_mapping, adj_matrix)

    # Create architecture

    current_x = padding  # + input_label_size[0] + text_padding
    max_right = padding

    id_to_node_list_map = dict()
    layer_counter = 0

    for index, layer_list in enumerate(model_layers):
        current_y = 0
        nodes = []
        column_width = 0
        column_spacing = layer_spacing
        last_node_spacing = node_spacing
        last_node_size = node_size
        for layer in layer_list:
            layer_name = getattr(layer, "name", None)
            if not layer_name:
                layer_name = f"layer_{layer_counter}"
            layer_counter += 1

            legacy_color = color_map.get(type(layer), {})
            current_defaults = global_defaults.copy()
            current_defaults.update(legacy_color)
            style = resolve_style(layer, layer_name, styles, current_defaults)

            local_node_size = style.get("node_size", node_size)
            local_node_spacing = style.get("node_spacing", node_spacing)
            local_layer_spacing = style.get("layer_spacing", layer_spacing)
            local_ellipsize_after = style.get("ellipsize_after", ellipsize_after)
            local_show_neurons = style.get("show_neurons", show_neurons)
            box_scale = style.get("box_scale", 3)

            if local_ellipsize_after is not None:
                local_ellipsize_after = int(local_ellipsize_after)

            column_width = max(column_width, local_node_size)
            column_spacing = max(column_spacing, local_layer_spacing)
            last_node_spacing = local_node_spacing
            last_node_size = local_node_size

            is_box = True
            units = 1
            
            if local_show_neurons:
                if hasattr(layer, 'units'):
                    is_box = False
                    units = layer.units
                elif hasattr(layer, 'filters'):
                    is_box = False
                    units = layer.filters
                elif is_internal_input(layer) and not inout_as_tensor:
                    is_box = False
                    # Normalize input_shape to handle both tuple and list formats
                    input_shape = layer.input_shape
                    if isinstance(input_shape, tuple):
                        shape = input_shape
                    elif isinstance(input_shape, list) and len(input_shape) == 1:
                        shape = input_shape[0]
                    else:
                        raise RuntimeError(f"not supported input shape {input_shape}")
                    units = self_multiply(shape)

            if local_ellipsize_after is None or local_ellipsize_after <= 0:
                n = units
            else:
                n = min(units, local_ellipsize_after)
            layer_nodes = list()

            for i in range(n):
                scale = 1
                if not is_box:
                    if local_ellipsize_after and local_ellipsize_after > 1 and i == local_ellipsize_after - 2:
                        c = Ellipses()
                    else:
                        c = Circle()
                else:
                    c = Box()
                    scale = box_scale

                c.x1 = current_x
                c.y1 = current_y
                c.x2 = c.x1 + local_node_size
                c.y2 = c.y1 + local_node_size * scale

                current_y = c.y2 + local_node_spacing
                max_right = max(max_right, c.x2)

                c.fill = style.get('fill') if style.get('fill') is not None else 'orange'
                c.outline = style.get('outline') if style.get('outline') is not None else 'black'
                c.style = style

                layer_nodes.append(c)

            id_to_node_list_map[id(layer)] = layer_nodes
            nodes.extend(layer_nodes)
            current_y += 2 * local_node_size

        layer_y.append(current_y - last_node_spacing - 2 * last_node_size)
        layers.append(nodes)
        current_x += column_width + column_spacing

    # Generate image

    img_width = max_right + padding
    img_height = max(*layer_y) + 2 * padding
    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)

    draw = aggdraw.Draw(img)

    # y correction (centering)
    for i, layer in enumerate(layers):
        y_off = (img.height - layer_y[i]) / 2
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off

    for start_idx, end_idx in zip(*np.where(adj_matrix > 0)):
        start_id = next(get_keys_by_value(id_to_num_mapping, start_idx))
        end_id = next(get_keys_by_value(id_to_num_mapping, end_idx))

        start_layer_list = id_to_node_list_map[start_id]
        end_layer_list = id_to_node_list_map[end_id]

        # draw connectors
        for start_node_idx, start_node in enumerate(start_layer_list):
            for end_node in end_layer_list:
                if not isinstance(start_node, Ellipses) and not isinstance(end_node, Ellipses):
                    _draw_connector(draw, start_node, end_node, color=connector_fill, width=connector_width)

    for i, layer in enumerate(layers):
        for node_index, node in enumerate(layer):
            node.draw(draw)

    draw.flush()

    if to_file is not None:
        img.save(to_file)

    return img


def _draw_connector(draw, start_node, end_node, color, width):
    style = getattr(start_node, "style", {}) or {}
    use_color = style.get("connector_fill", color)
    use_width = style.get("connector_width", width)
    pen = aggdraw.Pen(get_rgba_tuple(use_color), use_width)
    x1 = start_node.x2
    y1 = start_node.y1 + (start_node.y2 - start_node.y1) / 2
    x2 = end_node.x1
    y2 = end_node.y1 + (end_node.y2 - end_node.y1) / 2
    draw.line([x1, y1, x2, y2], pen)

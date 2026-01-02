from typing import Any, Dict, Mapping, Optional, Union, List, Sequence, Tuple
import aggdraw
from PIL import Image, ImageDraw, ImageFont
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
               image_fit: str = 'contain',
               circular_crop: bool = True,
               layered_groups: Optional[Sequence[Dict[str, Any]]] = None,
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
    :param image_fit: Default image fitting mode ('contain', 'cover', 'fill', 'match_aspect').
    :param circular_crop: If True, images on nodes will be cropped to a circle. Default is True.
    :param layered_groups: List of dicts defining groups of layers to highlight with a background rectangle.
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
            "styles": None,
            "image_fit": 'contain',
            "circular_crop": True,
            "layered_groups": None,
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
            "styles": styles,
            "image_fit": image_fit,
            "circular_crop": circular_crop,
            "layered_groups": layered_groups,
        }

        custom_keys = [
            key for key, value in current_params.items()
            if key in defaults and value != defaults[key]
        ]

        if len(custom_keys) >= 5:
            warnings.warn(
                "graph_view received many custom keyword arguments. "
                "Consider using visualkeras.show(..., mode='graph', preset=...) and the GraphOptions dataclass for a simpler workflow.",
                UserWarning,
                stacklevel=2,
            )

    if preset is not None or options is not None:
        defaults = GraphOptions().to_kwargs()
        defaults["color_map"] = None
        defaults["styles"] = None
        defaults["layered_groups"] = None

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
            "styles": styles,
            "image_fit": image_fit,
            "circular_crop": circular_crop,
            "layered_groups": layered_groups,
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
        styles = resolved["styles"]
        image_fit = resolved["image_fit"]
        circular_crop = resolved["circular_crop"]
        layered_groups = resolved["layered_groups"]

        if color_map is not None and not isinstance(color_map, dict):
            color_map = dict(color_map)

    if color_map is None:
        color_map = dict()

    if styles is not None and not isinstance(styles, dict):
        styles = dict(styles)

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
        "image_fit": image_fit,
        "circular_crop": circular_crop,
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

            # --- Image Handling ---
            image_path = style.get("image")
            image_indices = style.get("image_indices")
            node_image = None
            if image_path:
                try:
                    node_image = Image.open(image_path).convert("RGBA")
                except Exception as e:
                    warnings.warn(f"Could not load image {image_path} for layer {layer_name}: {e}")

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
                
                if node_image and not isinstance(c, Ellipses):
                    if image_indices is None or i in image_indices:
                        c.image = node_image
                        c.image_fit = style.get("image_fit", image_fit)
                        c.circular_crop = style.get("circular_crop", circular_crop)

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
    
    # Calculate y offsets for centering
    y_offsets = []
    for i, layer in enumerate(layers):
        y_off = (img_height - layer_y[i]) / 2
        y_offsets.append(y_off)
        
    # Apply offsets to nodes
    for i, layer in enumerate(layers):
        y_off = y_offsets[i]
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off
            
    # Calculate group bounds and expand image if needed
    min_x, min_y = 0, 0
    max_x, max_y = img_width, img_height
    
    if layered_groups:
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for group in layered_groups:
            group_nodes = _get_graph_group_nodes(id_to_node_list_map, group, model)
            if not group_nodes: continue
            
            g_min_x = float('inf')
            g_max_x = float('-inf')
            g_min_y = float('inf')
            g_max_y = float('-inf')
            
            for node in group_nodes:
                g_min_x = min(g_min_x, node.x1)
                g_max_x = max(g_max_x, node.x2)
                g_min_y = min(g_min_y, node.y1)
                g_max_y = max(g_max_y, node.y2)
                
            padding_val = group.get("padding", 10)
            g_min_x -= padding_val
            g_max_x += padding_val
            g_min_y -= padding_val
            g_max_y += padding_val
            
            min_x = min(min_x, g_min_x)
            max_x = max(max_x, g_max_x)
            min_y = min(min_y, g_min_y)
            max_y = max(max_y, g_max_y)
            
            # Caption
            caption = group.get("name", group.get("caption"))
            if caption:
                font = _get_font(group)
                text_w, text_h = _measure_text(dummy_draw, caption, font)
                
                center_x = (g_min_x + g_max_x) / 2
                text_x1 = center_x - text_w / 2
                text_x2 = center_x + text_w / 2
                
                gap = group.get("text_spacing", 5)
                text_bottom = g_max_y + gap + text_h
                
                min_x = min(min_x, text_x1)
                max_x = max(max_x, text_x2)
                max_y = max(max_y, text_bottom)

    final_width = max_x - min_x
    final_height = max_y - min_y
    
    img = Image.new('RGBA', (int(ceil(final_width)), int(ceil(final_height))), background_fill)
    draw = aggdraw.Draw(img)
    
    # Shift nodes if needed
    shift_x = -min_x
    shift_y = -min_y
    
    if shift_x != 0 or shift_y != 0:
        for layer in layers:
            for node in layer:
                node.x1 += shift_x
                node.x2 += shift_x
                node.y1 += shift_y
                node.y2 += shift_y
                
    if layered_groups:
        _draw_graph_group_boxes(draw, id_to_node_list_map, layered_groups, model)

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
            
            if getattr(node, 'image', None):
                draw.flush()
                image = node.image
                fit = node.image_fit
                w = node.x2 - node.x1
                h = node.y2 - node.y1
                
                resized = resize_image_to_fit(image, int(w), int(h), fit)
                
                if getattr(node, 'circular_crop', False):
                    # Supersampling for anti-aliasing
                    super_scale = 4
                    mask_w = int(w) * super_scale
                    mask_h = int(h) * super_scale
                    
                    mask = Image.new("L", (mask_w, mask_h), 0)
                    mask_draw = ImageDraw.Draw(mask)
                    mask_draw.ellipse((0, 0, mask_w, mask_h), fill=255)
                    
                    # Resize mask down smoothly
                    mask = mask.resize((int(w), int(h)), Image.LANCZOS)
                    
                    # Apply mask to the resized image
                    if resized.mode == 'RGBA':
                        cropped = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
                        cropped.paste(resized, (0, 0), mask=mask)
                        resized = cropped
                    else:
                        resized.putalpha(mask)

                img.paste(resized, (int(node.x1), int(node.y1)), resized)
                draw = aggdraw.Draw(img)
                
                # Draw the node outline on top (transparent fill)
                # We access _fill directly to avoid the setter logic which might fail on None
                original_fill = node._fill
                node._fill = (0, 0, 0, 0) 
                node.draw(draw)
                node._fill = original_fill
                
            else:
                node.draw(draw)

    if layered_groups:
        _draw_graph_group_captions(img, id_to_node_list_map, layered_groups, model)

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

def _get_graph_group_nodes(id_to_node_list_map: Dict[int, List[Any]], group: Dict[str, Any], model) -> List[Any]:
    layers_ref = group.get("layers", [])
    if not layers_ref:
        return []
        
    group_nodes = []
    
    # Build lookup maps
    name_to_layer = {}
    for layer in model.layers:
        name = getattr(layer, 'name', None)
        if name:
            name_to_layer[name] = layer
            
    for ref in layers_ref:
        layer = None
        if isinstance(ref, str):
            layer = name_to_layer.get(ref)
        else:
            layer = ref
            
        if layer:
            nodes = id_to_node_list_map.get(id(layer))
            if nodes:
                group_nodes.extend(nodes)
                
    return group_nodes

def _draw_graph_group_boxes(draw, id_to_node_list_map, groups, model):
    for group in groups:
        nodes = _get_graph_group_nodes(id_to_node_list_map, group, model)
        if not nodes: continue
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for node in nodes:
            min_x = min(min_x, node.x1)
            max_x = max(max_x, node.x2)
            min_y = min(min_y, node.y1)
            max_y = max(max_y, node.y2)
            
        padding = group.get("padding", 10)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        fill = group.get("fill", (200, 200, 200, 100))
        outline = group.get("outline", "black")
        width = group.get("width", 1)
        
        pen = aggdraw.Pen(get_rgba_tuple(outline), width)
        brush = aggdraw.Brush(get_rgba_tuple(fill))
        
        draw.rectangle([min_x, min_y, max_x, max_y], pen, brush)

def _draw_graph_group_captions(img, id_to_node_list_map, groups, model):
    draw = ImageDraw.Draw(img)
    for group in groups:
        caption = group.get("name", group.get("caption"))
        if not caption: continue
        
        nodes = _get_graph_group_nodes(id_to_node_list_map, group, model)
        if not nodes: continue
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for node in nodes:
            min_x = min(min_x, node.x1)
            max_x = max(max_x, node.x2)
            min_y = min(min_y, node.y1)
            max_y = max(max_y, node.y2)
            
        padding = group.get("padding", 10)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        font = _get_font(group)
        color = group.get("font_color", "black")
        gap = group.get("text_spacing", 5)
        
        text_w, text_h = _measure_text(draw, caption, font)
        
        center_x = (min_x + max_x) / 2
        text_x = center_x - text_w / 2
        text_y = max_y + gap
        
        draw.text((text_x, text_y), caption, fill=color, font=font)

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



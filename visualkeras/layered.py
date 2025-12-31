from typing import Any, Callable, Mapping, Optional, Union, List, Dict, Sequence, Tuple
import aggdraw
from PIL import ImageFont
from math import ceil
from .utils import *
from .layer_utils import *
from .options import LayeredOptions, LAYERED_PRESETS, LAYERED_TEXT_CALLABLES
import warnings

try:
    from tensorflow.keras import layers
except:
    try:
        from tensorflow.python.keras import layers
    except:
        try:
            from keras import layers
        except:
            warnings.warn("Could not import the 'layers' module from Keras. text_callable will not work.")

_BUILT_IN_TEXT_CALLABLES = tuple(LAYERED_TEXT_CALLABLES.values())


def _resolve_layer_output_shape(layer) -> Any:
    """
    Attempt to retrieve a layer's output shape across keras/tensorflow versions.

    Prefers an explicit ``output_shape`` attribute, falls back to the tensor's
    shape, and finally tries ``compute_output_shape`` when available.
    """
    shape = getattr(layer, "output_shape", None)
    if shape is not None:
        return _shape_to_tuple(shape)

    output = getattr(layer, "output", None)
    tensor_shape = getattr(output, "shape", None)
    if tensor_shape is not None:
        return _shape_to_tuple(tensor_shape)

    compute_output_shape = getattr(layer, "compute_output_shape", None)
    if callable(compute_output_shape):
        input_shape = getattr(layer, "input_shape", None)
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

def _get_group_boxes(boxes: List[Box], group: Dict[str, Any]) -> List[Box]:
    layers_ref = group.get("layers", [])
    if not layers_ref:
        return []
        
    group_boxes = []
    for box in boxes:
        if not hasattr(box, 'layer'):
            continue
            
        layer = box.layer
        # Check if node matches any layer in the group
        for layer_ref in layers_ref:
            if layer is layer_ref:
                group_boxes.append(box)
                break
            # Check name match
            layer_name = getattr(layer, 'name', '')
            if isinstance(layer_ref, str) and (layer_name == layer_ref):
                group_boxes.append(box)
                break
    return group_boxes


def _draw_layered_group_boxes(draw, boxes, groups, draw_reversed):
    for group in groups:
        group_boxes = _get_group_boxes(boxes, group)
        if not group_boxes: continue
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for box in group_boxes:
            if draw_reversed:
                min_x = min(min_x, box.x1 - box.de)
                max_x = max(max_x, box.x2)
                min_y = min(min_y, box.y1 - box.de)
                max_y = max(max_y, box.y2)
            else:
                min_x = min(min_x, box.x1)
                max_x = max(max_x, box.x2 + box.de)
                min_y = min(min_y, box.y1 - box.de)
                max_y = max(max_y, box.y2)
                
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


def _draw_layered_group_captions(img, boxes, groups, draw_reversed):
    draw = ImageDraw.Draw(img)
    for group in groups:
        caption = group.get("name", group.get("caption"))
        if not caption: continue
        
        group_boxes = _get_group_boxes(boxes, group)
        if not group_boxes: continue
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for box in group_boxes:
            if draw_reversed:
                min_x = min(min_x, box.x1 - box.de)
                max_x = max(max_x, box.x2)
                min_y = min(min_y, box.y1 - box.de)
                max_y = max(max_y, box.y2)
            else:
                min_x = min(min_x, box.x1)
                max_x = max(max_x, box.x2 + box.de)
                min_y = min(min_y, box.y1 - box.de)
                max_y = max(max_y, box.y2)
                
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


def layered_view(model, 
                 to_file: str = None, 
                 min_z: int = 20, 
                 min_xy: int = 20, 
                 max_z: int = 400,
                 max_xy: int = 2000,
                 scale_z: float = 1.5, 
                 scale_xy: float = 4, 
                 type_ignore: list = None, 
                 index_ignore: list = None,
                 color_map: dict = None, 
                 one_dim_orientation: str = 'z', 
                 index_2D: list = [],
                 background_fill: Any = 'white', 
                 draw_volume: bool = True,
                 draw_reversed: bool = False, 
                 padding: int = 10,
                 text_callable: Callable[[int, layers.Layer], tuple] = None,
                 text_vspacing: int = 4,
                 spacing: int = 10, 
                 draw_funnel: bool = True, 
                 shade_step=10, 
                 legend: bool = False,
                 legend_text_spacing_offset = 15,
                 font: ImageFont = None, 
                 font_color: Any = 'black', 
                 show_dimension=False,
                 sizing_mode: str = 'accurate',
                 dimension_caps: dict = None,
                 relative_base_size: int = 20,
                 connector_fill: Any = 'gray',
                 connector_width: int = 1,
                 image_fit: str = "fill",
                 image_axis: str = "z",
                 layered_groups: Optional[Sequence[Dict[str, Any]]] = None,
                 styles: Optional[Mapping[Union[str, type], Dict[str, Any]]] = None,
                 *,
                 options: Union[LayeredOptions, Mapping[str, Any], None] = None,
                 preset: Union[str, None] = None) -> Image:
    """
    Generates a architecture visualization for a given linear keras model (i.e. one input and output tensor for each
    layer) in layered style (great for CNN).

    :param model: A keras model that will be visualized.
    :param to_file: Path to the file to write the created image to. If the image does not exist yet it will be created, else overwritten. Image type is inferred from the file ending. Providing None will disable writing.
    :param min_z: Minimum z size in pixel a layer will have.
    :param min_xy: Minimum x and y size in pixel a layer will have.
    :param max_z: Maximum z size in pixel a layer will have.
    :param max_xy: Maximum x and y size in pixel a layer will have.
    :param scale_z: Scalar multiplier for the z size of each layer.
    :param scale_xy: Scalar multiplier for the x and y size of each layer.
    :param type_ignore: List of layer types in the keras model to ignore during drawing.
    :param index_ignore: List of layer indexes in the keras model to ignore during drawing.
    :param color_map: Dict defining fill and outline for each layer by class type. Will fallback to default values for not specified classes.
    :param one_dim_orientation: Axis on which one dimensional layers should be drawn. Can  be 'x', 'y' or 'z'.
    :param index_2D: When draw_volume is True, the indexes in this list will be drawn in 2D.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :param draw_volume: Flag to switch between 3D volumetric view and 2D box view.
    :param draw_reversed: Draw 3D boxes reversed, going from front-right to back-left.
    :param padding: Distance in pixel before the first and after the last layer.
    :param text_callable: Callable receiving ``(layer_index, layer)`` and returning a
        ``(text, above)`` tuple describing annotations to draw per layer. Built-in
        presets are available via ``visualkeras.show(..., text_callable='name')``.
    :param text_vspacing: The vertical spacing between lines of text which are drawn as a result of the text_callable.
    :param spacing: Spacing in pixel between two layers
    :param draw_funnel: If set to True, a funnel will be drawn between consecutive layers
    :param shade_step: Deviation in lightness for drawing shades (only in volumetric view)
    :param legend: Add a legend of the layers to the image
    :param legend_text_spacing_offset: Offset the amount of space allocated for legend text. Useful when legend text is being cut off
    :param font: Font that will be used for the legend. Leaving this set to None, will use the default font.
    :param font_color: Color for the font if used. Can be str or (R,G,B,A).
    :param show_dimension: If legend is set to True and this is set to True, the dimensions of the layers will be shown in the legend.    
    :param sizing_mode: Strategy for handling layer dimensions. Options are:
        1) 'accurate': Use actual dimensions with scaling (default, may create very large visualizations);
        2) 'balanced': Smart scaling that balances accuracy with visual clarity (recommended for modern models);
        3) 'capped': Cap dimensions at specified limits while preserving ratios;
        4) 'logarithmic': Use logarithmic scaling for very large dimensions;
        5) 'relative': True proportional scaling where visual_size = dimension × relative_base_size.
                      Each layer's visual size is directly proportional to its dimension count.
    :param dimension_caps: Custom dimension limits when using 'capped' mode. Dict with keys:
        1) 'channels': Maximum size for channel dimensions (default: max_z);
        2) 'sequence': Maximum size for sequence/spatial dimensions (default: max_xy);
        3) 'general': Maximum size for other dimensions (default: max(max_z, max_xy));
    :param relative_base_size: Base size in pixels for 'relative' sizing mode. 
        Represents the visual size (in pixels) that a dimension of size 1 would have.
        All dimensions scale proportionally: visual_size = dimension × relative_base_size.
        For example, if relative_base_size=5:
        - A layer with 64 units gets visual size 64×5=320 pixels
        - A layer with 32 units gets visual size 32×5=160 pixels (exactly half)  
        - A layer with 16 units gets visual size 16×5=80 pixels (exactly half of 32)
        This maintains true proportional relationships between all layers (default: 20).
    :param options: Optional configuration bundle (``LayeredOptions`` or mapping)
        providing defaults for the renderer. Values passed directly to
        ``layered_view`` override the bundle.
    :param preset: Name of an entry in ``visualkeras.LAYERED_PRESETS`` to use as a
        starting point. Combine with ``options`` or explicit keyword arguments for
        tweaks.


    :return: Generated architecture image.
    """
    using_presets = options is not None or preset is not None

    if not using_presets:
        defaults = LayeredOptions().to_kwargs()
        defaults.update({
            "to_file": None,
            "type_ignore": None,
            "index_ignore": None,
            "color_map": None,
            "one_dim_orientation": 'z',
            "index_2D": [],
            "background_fill": 'white',
            "draw_volume": True,
            "draw_reversed": False,
            "padding": 10,
            "text_callable": None,
            "text_vspacing": 4,
            "spacing": 10,
            "draw_funnel": True,
            "shade_step": 10,
            "legend": False,
            "legend_text_spacing_offset": 15,
            "font": None,
            "font_color": 'black',
            "show_dimension": False,
            "sizing_mode": 'accurate',
            "dimension_caps": None,
            "relative_base_size": 20,
            "connector_fill": "gray",
            "connector_width": 1,
            "image_fit": "fill",
            "image_axis": "z",
            "layered_groups": None,
            "styles": None,
        })

        current_params = {
            "to_file": to_file,
            "min_z": min_z,
            "min_xy": min_xy,
            "max_z": max_z,
            "max_xy": max_xy,
            "scale_z": scale_z,
            "scale_xy": scale_xy,
            "type_ignore": type_ignore,
            "index_ignore": index_ignore,
            "color_map": color_map,
            "one_dim_orientation": one_dim_orientation,
            "index_2D": index_2D,
            "background_fill": background_fill,
            "draw_volume": draw_volume,
            "draw_reversed": draw_reversed,
            "padding": padding,
            "text_callable": text_callable,
            "text_vspacing": text_vspacing,
            "spacing": spacing,
            "draw_funnel": draw_funnel,
            "shade_step": shade_step,
            "legend": legend,
            "legend_text_spacing_offset": legend_text_spacing_offset,
            "font": font,
            "font_color": font_color,
            "show_dimension": show_dimension,
            "sizing_mode": sizing_mode,
            "dimension_caps": dimension_caps,
            "relative_base_size": relative_base_size,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "image_fit": image_fit,
            "image_axis": image_axis,
            "layered_groups": layered_groups,
            "styles": styles,
        }

        custom_keys = [
            key for key, value in current_params.items()
            if key in defaults and value != defaults[key]
        ]

        if len(custom_keys) >= 5:
            warnings.warn(
                "layered_view received many custom keyword arguments. "
                "Consider using visualkeras.show(..., preset=...) for a simpler workflow.",
                UserWarning,
                stacklevel=2,
            )

    if preset is not None or options is not None:
        defaults = LayeredOptions().to_kwargs()
        defaults["type_ignore"] = None
        defaults["index_ignore"] = None
        defaults["color_map"] = None
        defaults["text_callable"] = None
        defaults["dimension_caps"] = None
        defaults["font"] = None
        defaults["index_2D"] = []
        defaults["layered_groups"] = None
        defaults["styles"] = None

        resolved = dict(defaults)

        if preset is not None:
            try:
                resolved.update(LAYERED_PRESETS[preset].to_kwargs())
            except KeyError as exc:
                available = ", ".join(sorted(LAYERED_PRESETS.keys()))
                raise ValueError(
                    f"Unknown layered preset '{preset}'. Available presets: {available}"
                ) from exc

        if options is not None:
            if isinstance(options, LayeredOptions):
                option_values = options.to_kwargs()
            elif isinstance(options, Mapping):
                option_values = dict(options)
            else:
                raise TypeError(
                    "options must be a LayeredOptions instance or a mapping of keyword arguments."
                )
            resolved.update(option_values)

        explicit_values = {
            "to_file": to_file,
            "min_z": min_z,
            "min_xy": min_xy,
            "max_z": max_z,
            "max_xy": max_xy,
            "scale_z": scale_z,
            "scale_xy": scale_xy,
            "type_ignore": type_ignore,
            "index_ignore": index_ignore,
            "color_map": color_map,
            "one_dim_orientation": one_dim_orientation,
            "index_2D": index_2D,
            "background_fill": background_fill,
            "draw_volume": draw_volume,
            "draw_reversed": draw_reversed,
            "padding": padding,
            "text_callable": text_callable,
            "text_vspacing": text_vspacing,
            "spacing": spacing,
            "draw_funnel": draw_funnel,
            "shade_step": shade_step,
            "legend": legend,
            "legend_text_spacing_offset": legend_text_spacing_offset,
            "font": font,
            "font_color": font_color,
            "show_dimension": show_dimension,
            "sizing_mode": sizing_mode,
            "dimension_caps": dimension_caps,
            "relative_base_size": relative_base_size,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "layered_groups": layered_groups,
            "styles": styles,
        }

        for key, value in explicit_values.items():
            if key not in defaults:
                continue
            if value != defaults[key]:
                resolved[key] = value
        
        to_file = resolved["to_file"]
        min_z = resolved["min_z"]
        min_xy = resolved["min_xy"]
        max_z = resolved["max_z"]
        max_xy = resolved["max_xy"]
        scale_z = resolved["scale_z"]
        scale_xy = resolved["scale_xy"]
        type_ignore = resolved["type_ignore"]
        index_ignore = resolved["index_ignore"]
        color_map = resolved["color_map"]
        one_dim_orientation = resolved["one_dim_orientation"]
        index_2D = resolved["index_2D"]
        background_fill = resolved["background_fill"]
        draw_volume = resolved["draw_volume"]
        draw_reversed = resolved["draw_reversed"]
        padding = resolved["padding"]
        text_callable = resolved["text_callable"]
        text_vspacing = resolved["text_vspacing"]
        spacing = resolved["spacing"]
        draw_funnel = resolved["draw_funnel"]
        shade_step = resolved["shade_step"]
        legend = resolved["legend"]
        legend_text_spacing_offset = resolved["legend_text_spacing_offset"]
        font = resolved["font"]
        font_color = resolved["font_color"]
        show_dimension = resolved["show_dimension"]
        sizing_mode = resolved["sizing_mode"]
        dimension_caps = resolved["dimension_caps"]
        relative_base_size = resolved["relative_base_size"]
        connector_fill = resolved["connector_fill"]
        connector_width = resolved["connector_width"]
        image_fit = resolved["image_fit"]
        image_axis = resolved["image_axis"]
        layered_groups = resolved["layered_groups"]
        styles = resolved["styles"]

    if styles is not None and not isinstance(styles, dict):
        styles = dict(styles)

    if styles is None:
        styles = {}

    global_defaults = {
        'fill': None, 
        'outline': 'black',
        'padding': padding,
        'spacing': spacing,
        'scale_z': scale_z,
        'scale_xy': scale_xy,
        'min_z': min_z,
        'max_z': max_z,
        'min_xy': min_xy,
        'max_xy': max_xy,
        'shade_step': shade_step,
        'font_color': font_color,
        'image_fit': image_fit,
        'image_axis': image_axis
    }

    if type_ignore is not None and not isinstance(type_ignore, list):
        type_ignore = list(type_ignore)
    if index_ignore is not None and not isinstance(index_ignore, list):
        index_ignore = list(index_ignore)
    if index_2D is None:
        index_2D = []
    elif not isinstance(index_2D, list):
        index_2D = list(index_2D)
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
                f"Unknown text_callable preset '{text_callable}'. "
                f"Available presets: {available}"
            ) from exc

    if callable(text_callable) and text_callable not in _BUILT_IN_TEXT_CALLABLES:
        warnings.warn(
            "Custom text_callable detected. Built-in caption templates are available "
            "via visualkeras.show(..., text_callable='name').",
            UserWarning,
            stacklevel=2,
        )

    # Deprecation warning for legend_text_spacing_offset
    if legend_text_spacing_offset != 0:
        warnings.warn("The legend_text_spacing_offset parameter is deprecated and will be removed in a future release.")

    boxes = list()
    layer_y = list()
    color_wheel = ColorWheel()
    current_z = padding
    x_off = -1

    layer_types = list()
    dimension_list = []

    img_height = 0
    max_right = 0

    if type_ignore is None:
        type_ignore = list()

    if index_ignore is None:
        index_ignore = list()

    if color_map is None:
        color_map = dict()

    # Pre-process groups to map layers to their groups
    layer_to_groups = {}
    if layered_groups:
        # Create a map of layer name to index for faster lookup
        name_to_index = {}
        for i, layer in enumerate(model.layers):
            name = getattr(layer, 'name', None)
            if name:
                name_to_index[name] = i

        for group in layered_groups:
            for ref in group.get("layers", []):
                idx = -1
                if isinstance(ref, str):
                    idx = name_to_index.get(ref, -1)
                else:
                    # Assume it's a layer object
                    try:
                        idx = model.layers.index(ref)
                    except ValueError:
                        pass
                
                if idx != -1:
                    layer_to_groups.setdefault(idx, []).append(group)

    last_rendered_index = -1

    for index, layer in enumerate(model.layers):

        # Ignore layers that the use has opted out to
        if type(layer) in type_ignore or index in index_ignore:
            continue

        # Do not render the SpacingDummyLayer, just increase the pointer
        if type(layer) == SpacingDummyLayer:
            current_z += layer.spacing
            continue
        
        # Adjust spacing to prevent group overlap
        if last_rendered_index != -1 and layered_groups:
             prev_groups = layer_to_groups.get(last_rendered_index, [])
             curr_groups = layer_to_groups.get(index, [])
             
             exiting = [g for g in prev_groups if g not in curr_groups]
             entering = [g for g in curr_groups if g not in prev_groups]
             
             clearance = max([g.get('padding', 10) for g in exiting] + [0]) + \
                         max([g.get('padding', 10) for g in entering] + [0])
             
             current_z += clearance

        layer_type = type(layer)

        if legend and show_dimension:
            layer_types.append(layer_type)
        elif layer_type not in layer_types:
            layer_types.append(layer_type)

        # Resolve Layer Name
        try:
            layer_name = getattr(layer, 'name', None) or f'{layer.__class__.__name__}_{index}'
        except AttributeError:
            layer_name = f'unknown_{index}'

        # Resolve Styles
        # Merge legacy color_map into the defaults for backward compatibility.
        legacy_color = color_map.get(type(layer), {})
        current_defaults = global_defaults.copy()
        current_defaults.update(legacy_color)
        
        style = resolve_style(layer, layer_name, styles, current_defaults)

        # Get the primary shape of the layer's output
        raw_shape = _resolve_layer_output_shape(layer)
        shape = extract_primary_shape(raw_shape, layer_name)

        # Use Styles for Dimensions
        # We pass the specific constraints and scalers from the style instead of the global args.
        x, y, z = calculate_layer_dimensions(
            shape, 
            style['scale_z'], 
            style['scale_xy'], 
            style['max_z'], 
            style['max_xy'], 
            style['min_z'], 
            style['min_xy'],
            one_dim_orientation, sizing_mode,
            dimension_caps, relative_base_size
        )

        # --- Image Handling ---
        image_path = style.get("image")
        node_image = None
        
        if image_path:
            try:
                node_image = Image.open(image_path).convert("RGBA")
                fit_mode = style.get("image_fit", image_fit)
                axis = style.get("image_axis", image_axis)
                
                if fit_mode == "match_aspect":
                    img_w, img_h = node_image.size
                    img_ratio = img_w / img_h
                    
                    if axis == 'z': # Front (Width x Height) -> (z x y)
                        surf_ratio = z / y if y > 0 else 1
                        if img_ratio > surf_ratio:
                            z = int(y * img_ratio)
                        else:
                            y = int(z / img_ratio)
                    elif axis == 'y': # Top (Width x Depth) -> (z x de)
                        # de = x / 3. We adjust x to achieve target de.
                        # Ratio = Width / Depth = z / de
                        # de = z / Ratio
                        if img_ratio > 0:
                            de_target = int(z / img_ratio)
                            x = de_target * 3
                    elif axis == 'x': # Side (Depth x Height) -> (de x y)
                        # Ratio = Depth / Height = de / y
                        # de = y * Ratio
                        de_target = int(y * img_ratio)
                        x = de_target * 3
                
                # Apply scale_image
                scale_factor = style.get("scale_image")
                if scale_factor is not None:
                    try:
                        scale_factor = float(scale_factor)
                        if scale_factor < 0: scale_factor = 0.0
                    except (ValueError, TypeError):
                        scale_factor = 1.0
                    
                    if axis == 'z':
                        z = int(z * scale_factor)
                        y = int(y * scale_factor)
                    elif axis == 'y':
                        z = int(z * scale_factor)
                        x = int(x * scale_factor)
                    elif axis == 'x':
                        x = int(x * scale_factor)
                        y = int(y * scale_factor)

            except Exception as e:
                warnings.warn(f"Failed to load image for layer '{layer_name}': {e}")
                image_path = None
                node_image = None

        if legend and show_dimension:
            dimension_string = str(shape)
            dimension_string = dimension_string[1:len(dimension_string)-1].split(", ")
            dimension = []
            for i in range(0, len(dimension_string)):
                if dimension_string[i].isnumeric():
                    dimension.append(dimension_string[i])
            dimension_list.append(dimension)

        box = Box()
        box.layer = layer # Store layer for grouping
        box.style = style  # Store style for later use
        box.image = node_image
        if node_image:
            box.image_fit = style.get("image_fit", image_fit)
            box.image_axis = style.get("image_axis", image_axis)

        # Use styles for visual properties
        # If fill is None (default), fallback to the color wheel
        if style.get('fill') is None:
            box.fill = color_wheel.get_color(layer_type)
        else:
            box.fill = style.get('fill')
        
        box.outline = style.get('outline', 'black')
        box.shade = style.get('shade_step', shade_step)
        
        # Update the color_map so the legend reflects this layer's appearance
        color_map[layer_type] = {'fill': box.fill, 'outline': box.outline}

        box.de = 0
        if draw_volume and index not in index_2D:
            box.de = x / 3

        if x_off == -1:
            x_off = box.de / 2

        # top left coordinate
        box.x1 = current_z - box.de / 2
        box.y1 = box.de

        # bottom right coordinate
        box.x2 = box.x1 + z
        box.y2 = box.y1 + y

        boxes.append(box)
        layer_y.append(box.y2 - (box.y1 - box.de))

        # Update image bounds
        hh = box.y2 - (box.y1 - box.de)
        if hh > img_height:
            img_height = hh

        if box.x2 + box.de > max_right:
            max_right = box.x2 + box.de

        # Use style-based spacing
        layer_spacing = style.get('spacing', spacing)
        current_z += z + layer_spacing
        
        last_rendered_index = index

    # Generate image
    min_scene_x = float('inf')
    max_scene_x = float('-inf')
    max_top_extent = 0
    max_bottom_extent = 0

    for i, box in enumerate(boxes):
        h = layer_y[i]
        half_h = h / 2

        max_top_extent = max(max_top_extent, half_h)
        max_bottom_extent = max(max_bottom_extent, half_h)

        visual_x1 = box.x1 + x_off
        visual_x2 = box.x2 + x_off
        if draw_reversed:
            visual_x1 += box.de
            visual_x2 += box.de

        min_scene_x = min(min_scene_x, visual_x1)
        max_scene_x = max(max_scene_x, visual_x2)

    if text_callable is not None:
        if font is None:
            font = ImageFont.load_default()

        box_idx = -1
        for index, layer in enumerate(model.layers):
            if type(layer) in type_ignore or type(layer) == SpacingDummyLayer or index in index_ignore:
                continue
            box_idx += 1

            box = boxes[box_idx]
            local_font = box.style.get('font', font)
            local_vspacing = box.style.get('text_vspacing', text_vspacing)

            text, above = text_callable(box_idx, layer)

            text_w = 0
            text_h = 0
            lines = text.split('\n')
            for line in lines:
                if hasattr(local_font, 'getsize'):
                    line_w, line_h = local_font.getsize(line)
                else:
                    bbox = local_font.getbbox(line)
                    line_w = bbox[2]
                    line_h = bbox[3]
                text_w = max(text_w, line_w)
                text_h += line_h

            text_h += (len(lines) - 1) * local_vspacing

            width = box.x2 - box.x1
            base_x = box.x1 + x_off
            if draw_reversed:
                base_x += box.de

            if above:
                center_x = base_x + box.de + width / 2
                max_top_extent = max(max_top_extent, (layer_y[box_idx] / 2) + text_h)
            else:
                center_x = base_x + width / 2
                max_bottom_extent = max(max_bottom_extent, (layer_y[box_idx] / 2) + text_h)

            t_x1 = center_x - text_w / 2
            t_x2 = center_x + text_w / 2

            min_scene_x = min(min_scene_x, t_x1)
            max_scene_x = max(max_scene_x, t_x2)

    if layered_groups:
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        for group in layered_groups:
            group_boxes = _get_group_boxes(boxes, group)
            if not group_boxes: continue
            
            g_min_x = float('inf')
            g_max_x = float('-inf')
            g_min_y = float('inf') # relative to center (negative is up)
            g_max_y = float('-inf')
            
            for box in group_boxes:
                # Reconstruct visual bounds in scene space
                idx = boxes.index(box)
                h = layer_y[idx]
                
                # Y bounds (relative to center)
                # Top is -h/2, Bottom is h/2
                g_min_y = min(g_min_y, -h/2)
                g_max_y = max(g_max_y, h/2)
                
                # X bounds
                visual_x1 = box.x1 + x_off
                visual_x2 = box.x2 + x_off
                
                if draw_reversed:
                    visual_x1 += box.de
                    visual_x2 += box.de
                    
                    # Back face extends to left/up
                    g_min_x = min(g_min_x, visual_x1 - box.de)
                    g_max_x = max(g_max_x, visual_x2)
                else:
                    # Normal mode
                    # Back face extends to right/up
                    g_min_x = min(g_min_x, visual_x1)
                    g_max_x = max(g_max_x, visual_x2 + box.de)
            
            # Apply padding
            padding_val = group.get("padding", 10)
            g_min_x -= padding_val
            g_max_x += padding_val
            g_min_y -= padding_val
            g_max_y += padding_val
            
            # Update scene extents
            min_scene_x = min(min_scene_x, g_min_x)
            max_scene_x = max(max_scene_x, g_max_x)
            max_top_extent = max(max_top_extent, -g_min_y) 
            max_bottom_extent = max(max_bottom_extent, g_max_y)
            
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
                
                min_scene_x = min(min_scene_x, text_x1)
                max_scene_x = max(max_scene_x, text_x2)
                max_bottom_extent = max(max_bottom_extent, text_bottom)

    total_content_height = max_top_extent + max_bottom_extent
    img_height = total_content_height
    center_y_pos = max_top_extent

    x_shift = padding - min_scene_x
    img_width = max_scene_x + x_shift + padding

    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)

    for i, node in enumerate(boxes):
        h = layer_y[i]
        node_top = center_y_pos - h / 2

        node.y1 = node_top + node.de
        node.y2 = node_top + h

        node.x1 += x_shift + x_off
        node.x2 += x_shift + x_off

    draw = aggdraw.Draw(img)

    # Correct x positions of reversed boxes
    if draw_reversed:
        for box in boxes:
            offset = box.de
            # offset = 0
            box.x1 = box.x1 + offset
            box.x2 = box.x2 + offset

    if layered_groups:
        _draw_layered_group_boxes(draw, boxes, layered_groups, draw_reversed)

    # Draw created boxes

    last_box = None

    if draw_reversed:
        for box in boxes:
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                # Top connection back
                draw.line([last_box.x2 - last_box.de, last_box.y1 - last_box.de,
                           box.x1 - box.de, box.y1 - box.de], pen)
                # Bottom connection back
                draw.line([last_box.x2 - last_box.de, last_box.y2 - last_box.de,
                           box.x1 - box.de, box.y2 - box.de], pen)

            last_box = box

        last_box = None

        for box in reversed(boxes):
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                # Top connection front
                draw.line([last_box.x1, last_box.y1,
                           box.x2, box.y1], pen)

                # Bottom connection front
                draw.line([last_box.x1, last_box.y2,
                           box.x2, box.y2], pen)

            box.draw(draw, draw_reversed=True)

            if getattr(box, 'image', None):
                draw.flush()
                image = box.image
                fit = box.image_fit
                axis = box.image_axis
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
                de = box.de
                
                if axis == 'z': # Front
                    w = x2 - x1
                    h = y2 - y1
                    resized = resize_image_to_fit(image, int(w), int(h), fit)
                    img.paste(resized, (int(x1), int(y1)), resized)
                    
                elif axis == 'y': # Top
                    # Reversed Top Face: TL(x1-de, y1-de), TR(x2-de, y1-de), BR(x2, y1), BL(x1, y1)
                    p1 = (x1 - de, y1 - de)
                    p2 = (x2 - de, y1 - de)
                    p3 = (x2, y1)
                    p4 = (x1, y1)
                    apply_affine_transform(img, image, [p1, p2, p3, p4], fit)
                    
                elif axis == 'x': # Side
                    # Reversed Side Face (Left): TL(x1-de, y1-de), TR(x1, y1), BR(x1, y2), BL(x1-de, y2-de)
                    p1 = (x1 - de, y1 - de)
                    p2 = (x1, y1)
                    p3 = (x1, y2)
                    p4 = (x1 - de, y2 - de)
                    apply_affine_transform(img, image, [p1, p2, p3, p4], fit)
                
                draw = aggdraw.Draw(img)

            last_box = box

    else:
        for box in boxes:
            pen = aggdraw.Pen(get_rgba_tuple(box.outline))

            if last_box is not None and draw_funnel:
                draw.line([last_box.x2 + last_box.de, last_box.y1 - last_box.de,
                           box.x1 + box.de, box.y1 - box.de], pen)
                draw.line([last_box.x2 + last_box.de, last_box.y2 - last_box.de,
                           box.x1 + box.de, box.y2 - box.de], pen)

                draw.line([last_box.x2, last_box.y2,
                           box.x1, box.y2], pen)

                draw.line([last_box.x2, last_box.y1,
                           box.x1, box.y1], pen)

            box.draw(draw, draw_reversed=False)

            if getattr(box, 'image', None):
                draw.flush()
                image = box.image
                fit = box.image_fit
                axis = box.image_axis
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
                de = box.de
                
                if axis == 'z': # Front
                    w = x2 - x1
                    h = y2 - y1
                    resized = resize_image_to_fit(image, int(w), int(h), fit)
                    img.paste(resized, (int(x1), int(y1)), resized)
                    
                elif axis == 'y': # Top
                    # Normal Top Face: TL(x1, y1), TR(x2, y1), BR(x2+de, y1-de), BL(x1+de, y1-de)
                    # Wait, Box.draw normal top:
                    # draw.polygon([self.x1, self.y1,
                    #               self.x1 + self.de, self.y1 - self.de,
                    #               self.x2 + self.de, self.y1 - self.de,
                    #               self.x2, self.y1
                    #               ], pen, brush_s1)
                    # Order: BL, TL, TR, BR (relative to face?)
                    # Let's map to TL, TR, BR, BL.
                    # TL: (x1+de, y1-de)
                    # TR: (x2+de, y1-de)
                    # BR: (x2, y1)
                    # BL: (x1, y1)
                    
                    p1 = (x1 + de, y1 - de)
                    p2 = (x2 + de, y1 - de)
                    p3 = (x2, y1)
                    p4 = (x1, y1)
                    apply_affine_transform(img, image, [p1, p2, p3, p4], fit)
                    
                elif axis == 'x': # Side
                    # Normal Side Face (Right): TL(x2, y1), TR(x2+de, y1-de), BR(x2+de, y2-de), BL(x2, y2)
                    p1 = (x2, y1)
                    p2 = (x2 + de, y1 - de)
                    p3 = (x2 + de, y2 - de)
                    p4 = (x2, y2)
                    apply_affine_transform(img, image, [p1, p2, p3, p4], fit)
                
                draw = aggdraw.Draw(img)

            last_box = box

    draw.flush()

    if text_callable is not None:
        draw_text = ImageDraw.Draw(img)
        i = -1
        for index, layer in enumerate(model.layers):
            if type(layer) in type_ignore or type(layer) == SpacingDummyLayer or index in index_ignore:
                continue
            i += 1
            
            # Retrieve Styles
            box = boxes[i]
            local_font = box.style.get('font', font)
            local_font_color = box.style.get('font_color', font_color)
            local_vspacing = box.style.get('text_vspacing', text_vspacing)

            text, above = text_callable(i, layer)
            text_height = 0
            text_x_adjust = []
            for line in text.split('\n'):
                # Use local_font for measurements
                if hasattr(local_font, 'getsize'):
                    line_height = local_font.getsize(line)[1]
                    text_x_adjust.append(local_font.getsize(line)[0])
                else:
                    line_height = local_font.getbbox(line)[3]
                    text_x_adjust.append(local_font.getbbox(line)[2])
                
                text_height += line_height

            # Use local_vspacing
            text_height += (len(text.split('\n')) - 1) * local_vspacing

            text_x = box.x1 + (box.x2 - box.x1) / 2
            text_y = box.y2
            if above:
                text_x = box.x1 + box.de + (box.x2 - box.x1) / 2
                text_y = box.y1 - box.de - text_height
            
            # Use max width of the specific font
            text_x -= max(text_x_adjust or [0]) / 2 
            
            anchor = 'la'
            if above:
                anchor = 'la'
        
            draw_text.multiline_text(
                (text_x, text_y), 
                text, 
                font=local_font,
                fill=local_font_color,
                anchor=anchor, 
                align='center',
                spacing=local_vspacing
            )

    # Create layer color legend
    if legend:
        if font is None:
            font = ImageFont.load_default()

        if hasattr(font, 'getsize'):
            text_height = font.getsize("Ag")[1]
        else:
            text_height = font.getbbox("Ag")[3]
        cube_size = text_height

        de = 0
        if draw_volume:
            de = cube_size // 2

        patches = list()

        if show_dimension:
            counter = 0

        for layer_type in layer_types:
            if show_dimension:
                label = layer_type.__name__ + "(" + str(dimension_list[counter]) + ")"
                counter += 1
            else:
                label = layer_type.__name__

            
            if hasattr(font, 'getsize'):
                text_size = font.getsize(label)
            else:
                # Get last two values of the bounding box
                # getbbox returns 4 dimensions in total, where the first two are always zero, 
                # So we fetch the last two dimensions to match the behavior of getsize
                text_size = font.getbbox(label)[2:]
            label_patch_size = (2 * cube_size + de + spacing + text_size[0], cube_size + de)

            # this only works if cube_size is bigger than text height

            img_box = Image.new('RGBA', label_patch_size, background_fill)
            img_text = Image.new('RGBA', label_patch_size, (0, 0, 0, 0))
            draw_box = aggdraw.Draw(img_box)
            draw_text = ImageDraw.Draw(img_text)

            box = Box()
            box.x1 = cube_size
            box.x2 = box.x1 + cube_size
            box.y1 = de
            box.y2 = box.y1 + cube_size
            box.de = de
            box.shade = shade_step
            box.fill = color_map.get(layer_type, {}).get('fill', "#000000")
            box.outline = color_map.get(layer_type, {}).get('outline', "#000000")
            box.draw(draw_box, draw_reversed)

            text_x = box.x2 + box.de + spacing
            text_y = (label_patch_size[1] - text_height) / 2  # 2D center; use text_height and not the current label!
            draw_text.text((text_x, text_y), label, font=font, fill=font_color)

            draw_box.flush()
            img_box.paste(img_text, mask=img_text)
            patches.append(img_box)

        legend_image = linear_layout(patches, max_width=img.width, max_height=img.height, padding=padding,
                                     spacing=spacing,
                                     background_fill=background_fill, horizontal=True)
        img = vertical_image_concat(img, legend_image, background_fill=background_fill)

    if layered_groups:
        _draw_layered_group_captions(img, boxes, layered_groups, draw_reversed)

    if to_file is not None:
        img.save(to_file)

    return img

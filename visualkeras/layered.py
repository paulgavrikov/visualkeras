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
            class _LayerNamespace:
                class Layer:
                    pass

            layers = _LayerNamespace()

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


def _get_logo_boxes(boxes: List[Box], group: Dict[str, Any]) -> List[Box]:
    layers_ref = group.get("layers", [])
    if not layers_ref:
        return []
        
    target_boxes = []
    
    # Build lookup maps
    name_to_boxes = {}
    type_to_boxes = {}
    
    for box in boxes:
        if not hasattr(box, 'layer'): continue
        
        layer_name = getattr(box.layer, 'name', None)
        if layer_name:
            if layer_name not in name_to_boxes:
                name_to_boxes[layer_name] = []
            name_to_boxes[layer_name].append(box)
            
        layer_type = type(box.layer)
        if layer_type not in type_to_boxes:
            type_to_boxes[layer_type] = []
        type_to_boxes[layer_type].append(box)
        
    for ref in layers_ref:
        if isinstance(ref, str):
            if ref in name_to_boxes:
                target_boxes.extend(name_to_boxes[ref])
        elif isinstance(ref, type):
            if ref in type_to_boxes:
                target_boxes.extend(type_to_boxes[ref])
                
    return target_boxes


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
                 logo_groups: Optional[Sequence[Dict[str, Any]]] = None,
                 logos_legend: Union[bool, Dict[str, Any]] = False,
                 styles: Optional[Mapping[Union[str, type], Dict[str, Any]]] = None,
                 *,
                 options: Union[LayeredOptions, Mapping[str, Any], None] = None,
                 preset: Union[str, None] = None) -> Image:
    """Render a Keras model as a layered architecture diagram.

    This renderer is best suited to sequential or effectively linear models
    where layer order and tensor shape progression are the main story.

    Parameters
    ----------
    model : Any
        Keras model instance to visualize.

        Layered view works best when the model can be understood as a left to
        right sequence of transformations. It is usually the clearest choice
        for CNN style architectures and other models where tensor size changes
        are part of the explanation.
    to_file : str, optional
        Path to save the rendered image. The image format is inferred from the
        file extension.

        The rendered ``PIL.Image`` is returned whether or not this value is
        provided. Use this when you want the renderer to both write an image to
        disk and keep the in-memory result for further processing.
    min_z : int, default=20
        Minimum rendered depth in pixels for a layer box.

        This lower bound is applied after scaling. It prevents layers with very
        small channel counts from collapsing into thin slivers that are hard to
        see or compare.
    min_xy : int, default=20
        Minimum rendered width and height in pixels for a layer box.

        This is especially useful when a model mixes small tensors with much
        larger ones. A reasonable minimum keeps every layer visible without
        letting a few large layers define the entire layout.
    max_z : int, default=400
        Maximum rendered depth in pixels for a layer box.

        Use this to keep channel-heavy layers from dominating the visual depth
        of the diagram. It is most useful for deep convolutional stacks where
        late layers would otherwise become excessively thick.
    max_xy : int, default=2000
        Maximum rendered width and height in pixels for a layer box.

        This cap protects the layout from becoming impractically large when the
        model contains very large spatial dimensions or long sequences. It acts
        as a safety rail after scaling has been applied.
    scale_z : float, default=1.5
        Multiplier applied to the depth dimension before clamping.

        Increase this value when channel depth should read more strongly in the
        diagram. Reduce it when depth cues feel exaggerated or when channel rich
        layers overshadow the rest of the architecture.
    scale_xy : float, default=4
        Multiplier applied to the width and height dimensions before clamping.

        This is one of the main controls for the overall apparent size of the
        rendered layers. Lower values usually make crowded diagrams easier to
        fit, while higher values make individual layers easier to inspect.
    type_ignore : list, optional
        Sequence of layer classes to exclude from rendering.

        This is the simplest way to hide utility or low-information layers such
        as dropout, padding, or normalization layers without modifying the
        model itself. Every instance of a matching class is skipped.
    index_ignore : list, optional
        Sequence of layer indices to exclude from rendering.

        Use this when you want precise control over individual layers rather
        than entire layer types. Indices refer to positions in the model's
        layer list before rendering-time filtering is applied.
    color_map : dict, optional
        Mapping from layer class to style values such as ``fill`` and
        ``outline``.

        This provides broad styling by layer type and is the quickest way to
        create a consistent color language across the diagram. It is best suited
        to coarse styling rules, while ``styles`` is better for fine-grained
        per-layer overrides.
    one_dim_orientation : {'x', 'y', 'z'}, default='z'
        Axis used when rendering one dimensional layers such as dense or
        flattened outputs.

        Dense and flattened layers do not naturally have both width and height,
        so this setting controls how they are represented visually. Changing it
        can make mixed CNN and MLP models much easier to read.
    index_2D : list, optional
        Layer indices that should be forced into flat 2D rendering even when
        ``draw_volume`` is enabled.

        This is useful when most of the model benefits from 3D boxes but a few
        layers read better as flat blocks. Common cases include classifier heads
        and summary style terminal layers.
    background_fill : Any, default='white'
        Background color for the final image.

        This value accepts any Pillow-compatible color form, including named
        colors and RGBA tuples. Darker backgrounds often pair well with bright
        fills, while neutral backgrounds keep the focus on shape and layout.
    draw_volume : bool, default=True
        If ``True``, render boxes with 3D depth cues. If ``False``, render flat
        2D rectangles.

        The volumetric mode is usually the signature layered-view look. Flat
        mode is simpler and often preferable for documentation, compact figures,
        or models where depth would add noise rather than clarity.
    draw_reversed : bool, default=False
        Reverse the 3D viewing direction when ``draw_volume`` is enabled.

        This changes which faces of a 3D layer box are visible. It can be
        helpful for decoder style networks or for diagrams where the default
        perspective makes the flow feel visually backward.
    padding : int, default=10
        Outer padding around the full diagram in pixels.

        Increase this when legends, labels, or grouped overlays are too close
        to the image boundary. Padding affects the whole canvas rather than
        spacing between individual layers.
    text_callable : callable, optional
        Callable receiving ``(layer_index, layer)`` and returning
        ``(text, above)`` to annotate a layer. Built-in helpers are available in
        ``visualkeras.options.LAYERED_TEXT_CALLABLES``.

        This is the main hook for custom per-layer text. Use it when you want
        labels such as layer names, tensor shapes, block roles, or any other
        model-specific notes placed above or below each rendered box.
    text_vspacing : int, default=4
        Vertical spacing between lines produced by ``text_callable``.

        Increase this for multiline labels that feel cramped. Smaller values
        help conserve vertical space when you are already using generous
        padding or wide layer spacing.
    spacing : int, default=10
        Horizontal spacing between consecutive rendered layers.

        This is the main control for how tightly packed the diagram feels.
        Increasing it can make grouped stages and labels easier to follow, while
        smaller values produce more compact figures.
    draw_funnel : bool, default=True
        If ``True``, draw tapered transitions between consecutive layers.

        Funnels emphasize size changes between adjacent layers. They can be
        visually helpful in CNN diagrams, but turning them off produces a
        cleaner and more schematic look.
    shade_step : int, default=10
        Amount of shading variation used for 3D faces.

        Larger values create stronger contrast between faces and make the depth
        effect more pronounced. Smaller values produce a flatter and more subtle
        appearance.
    legend : bool, default=False
        If ``True``, add a legend describing rendered layer types and colors.

        A legend is useful when the diagram uses custom colors or contains many
        repeated layer classes. For small internal diagrams it may be unnecessary,
        but for external readers it often improves readability.
    legend_text_spacing_offset : int, default=15
        Extra width reserved for legend labels.

        Increase this when legend text is clipped or when long layer names need
        more room. This setting affects legend layout only.
    font : PIL.ImageFont.ImageFont, optional
        Font used for legend and annotation text. If omitted, the default PIL
        font is used.

        Use a custom font when you need the figure to match a publication or
        presentation style. The font choice can have a noticeable effect on the
        final layout, especially when legends or custom labels are enabled.
    font_color : Any, default='black'
        Text color used for legends and annotations.

        This should contrast clearly with ``background_fill`` and any other
        styling applied to the figure. In practice, it is often adjusted together
        with ``font`` and ``legend`` settings.
    show_dimension : bool, default=False
        If ``True`` and ``legend`` is enabled, include output dimensions in the
        legend entries.

        This adds shape information to the legend without requiring custom text
        on every layer. It is useful when you want to preserve a clean diagram
        while still exposing the underlying tensor sizes.
    sizing_mode : {'accurate', 'balanced', 'capped', 'logarithmic', 'relative'}, default='accurate'
        Strategy used to convert tensor dimensions into rendered sizes.

        ``accurate`` stays closest to the underlying tensor dimensions after
        scaling and clamping. ``balanced`` reduces extreme differences so the
        whole figure remains readable. ``capped`` respects ``dimension_caps`` to
        constrain large dimensions. ``logarithmic`` compresses very large ranges.
        ``relative`` scales layers directly from their dimensions using
        ``relative_base_size``.
    dimension_caps : dict, optional
        Custom caps used by ``capped`` mode. Supported keys are ``channels``,
        ``sequence``, and ``general``.

        This is useful when a small number of very large layers distort the
        overall layout. By capping specific dimension groups, you can keep the
        diagram readable while still preserving meaningful differences.
    relative_base_size : int, default=20
        Base pixel unit used by ``relative`` sizing mode.

        In ``relative`` mode, visual size is driven directly by the actual
        tensor dimensions. This value defines the pixel size associated with a
        dimension of one and therefore controls the overall scale of the figure.
    connector_fill : Any, default='gray'
        Color used for connector and transition elements.

        This should usually complement rather than compete with the layer fills.
        Neutral colors tend to work best when the boxes themselves already carry
        most of the semantic styling.
    connector_width : int, default=1
        Line width used for connector and transition elements.

        Increase this for presentation sized figures or when connectors are hard
        to distinguish at your chosen output resolution.
    image_fit : str, default='fill'
        Default fit mode for images injected through ``styles``. Individual
        style entries can override this.

        This controls how images are resized within the layer face they occupy.
        Choose a mode based on whether you prefer full coverage, preserved aspect
        ratio, or exact fill behavior.
    image_axis : {'x', 'y', 'z'}, default='z'
        Default axis used when rendering per-layer images in 3D mode.

        This matters when images are applied to volumetric boxes. It determines
        which face or orientation should be treated as the primary image plane
        unless a per-layer override is supplied through ``styles``.
    layered_groups : sequence of dict, optional
        Group definitions used to draw labeled background regions behind sets of
        layers.

        Groups are useful for separating architectural stages such as feature
        extraction, bottlenecks, and classifier heads. They add visual structure
        without changing the rendered layers themselves.
    logo_groups : sequence of dict, optional
        Logo placement definitions used to add icons or other overlays to
        selected layers.

        This is mainly intended for annotated presentation graphics and other
        highly styled diagrams. It is less common than ``layered_groups`` but can
        be useful when you want compact visual markers on specific layers.
    logos_legend : bool or dict, default=False
        If truthy, render a legend describing entries supplied through
        ``logo_groups``.

        A simple boolean enables the default legend behavior. A mapping allows
        more control over how the legend is rendered when logo overlays are part
        of the figure.
    styles : mapping, optional
        Fine-grained per-layer style overrides keyed by layer name or layer
        class.

        This is the most flexible styling mechanism in layered mode. Use it for
        per-layer images, detailed text and outline overrides, or any case where
        ``color_map`` is too coarse. See the layered API reference for supported
        style keys and examples.
    options : LayeredOptions or mapping, optional
        Configuration bundle applied after ``preset`` and before explicit keyword
        arguments.

        This is the preferred way to reuse a consistent layered style across
        multiple models. It also keeps larger examples and application code much
        easier to read than passing many keyword arguments inline.
    preset : str, optional
        Name of a preset from ``visualkeras.LAYERED_PRESETS``. Layered mode
        currently provides ``default``, ``compact``, and ``presentation``.

        Presets are useful starting points rather than strict modes. They can be
        combined with ``options`` and explicit overrides when you want the
        convenience of a predefined style without giving up control.

    Returns
    -------
    PIL.Image.Image
        Rendered layered diagram.

    Notes
    -----
    Configuration precedence is ``preset`` followed by ``options`` followed by
    explicit keyword arguments.

    Full documentation:
    https://visualkeras.readthedocs.io/en/latest/api/layered.html
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

    # Prepare logos
    box_logos = {}
    if logo_groups:
        for group in logo_groups:
            path = group.get("file")
            if not path: continue
            try:
                logo_img = Image.open(path)
            except:
                continue
            
            target_boxes = _get_logo_boxes(boxes, group)
            for box in target_boxes:
                if id(box) not in box_logos:
                    box_logos[id(box)] = []
                box_logos[id(box)].append((group, logo_img))

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

            if id(box) in box_logos:
                draw.flush()
                for group, logo_img in box_logos[id(box)]:
                    draw_node_logo(img, box, logo_img, group, draw_volume, draw_reversed=True)
                draw = aggdraw.Draw(img)

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

            if id(box) in box_logos:
                draw.flush()
                for group, logo_img in box_logos[id(box)]:
                    draw_node_logo(img, box, logo_img, group, draw_volume, draw_reversed=False)
                draw = aggdraw.Draw(img)

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

    if logos_legend:
        if font is None:
            font = ImageFont.load_default()
        img = draw_logos_legend(img, logo_groups, logos_legend, background_fill, font, font_color)

    if to_file is not None:
        img.save(to_file)

    return img

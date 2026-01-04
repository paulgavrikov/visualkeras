"""LeNet-style (feature-map stack) renderer.

This renderer aims to reproduce the classic "stack of feature maps" diagrams
commonly used to illustrate convolutional neural networks, while still being
able to represent dense (vector) layers and other non-convolutional components.

It is intentionally separate from ``functional_view``: it renders a mostly
sequential pipeline left-to-right (skipping ignored layers) and draws
connections based on the destination layer's kernel/pool parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import math
import aggdraw
from PIL import Image, ImageDraw, ImageFont

from .layer_utils import get_layers, extract_primary_shape
from .options import LenetOptions, LENET_PRESETS
from .utils import get_rgba_tuple, fade_color, resolve_style


# ---------------------------------------------------------------------------
# Shape helpers (copied/trimmed from layered.py for version-robustness)
# ---------------------------------------------------------------------------

def _shape_to_tuple(shape: Any) -> Any:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return shape
    if hasattr(shape, "as_list"):
        try:
            return tuple(shape.as_list())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(shape, list):
        return tuple(shape)
    return shape


def _resolve_layer_output_shape(layer: Any) -> Any:
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


def _clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def _as_tuple2(value: Any) -> Tuple[int, int]:
    if value is None:
        return (1, 1)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return (1, int(value[0]))
    return (1, 1)


def _clamp_rect_to_face(
    center_x: float,
    center_y: float,
    rect_w: float,
    rect_h: float,
    face_rect: Tuple[float, float, float, float],
    *,
    margin: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Clamp a rectangle (defined by center + size) to lie within a face rect."""
    fx1, fy1, fx2, fy2 = face_rect
    # available size inside margins
    avail_w = max(0.0, (fx2 - fx1) - 2.0 * margin)
    avail_h = max(0.0, (fy2 - fy1) - 2.0 * margin)
    rw = min(float(rect_w), avail_w) if avail_w > 0 else 0.0
    rh = min(float(rect_h), avail_h) if avail_h > 0 else 0.0
    half_w = rw / 2.0
    half_h = rh / 2.0
    if avail_w <= 0 or avail_h <= 0:
        # Face too small; collapse to center
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        return (cx, cy, cx, cy)
    cx = min(max(center_x, fx1 + margin + half_w), fx2 - margin - half_w)
    cy = min(max(center_y, fy1 + margin + half_h), fy2 - margin - half_h)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _with_alpha(rgba: Tuple[int, int, int, int], alpha: int) -> Tuple[int, int, int, int]:
    return (rgba[0], rgba[1], rgba[2], int(max(0, min(255, alpha))))


@dataclass
class RenderShape:
    kind: str  # "spatial" or "vector"
    h: int
    w: int
    c: int
    # Original-ish dims (best effort; used for kernel->patch ratio)
    h_dim: int
    w_dim: int
    c_dim: int


def _canonicalize_shape(layer: Any, shape: Any) -> RenderShape:
    """Return a canonical (H,W,C) representation + kind."""
    if shape is None:
        return RenderShape("vector", 1, 1, 1, 1, 1, 1)

    # Multi-output: best effort pick first output.
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]

    shape = _shape_to_tuple(shape)
    if not isinstance(shape, (list, tuple)):
        return RenderShape("vector", 1, 1, 1, 1, 1, 1)

    # Drop batch dim if present.
    dims = list(shape[1:]) if len(shape) > 1 else list(shape)
    rank = len(dims)

    data_format = getattr(layer, "data_format", None)

    def num(x: Any) -> int:
        try:
            return int(x) if x is not None else 1
        except Exception:  # noqa: BLE001
            return 1

    # 2D conv/pool typical: (H,W,C) or (C,H,W)
    if rank == 3:
        if data_format == "channels_first":
            c_dim, h_dim, w_dim = (num(dims[0]), num(dims[1]), num(dims[2]))
        else:
            h_dim, w_dim, c_dim = (num(dims[0]), num(dims[1]), num(dims[2]))
        return RenderShape("spatial", h_dim, w_dim, c_dim, h_dim, w_dim, c_dim)

    # 1D conv typical: (L,C) after batch removal
    if rank == 2:
        # Treat as vector for most cases, but Conv1D/Pooling1D looks nicer as 1×L×C.
        layer_name = type(layer).__name__.lower()
        if "conv1d" in layer_name or "pool1d" in layer_name:
            w_dim, c_dim = (num(dims[0]), num(dims[1]))
            return RenderShape("spatial", 1, w_dim, c_dim, 1, w_dim, c_dim)

        units = 1
        for d in dims:
            units *= num(d)
        return RenderShape("vector", 1, 1, units, 1, 1, units)

    # rank 1 or unknown: vector
    units = 1
    for d in dims:
        units *= num(d)
    return RenderShape("vector", 1, 1, units, 1, 1, units)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

class FeatureMapStack:
    """A stack of 2D feature maps drawn as offset rectangles (LeNet style)."""

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        channels: int,
        *,
        map_spacing: int,
        max_visual_channels: int,
        fill: Any,
        outline: Any,
        line_width: int = 1,
        shade_step: int = 6,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(max(1, channels))
        self.map_spacing = int(map_spacing)
        self.max_visual_channels = int(max(1, max_visual_channels))
        self.fill = fill
        self.outline = outline
        self.line_width = int(max(1, line_width))
        self.shade_step = int(max(0, shade_step))

    @property
    def visible_count(self) -> int:
        return min(self.channels, self.max_visual_channels)

    @property
    def offset(self) -> int:
        return (self.visible_count - 1) * self.map_spacing if self.visible_count > 0 else 0

    def bounds(self) -> Tuple[float, float, float, float]:
        left = self.x - self.offset
        top = self.y - self.offset
        right = self.x + self.width
        bottom = self.y + self.height
        return (left, top, right, bottom)

    def front_rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def front_anchor(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.front_rect()
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def left_mid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.front_rect()
        return (x1, (y1 + y2) / 2.0)

    def right_mid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.front_rect()
        return (x2, (y1 + y2) / 2.0)

    def draw(self, draw: aggdraw.Draw) -> None:
        pen = aggdraw.Pen(get_rgba_tuple(self.outline), self.line_width)
        base_fill = get_rgba_tuple(self.fill)

        # Draw from back to front.
        for i in range(self.visible_count - 1, -1, -1):
            ox = i * self.map_spacing
            oy = i * self.map_spacing
            x1 = self.x - ox
            y1 = self.y - oy
            x2 = x1 + self.width
            y2 = y1 + self.height

            if self.shade_step > 0:
                brush_color = fade_color(base_fill, (self.visible_count - 1 - i) * self.shade_step)
            else:
                brush_color = base_fill
            brush = aggdraw.Brush(brush_color)
            draw.rectangle([x1, y1, x2, y2], pen, brush)


class PyramidConnection:
    """Kernel/pool receptive-field style connector between two spatial stacks."""

    def __init__(
        self,
        src: FeatureMapStack,
        dst: FeatureMapStack,
        *,
        src_patch: Tuple[float, float, float, float],
        dst_patch: Tuple[float, float, float, float],
        connector_fill: Any,
        connector_width: int,
        patch_fill: Any,
        patch_outline: Any,
        draw_patches: bool = True,
        polygon_alpha: int = 90,
    ) -> None:
        self.src = src
        self.dst = dst
        self.src_patch = src_patch
        self.dst_patch = dst_patch
        self.connector_fill = connector_fill
        self.connector_width = int(max(1, connector_width))
        self.patch_fill = patch_fill
        self.patch_outline = patch_outline
        self.draw_patches = bool(draw_patches)
        self.polygon_alpha = int(max(0, min(255, polygon_alpha)))

    def draw(self, draw: aggdraw.Draw) -> None:
        pen = aggdraw.Pen(get_rgba_tuple(self.connector_fill), self.connector_width)

        # Wedge polygon between patch edges
        sx1, sy1, sx2, sy2 = self.src_patch
        dx1, dy1, dx2, dy2 = self.dst_patch
        poly = [sx2, sy1, dx1, dy1, dx1, dy2, sx2, sy2]

        brush = aggdraw.Brush(_with_alpha(get_rgba_tuple(self.connector_fill), self.polygon_alpha))
        draw.polygon(poly, pen, brush)

        # Edge lines (helps readability)
        draw.line([sx2, sy1, dx1, dy1], pen)
        draw.line([sx2, sy2, dx1, dy2], pen)

        if self.draw_patches:
            ppen = aggdraw.Pen(get_rgba_tuple(self.patch_outline), 1)
            pbrush = aggdraw.Brush(get_rgba_tuple(self.patch_fill))
            draw.rectangle(list(self.src_patch), ppen, pbrush)
            draw.rectangle(list(self.dst_patch), ppen, pbrush)


class FunnelConnection:
    """Connector from spatial stack to vector stack."""

    def __init__(
        self,
        src: FeatureMapStack,
        dst: FeatureMapStack,
        *,
        connector_fill: Any,
        connector_width: int,
        polygon_alpha: int = 70,
    ) -> None:
        self.src = src
        self.dst = dst
        self.connector_fill = connector_fill
        self.connector_width = int(max(1, connector_width))
        self.polygon_alpha = int(max(0, min(255, polygon_alpha)))

    def draw(self, draw: aggdraw.Draw) -> None:
        pen = aggdraw.Pen(get_rgba_tuple(self.connector_fill), self.connector_width)
        brush = aggdraw.Brush(_with_alpha(get_rgba_tuple(self.connector_fill), self.polygon_alpha))

        sx1, sy1, sx2, sy2 = self.src.front_rect()
        dx1, dy1, dx2, dy2 = self.dst.front_rect()

        poly = [sx2, sy1, dx1, dy1, dx1, dy2, sx2, sy2]
        draw.polygon(poly, pen, brush)
        draw.line([sx2, sy1, dx1, dy1], pen)
        draw.line([sx2, sy2, dx1, dy2], pen)


class FullConnection:
    """Simple connector between vector stacks (dense/flatten/etc)."""

    def __init__(
        self,
        src: FeatureMapStack,
        dst: FeatureMapStack,
        *,
        connector_fill: Any,
        connector_width: int,
        polygon_alpha: int = 55,
    ) -> None:
        self.src = src
        self.dst = dst
        self.connector_fill = connector_fill
        self.connector_width = int(max(1, connector_width))
        self.polygon_alpha = int(max(0, min(255, polygon_alpha)))

    def draw(self, draw: aggdraw.Draw) -> None:
        pen = aggdraw.Pen(get_rgba_tuple(self.connector_fill), self.connector_width)
        brush = aggdraw.Brush(_with_alpha(get_rgba_tuple(self.connector_fill), self.polygon_alpha))

        sx1, sy1, sx2, sy2 = self.src.front_rect()
        dx1, dy1, dx2, dy2 = self.dst.front_rect()

        # Narrower trapezoid looks cleaner for vectors.
        sy_top = sy1 + (sy2 - sy1) * 0.2
        sy_bot = sy2 - (sy2 - sy1) * 0.2
        dy_top = dy1 + (dy2 - dy1) * 0.2
        dy_bot = dy2 - (dy2 - dy1) * 0.2

        poly = [sx2, sy_top, dx1, dy_top, dx1, dy_bot, sx2, sy_bot]
        draw.polygon(poly, pen, brush)
        draw.line([sx2, sy_top, dx1, dy_top], pen)
        draw.line([sx2, sy_bot, dx1, dy_bot], pen)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Robust text sizing for newer and older Pillow versions."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    return draw.textsize(text, font=font)


def _get_multiline_text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    spacing: int = 2,
) -> Tuple[int, int]:
    """Measure multiline text (\n-delimited) in a Pillow-version-safe way."""
    if not text:
        return (0, 0)
    parts = str(text).splitlines()
    widths: list[int] = []
    heights: list[int] = []
    for part in parts:
        w, h = _get_text_size(draw, part, font)
        widths.append(int(w))
        heights.append(int(h))
    total_h = int(sum(heights) + spacing * max(0, len(parts) - 1))
    return (int(max(widths) if widths else 0), total_h)


def _default_top_label(layer: Any, rshape: RenderShape) -> str:
    return type(layer).__name__


def _default_bottom_label(layer: Any, rshape: RenderShape) -> str:
    if rshape.kind == "spatial":
        return f"{rshape.c_dim}@{rshape.h_dim}×{rshape.w_dim}"
    return f"{rshape.c_dim}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lenet_view(
    model: Any,
    to_file: Optional[str] = None,
    min_xy: int = 20,
    max_xy: int = 220,
    scale_xy: float = 4.0,
    type_ignore: Optional[Sequence[type]] = None,
    index_ignore: Optional[Sequence[int]] = None,
    color_map: Optional[Mapping[type, Mapping[str, Any]]] = None,
    background_fill: Any = "black",
    padding: int = 20,
    layer_spacing: int = 40,
    map_spacing: int = 4,
    max_visual_channels: int = 12,
    connector_fill: Any = "gray",
    connector_width: int = 1,
    patch_fill: Any = "#7db7ff",
    patch_outline: Any = "black",
    patch_scale: float = 1.0,
    draw_connections: bool = True,
    draw_patches: bool = True,
    font: Optional[ImageFont.ImageFont] = None,
    font_color: Any = "white",
    top_label_callable: Optional[Callable[[Any, RenderShape], Optional[str]]] = None,
    bottom_label_callable: Optional[Callable[[Any, RenderShape], Optional[str]]] = None,
    top_label: bool = True,
    bottom_label: bool = True,
    styles: Optional[Mapping[Union[str, type], Dict[str, Any]]] = None,
    *,
    options: Union[LenetOptions, Mapping[str, Any], None] = None,
    preset: Optional[str] = None,
) -> Image.Image:
    """Generate a LeNet-style visualization for a Keras/TensorFlow model."""
    # --- preset/options resolution (match layered/graph/functional behavior) ---
    if preset is not None or options is not None:
        defaults = LenetOptions().to_kwargs()
        defaults["color_map"] = None
        defaults["styles"] = None
        defaults["font"] = None
        defaults["top_label_callable"] = None  # not in LenetOptions but allow compare
        defaults["bottom_label_callable"] = None

        resolved = dict(defaults)

        if preset is not None:
            try:
                resolved.update(LENET_PRESETS[preset].to_kwargs())
            except KeyError as exc:
                available = ", ".join(sorted(LENET_PRESETS.keys()))
                raise ValueError(
                    f"Unknown lenet preset '{preset}'. Available presets: {available}"
                ) from exc

        if options is not None:
            if isinstance(options, LenetOptions):
                option_values = options.to_kwargs()
            elif isinstance(options, Mapping):
                option_values = dict(options)
            else:
                raise TypeError("options must be a LenetOptions instance or a mapping of keyword arguments.")
            resolved.update(option_values)

        explicit_values: Dict[str, Any] = {
            "to_file": to_file,
            "min_xy": min_xy,
            "max_xy": max_xy,
            "scale_xy": scale_xy,
            "type_ignore": type_ignore,
            "index_ignore": index_ignore,
            "color_map": color_map,
            "background_fill": background_fill,
            "padding": padding,
            "layer_spacing": layer_spacing,
            "map_spacing": map_spacing,
            "max_visual_channels": max_visual_channels,
            "connector_fill": connector_fill,
            "connector_width": connector_width,
            "patch_fill": patch_fill,
            "patch_outline": patch_outline,
            "patch_scale": patch_scale,
            "draw_connections": draw_connections,
            "draw_patches": draw_patches,
            "font": font,
            "font_color": font_color,
            "top_label": top_label,
            "bottom_label": bottom_label,
            "styles": styles,
        }
        for key, value in explicit_values.items():
            if key not in defaults:
                continue
            if value != defaults[key]:
                resolved[key] = value

        # write back
        to_file = resolved["to_file"]
        min_xy = resolved["min_xy"]
        max_xy = resolved["max_xy"]
        scale_xy = resolved["scale_xy"]
        type_ignore = resolved["type_ignore"]
        index_ignore = resolved["index_ignore"]
        color_map = resolved["color_map"]
        background_fill = resolved["background_fill"]
        padding = resolved["padding"]
        layer_spacing = resolved["layer_spacing"]
        map_spacing = resolved["map_spacing"]
        max_visual_channels = resolved["max_visual_channels"]
        connector_fill = resolved["connector_fill"]
        connector_width = resolved["connector_width"]
        patch_fill = resolved["patch_fill"]
        patch_outline = resolved["patch_outline"]
        patch_scale = resolved["patch_scale"]
        draw_connections = resolved["draw_connections"]
        draw_patches = resolved["draw_patches"]
        font = resolved["font"]
        font_color = resolved["font_color"]
        top_label = resolved["top_label"]
        bottom_label = resolved["bottom_label"]
        styles = resolved["styles"]

    if color_map is None:
        color_map = {}
    if styles is None:
        styles = {}

    if top_label_callable is None:
        top_label_callable = _default_top_label
    if bottom_label_callable is None:
        bottom_label_callable = _default_bottom_label

    type_ignore = set(type_ignore or [])
    index_ignore = set(index_ignore or [])

    layers = list(get_layers(model))

    # Build renderable layers list
    render_layers: list[Tuple[int, Any, RenderShape, Dict[str, Any]]] = []
    global_defaults = {
        "fill": "#d9d9d9",
        "outline": "black",
        "line_width": 1,
        "shade_step": 6,
        # allow per-layer overrides
        "map_spacing": map_spacing,
        "max_visual_channels": max_visual_channels,
        "connector_fill": connector_fill,
        "connector_width": connector_width,
        "patch_fill": patch_fill,
        "patch_outline": patch_outline,
        "patch_scale": patch_scale,
    }

    for idx, layer in enumerate(layers):
        if idx in index_ignore:
            continue
        if type(layer) in type_ignore:
            continue

        name = getattr(layer, "name", f"layer_{idx}")
        legacy_color = color_map.get(type(layer), {})
        current_defaults = dict(global_defaults)
        current_defaults.update(legacy_color)
        style = resolve_style(layer, name, styles, current_defaults)

        raw_shape = _resolve_layer_output_shape(layer)
        primary = extract_primary_shape(raw_shape, name)
        rshape = _canonicalize_shape(layer, primary)
        render_layers.append((idx, layer, rshape, style))

    if not render_layers:
        # empty canvas
        img = Image.new("RGBA", (max(1, padding * 2), max(1, padding * 2)), get_rgba_tuple(background_fill))
        if to_file:
            img.save(to_file)
        return img

    # First pass: compute stack sizes (front face)
    stacks: list[Dict[str, Any]] = []
    max_total_h = 0
    for idx, layer, rshape, style in render_layers:
        w_px = _clamp_int(rshape.w * float(scale_xy), min_xy, max_xy)
        h_px = _clamp_int(rshape.h * float(scale_xy), min_xy, max_xy)

        # vectors: keep a pleasant aspect; small square works best
        if rshape.kind == "vector":
            w_px = min(w_px, max_xy // 2) if max_xy > 0 else w_px
            h_px = min(h_px, max_xy // 2) if max_xy > 0 else h_px
            w_px = max(min_xy, min(w_px, max_xy))
            h_px = max(min_xy, min(h_px, max_xy))

        ms = int(style.get("map_spacing", map_spacing))
        mvc = int(style.get("max_visual_channels", max_visual_channels))
        temp_stack = FeatureMapStack(
            0, 0, w_px, h_px, rshape.c,
            map_spacing=ms,
            max_visual_channels=mvc,
            fill=style.get("fill", global_defaults["fill"]),
            outline=style.get("outline", global_defaults["outline"]),
            line_width=int(style.get("line_width", 1)),
            shade_step=int(style.get("shade_step", 6)),
        )
        _, top, _, bottom = temp_stack.bounds()
        total_h = bottom - top
        max_total_h = max(max_total_h, int(math.ceil(total_h)))
        stacks.append({
            "layer": layer,
            "layer_index": idx,
            "rshape": rshape,
            "style": style,
            "stack": temp_stack,
        })

    # Second pass: assign positions using bounding boxes
    cursor = float(padding)
    global_top = float(padding)

    for obj in stacks:
        stack: FeatureMapStack = obj["stack"]
        total_w = (stack.width + stack.offset)
        total_h = (stack.height + stack.offset)

        # Place so that the bounding-left starts at cursor
        stack.x = cursor + stack.offset
        # Center vertically: set bounding-top = global_top + (max_total_h - total_h)/2
        stack.y = global_top + (max_total_h - total_h) / 2.0 + stack.offset

        cursor += total_w + float(layer_spacing)

    # Connections are built after final shifting (so patches align to faces).
    connections: list[Any] = []

    # Determine image bounds (include optional caption text)
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # Resolve a font early so we can size/allocate space for captions.
    if (top_label or bottom_label) and font is None:
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None

    measure_draw = None
    if font is not None and (top_label or bottom_label):
        _dummy = Image.new('RGB', (10, 10))
        measure_draw = ImageDraw.Draw(_dummy)

    for obj in stacks:
        st: FeatureMapStack = obj['stack']
        l, t, r, b = st.bounds()
        min_x = min(min_x, l)
        min_y = min(min_y, t)
        max_x = max(max_x, r)
        max_y = max(max_y, b)

        if measure_draw is None:
            continue

        layer = obj['layer']
        rshape: RenderShape = obj['rshape']
        cx, _cy = st.front_anchor()
        x1, y1, x2, y2 = st.front_rect()

        if bottom_label:
            text = bottom_label_callable(layer, rshape)
            if text:
                tw, th = _get_multiline_text_size(measure_draw, str(text), font)
                tx0 = cx - tw / 2.0
                ty0 = y2 + 6
                tx1 = tx0 + tw
                ty1 = ty0 + th
                min_x = min(min_x, tx0)
                min_y = min(min_y, ty0)
                max_x = max(max_x, tx1)
                max_y = max(max_y, ty1)

        if top_label:
            text = top_label_callable(layer, rshape)
            if text:
                tw, th = _get_multiline_text_size(measure_draw, str(text), font)
                vis_top = y1 - st.offset
                tx0 = cx - tw / 2.0
                ty0 = vis_top - th - 6
                tx1 = tx0 + tw
                ty1 = ty0 + th
                min_x = min(min_x, tx0)
                min_y = min(min_y, ty0)
                max_x = max(max_x, tx1)
                max_y = max(max_y, ty1)

    # Pad (additional safety, but avoid negative)
    img_w = int(max(1, math.ceil(max_x - min_x + padding)))
    img_h = int(max(1, math.ceil(max_y - min_y + padding)))

    # Shift everything if needed so min coords are inside padding/2
    shift_x = float(padding) / 2.0 - min_x
    shift_y = float(padding) / 2.0 - min_y
    for obj in stacks:
        st: FeatureMapStack = obj['stack']
        st.x += shift_x
        st.y += shift_y

    # Build connections between consecutive rendered layers (after shift)
    connections = []
    if draw_connections and len(stacks) >= 2:
        for i in range(len(stacks) - 1):
            src_obj = stacks[i]
            dst_obj = stacks[i + 1]
            src_stack: FeatureMapStack = src_obj['stack']
            dst_stack: FeatureMapStack = dst_obj['stack']
            src_shape: RenderShape = src_obj['rshape']
            dst_shape: RenderShape = dst_obj['rshape']

            # Use destination op params (receptive field / pooling window).
            k = getattr(dst_obj['layer'], 'kernel_size', None)
            p = getattr(dst_obj['layer'], 'pool_size', None)
            kernel = _as_tuple2(k if k is not None else p)

            # Style from destination layer (more intuitive).
            dst_style = dst_obj['style']
            conn_fill = dst_style.get('connector_fill', connector_fill)
            conn_w = int(dst_style.get('connector_width', connector_width))
            pfill = dst_style.get('patch_fill', patch_fill)
            pout = dst_style.get('patch_outline', patch_outline)
            pscale = float(dst_style.get('patch_scale', patch_scale))

            if src_shape.kind == 'spatial' and dst_shape.kind == 'spatial' and (k is not None or p is not None):
                sx1, sy1, sx2, sy2 = src_stack.front_rect()
                dx1, dy1, dx2, dy2 = dst_stack.front_rect()

                # src patch size based on kernel/pool ratio
                src_w_dim = max(1, int(src_shape.w_dim))
                src_h_dim = max(1, int(src_shape.h_dim))
                kh, kw = kernel
                patch_w = max(4, int((src_stack.width * (kw / src_w_dim)) * pscale))
                patch_h = max(4, int((src_stack.height * (kh / src_h_dim)) * pscale))
                patch_w = min(patch_w, int(src_stack.width * 0.6))
                patch_h = min(patch_h, int(src_stack.height * 0.6))

                # Place patch near the right edge of the source *front face*, clamped inside.
                cx = sx2 - 1 - patch_w / 2.0
                cy = sy1 + src_stack.height * 0.5
                sp = _clamp_rect_to_face(cx, cy, patch_w, patch_h, (sx1, sy1, sx2, sy2), margin=1.0)

                # Destination activation patch: small, near the left edge of the dst front face.
                dsz = max(3, int(min(dst_stack.width, dst_stack.height) * 0.22))
                dcx = dx1 + 1 + dsz / 2.0
                dcy = dy1 + dst_stack.height * 0.5
                dp = _clamp_rect_to_face(dcx, dcy, dsz, dsz, (dx1, dy1, dx2, dy2), margin=1.0)

                connections.append(
                    PyramidConnection(
                        src_stack,
                        dst_stack,
                        src_patch=sp,
                        dst_patch=dp,
                        connector_fill=conn_fill,
                        connector_width=conn_w,
                        patch_fill=pfill,
                        patch_outline=pout,
                        draw_patches=draw_patches,
                    )
                )
            elif src_shape.kind == 'spatial' and dst_shape.kind == 'vector':
                connections.append(
                    FunnelConnection(src_stack, dst_stack, connector_fill=conn_fill, connector_width=conn_w)
                )
            else:
                connections.append(
                    FullConnection(src_stack, dst_stack, connector_fill=conn_fill, connector_width=conn_w)
                )

    # Create canvas and draw
    img = Image.new("RGBA", (img_w, img_h), get_rgba_tuple(background_fill))
    draw = aggdraw.Draw(img)

    for obj in stacks:
        obj["stack"].draw(draw)
    for conn in connections:
        conn.draw(draw)

    draw.flush()

    # Text pass
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None

    if font is not None and (top_label or bottom_label):
        dtext = ImageDraw.Draw(img)
        for obj in stacks:
            layer = obj["layer"]
            rshape = obj["rshape"]
            stack: FeatureMapStack = obj["stack"]
            x1, y1, x2, y2 = stack.front_rect()
            cx = (x1 + x2) / 2.0

            if bottom_label:
                text = bottom_label_callable(layer, rshape)
                if text:
                    tw, th = _get_multiline_text_size(dtext, str(text), font)
                    dtext.multiline_text((cx - tw / 2.0, y2 + 6), str(text), font=font, fill=font_color, spacing=2, align='center')

            if top_label:
                text = top_label_callable(layer, rshape)
                if text:
                    # Account for stack offset: top of visible stack is y1 - offset
                    vis_top = y1 - stack.offset
                    tw, th = _get_multiline_text_size(dtext, str(text), font)
                    dtext.multiline_text((cx - tw / 2.0, vis_top - th - 6), str(text), font=font, fill=font_color, spacing=2, align='center')

    if to_file:
        img.save(to_file)

    return img
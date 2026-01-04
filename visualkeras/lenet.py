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

import hashlib
import random
import os

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


def _stable_seed(base_seed: Optional[int], *parts: Any) -> int:
    """Create a stable 64-bit seed from an optional base seed and arbitrary parts."""
    h = hashlib.sha256()
    if base_seed is not None:
        h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Face image helpers (front-face textures)
# ---------------------------------------------------------------------------

_FACE_IMAGE_CACHE: Dict[str, Image.Image] = {}


def _load_face_image(spec: Any) -> Optional[Image.Image]:
    """Load a face image from a path or return a provided PIL image.

    Returns an RGBA image or None on failure.
    """
    if spec is None:
        return None
    if isinstance(spec, Image.Image):
        try:
            return spec.convert("RGBA")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(spec, str):
        path = os.path.expanduser(spec)
        # Cache the decoded image; callers get a copy to avoid mutation issues.
        if path in _FACE_IMAGE_CACHE:
            return _FACE_IMAGE_CACHE[path].copy()
        try:
            img = Image.open(path).convert("RGBA")
        except Exception:  # noqa: BLE001
            return None
        _FACE_IMAGE_CACHE[path] = img
        return img.copy()
    return None


def _parse_face_image_style(style: Mapping[str, Any]) -> Tuple[Optional[Any], str, int, Optional[int]]:
    """Extract face-image parameters from a style dict.

    Supports either:
      - face_image = "/path/to.png"
      - face_image = {"path": "...", "fit": "...", "alpha": 200, "inset": 1}
    and optional top-level overrides:
      - face_image_fit, face_image_alpha, face_image_inset
    """
    spec = style.get("face_image", None)
    fit = str(style.get("face_image_fit", "cover")).strip().lower()
    alpha = int(style.get("face_image_alpha", 255))
    inset = style.get("face_image_inset", None)
    if inset is not None:
        try:
            inset = int(inset)
        except Exception:  # noqa: BLE001
            inset = None

    if isinstance(spec, Mapping):
        # Nested spec dict for convenience
        spec_map: Mapping[str, Any] = spec
        spec_path = spec_map.get("path", spec_map.get("src", spec_map.get("file", None)))
        if spec_path is not None:
            spec = spec_path
        if "fit" in spec_map:
            fit = str(spec_map.get("fit")).strip().lower()
        if "alpha" in spec_map:
            try:
                alpha = int(spec_map.get("alpha"))
            except Exception:  # noqa: BLE001
                pass
        if "inset" in spec_map:
            try:
                inset = int(spec_map.get("inset"))
            except Exception:  # noqa: BLE001
                pass

    # Normalize + clamp
    if fit not in {"cover", "contain", "match_aspect", "fill"}:
        fit = "cover"
    alpha = int(max(0, min(255, alpha)))
    return spec, fit, alpha, inset


def _adjust_wh_for_image_aspect(
    w_px: int,
    h_px: int,
    img: Image.Image,
    *,
    min_xy: int,
    max_xy: int,
) -> Tuple[int, int]:
    """Adjust (w,h) so the face matches the image aspect ratio (best-effort)."""
    try:
        iw, ih = img.size
    except Exception:  # noqa: BLE001
        return (w_px, h_px)
    if iw <= 0 or ih <= 0:
        return (w_px, h_px)
    aspect = float(iw) / float(ih)

    def _cl(v: float) -> int:
        return int(max(min_xy, min(max_xy, round(v))))

    # Prefer keeping height stable (less vertical jitter) when possible.
    w1 = _cl(h_px * aspect)
    if min_xy <= w1 <= max_xy:
        return (w1, _cl(h_px))

    h1 = _cl(w_px / aspect)
    if min_xy <= h1 <= max_xy:
        return (_cl(w_px), h1)

    # Fallback: clamp both with minimal distortion.
    return (_cl(w1), _cl(h1))


def _fit_image_to_rect(
    img: Image.Image,
    w: int,
    h: int,
    *,
    fit: str,
    background: Tuple[int, int, int, int],
) -> Image.Image:
    """Fit an image into a (w,h) rect using the requested mode."""
    fit = (fit or "cover").strip().lower()
    if fit == "match_aspect":
        # After aspect-matching, 'contain' shows the full image with no crop.
        fit = "contain"

    if w <= 0 or h <= 0:
        return Image.new("RGBA", (max(1, w), max(1, h)), background)

    try:
        iw, ih = img.size
    except Exception:  # noqa: BLE001
        iw, ih = (0, 0)

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))

    if iw <= 0 or ih <= 0:
        return Image.new("RGBA", (w, h), background)

    if fit == "fill":
        return img.resize((w, h), resample=resample)

    if fit == "contain":
        scale = min(float(w) / float(iw), float(h) / float(ih))
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        im2 = img.resize((nw, nh), resample=resample)
        canvas = Image.new("RGBA", (w, h), background)
        canvas.paste(im2, ((w - nw) // 2, (h - nh) // 2), im2)
        return canvas

    # cover (default)
    scale = max(float(w) / float(iw), float(h) / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    im2 = img.resize((nw, nh), resample=resample)

    left = max(0, (nw - w) // 2)
    top = max(0, (nh - h) // 2)
    return im2.crop((left, top, left + w, top + h))


def _apply_face_images(img: Image.Image, stacks: Sequence[Dict[str, Any]]) -> None:
    """Paste configured face images onto the front face of stacks."""
    for obj in stacks:
        style = obj.get("style", {}) or {}
        st: FeatureMapStack = obj.get("stack")
        if st is None:
            continue

        spec, fit, alpha, inset = _parse_face_image_style(style)
        if not spec:
            continue

        face_img = style.get("_face_image_obj", None)
        if not isinstance(face_img, Image.Image):
            face_img = _load_face_image(spec)
            if face_img is None:
                continue
            # Save for later reuse in the same render call.
            try:
                style["_face_image_obj"] = face_img
            except Exception:  # noqa: BLE001
                pass

        x1f, y1f, x2f, y2f = st.front_rect()
        ix1 = int(round(x1f))
        iy1 = int(round(y1f))
        ix2 = int(round(x2f))
        iy2 = int(round(y2f))

        # Inset so we don't paint over the outline
        inset_px = st.line_width if inset is None else int(max(0, inset))
        ix1 += inset_px
        iy1 += inset_px
        ix2 -= inset_px
        iy2 -= inset_px

        w = max(1, ix2 - ix1)
        h = max(1, iy2 - iy1)

        bg = get_rgba_tuple(style.get("fill", st.fill))
        fitted = _fit_image_to_rect(face_img, w, h, fit=fit, background=bg)

        if alpha < 255:
            # Multiply alpha channel
            r, g, b, a = fitted.split()
            a = a.point(lambda v: int(v * (alpha / 255.0)))
            fitted = Image.merge("RGBA", (r, g, b, a))

        # Composite onto the base image
        try:
            img.alpha_composite(fitted, (ix1, iy1))
        except Exception:  # noqa: BLE001
            # Fallback
            img.paste(fitted, (ix1, iy1), fitted)

def _sample_patch_ratios(rng: random.Random, *, x_lo: float, x_hi: float, y_lo: float = 0.12, y_hi: float = 0.88) -> Tuple[float, float]:
    """Sample (rx, ry) in [0,1] normalized coordinates with bounded ranges."""
    rx = x_lo + (x_hi - x_lo) * rng.random()
    ry = y_lo + (y_hi - y_lo) * rng.random()
    # clamp just in case
    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    return rx, ry


def _place_patch_by_ratio(
    face: Tuple[float, float, float, float],
    patch_w: float,
    patch_h: float,
    rx: float,
    ry: float,
    *,
    margin: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Place a patch within a face using normalized ratios, clamped to the face."""
    fx1, fy1, fx2, fy2 = face
    fw = max(0.0, fx2 - fx1)
    fh = max(0.0, fy2 - fy1)

    # Available travel range for the patch's top-left (after margins)
    avail_w = max(0.0, fw - 2 * margin - patch_w)
    avail_h = max(0.0, fh - 2 * margin - patch_h)

    # If the patch is too large, fall back to center ratios.
    rx_eff = rx if avail_w > 0 else 0.5
    ry_eff = ry if avail_h > 0 else 0.5

    cx = fx1 + margin + patch_w / 2.0 + rx_eff * avail_w
    cy = fy1 + margin + patch_h / 2.0 + ry_eff * avail_h
    return _clamp_rect_to_face(cx, cy, patch_w, patch_h, face, margin=margin)


def _rects_overlap_1d(a1: float, a2: float, b1: float, b2: float, *, eps: float = 0.0) -> bool:
    """Return True if [a1,a2] overlaps/touches [b1,b2] within eps."""
    return (a1 <= b2 + eps) and (a2 >= b1 - eps)


def _clamp_rect_topleft(
    rect: Tuple[float, float, float, float],
    face: Tuple[float, float, float, float],
    *,
    margin: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Clamp rect (x1,y1,x2,y2) inside face, preserving its size."""
    x1, y1, x2, y2 = rect
    fx1, fy1, fx2, fy2 = face
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    # protect against negative available space
    max_x1 = fx2 - margin - w
    max_y1 = fy2 - margin - h
    x1 = min(max(x1, fx1 + margin), max_x1)
    y1 = min(max(y1, fy1 + margin), max_y1)
    return (x1, y1, x1 + w, y1 + h)


def _set_rect_x1(rect: Tuple[float, float, float, float], new_x1: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    return (new_x1, y1, new_x1 + w, y2)


def _set_rect_y1(rect: Tuple[float, float, float, float], new_y1: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    h = y2 - y1
    return (x1, new_y1, x2, new_y1 + h)


def _choose_y_from_ranges(
    rng: random.Random,
    prefer: float,
    ranges: Sequence[Tuple[float, float]],
) -> Optional[float]:
    if not ranges:
        return None
    candidates = []
    for lo, hi in ranges:
        y = max(lo, min(prefer, hi))
        candidates.append((abs(prefer - y), y, lo, hi))
    mind = min(c[0] for c in candidates)
    best = [c for c in candidates if abs(c[0] - mind) <= 1e-6]
    _, y, lo, hi = rng.choice(best)
    # tiny jitter for variety while staying close
    if hi - lo >= 1.0:
        jitter = (rng.random() - 0.5) * min(6.0, hi - lo)
        y = max(lo, min(hi, y + jitter))
    return y


def _enforce_in_out_patch_separation(
    face: Tuple[float, float, float, float],
    incoming: Tuple[float, float, float, float],
    outgoing: Tuple[float, float, float, float],
    *,
    rng: random.Random,
    x_eps: float = 1.0,
    v_gap: float = 2.0,
    margin: float = 1.0,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Ensure ordering + avoid 'touching' for incoming/outgoing patches on the same face.

    Requirements:
    - The outgoing patch (to the next layer) should not have a left edge further left than the incoming patch.
    - If patches are horizontally close (overlap/touch), enforce enough vertical separation so they don't touch.
    """
    inc = _clamp_rect_topleft(incoming, face, margin=margin)
    out = _clamp_rect_topleft(outgoing, face, margin=margin)

    # Enforce ordering by left edge: out.x1 >= inc.x1
    if out[0] < inc[0]:
        fx1, fy1, fx2, fy2 = face
        out_w = out[2] - out[0]
        max_out_x1 = fx2 - margin - out_w
        out = _set_rect_x1(out, min(max_out_x1, inc[0]))
        out = _clamp_rect_topleft(out, face, margin=margin)
        if out[0] < inc[0]:
            # If still violated due to clamping, move incoming left as needed.
            inc_w = inc[2] - inc[0]
            inc = _set_rect_x1(inc, out[0])
            inc = _clamp_rect_topleft(inc, face, margin=margin)

    # If horizontally close, enforce vertical separation (disallow overlap/touch)
    x_close = out[0] <= inc[2] + x_eps  # outgoing is on the right (or close)
    if x_close and _rects_overlap_1d(inc[1], inc[3], out[1], out[3], eps=v_gap):
        fx1, fy1, fx2, fy2 = face

        # First, try moving outgoing away from incoming (above or below).
        out_h = out[3] - out[1]
        min_y = fy1 + margin
        max_y = fy2 - margin - out_h
        above_max = min(max_y, inc[1] - v_gap - out_h)
        below_min = max(min_y, inc[3] + v_gap)
        ranges: list[Tuple[float, float]] = []
        if min_y <= above_max:
            ranges.append((min_y, above_max))
        if below_min <= max_y:
            ranges.append((below_min, max_y))

        new_y = _choose_y_from_ranges(rng, out[1], ranges)
        if new_y is not None:
            out = _set_rect_y1(out, new_y)
            out = _clamp_rect_topleft(out, face, margin=margin)

        # If still overlapping (very tight face), try moving incoming instead.
        if _rects_overlap_1d(inc[1], inc[3], out[1], out[3], eps=v_gap):
            inc_h = inc[3] - inc[1]
            min_y2 = fy1 + margin
            max_y2 = fy2 - margin - inc_h
            above_max2 = min(max_y2, out[1] - v_gap - inc_h)
            below_min2 = max(min_y2, out[3] + v_gap)
            ranges2: list[Tuple[float, float]] = []
            if min_y2 <= above_max2:
                ranges2.append((min_y2, above_max2))
            if below_min2 <= max_y2:
                ranges2.append((below_min2, max_y2))
            new_y2 = _choose_y_from_ranges(rng, inc[1], ranges2)
            if new_y2 is not None:
                inc = _set_rect_y1(inc, new_y2)
                inc = _clamp_rect_topleft(inc, face, margin=margin)

    return inc, out



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
    seed: Optional[int] = None,
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
            "seed": seed,
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
        seed = resolved.get("seed", None)
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
        "face_image": None,
        "face_image_fit": "cover",
        "face_image_alpha": 255,
        "face_image_inset": None,
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

        
        # Optional: texture the front face with an image.
        # If fit == 'match_aspect', adjust the face dimensions to match the image aspect ratio.
        spec, fit_mode, _, _inset = _parse_face_image_style(style)
        if spec:
            face_img = style.get("_face_image_obj", None)
            if not isinstance(face_img, Image.Image):
                face_img = _load_face_image(spec)
                if face_img is not None:
                    style["_face_image_obj"] = face_img
            if isinstance(face_img, Image.Image) and fit_mode == "match_aspect":
                w_px, h_px = _adjust_wh_for_image_aspect(w_px, h_px, face_img, min_xy=min_xy, max_xy=max_xy)

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

    
    # Pre-sample per-layer patch placement ratios so boxes appear randomly placed on faces,
    # while keeping outgoing boxes on a layer to the right of incoming boxes.
    patch_ratios_in: Dict[int, Tuple[float, float]] = {}
    patch_ratios_out: Dict[int, Tuple[float, float]] = {}
    for li, obj in enumerate(stacks):
        lname = getattr(obj.get("layer"), "name", f"layer_{li}")
        if li > 0:
            rng_in = random.Random(_stable_seed(seed, "in", li, lname))
            patch_ratios_in[li] = _sample_patch_ratios(rng_in, x_lo=0.06, x_hi=0.44)
        if li < len(stacks) - 1:
            rng_out = random.Random(_stable_seed(seed, "out", li, lname))
            patch_ratios_out[li] = _sample_patch_ratios(rng_out, x_lo=0.56, x_hi=0.94)

# Build connections between consecutive rendered layers (after shift)
    connections = []
    if draw_connections and len(stacks) >= 2:
        # First pass: compute edge definitions (patch rectangles can be post-processed per-layer).
        edge_defs: list[Dict[str, Any]] = []
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

                # Place patch randomly on the source front face (outgoing patch is biased to the right).
                rx, ry = patch_ratios_out.get(i, (0.80, 0.50))
                sp = _place_patch_by_ratio((sx1, sy1, sx2, sy2), patch_w, patch_h, rx, ry, margin=1.0)

                # Destination activation patch: small, randomly placed (incoming patch is biased to the left).
                dsz = max(3, int(min(dst_stack.width, dst_stack.height) * 0.22))
                rx2, ry2 = patch_ratios_in.get(i + 1, (0.20, 0.50))
                dp = _place_patch_by_ratio((dx1, dy1, dx2, dy2), float(dsz), float(dsz), rx2, ry2, margin=1.0)

                edge_defs.append(
                    dict(
                        type='pyramid',
                        src=src_stack,
                        dst=dst_stack,
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
                edge_defs.append(
                    dict(
                        type='funnel',
                        src=src_stack,
                        dst=dst_stack,
                        connector_fill=conn_fill,
                        connector_width=conn_w,
                    )
                )
            else:
                edge_defs.append(
                    dict(
                        type='full',
                        src=src_stack,
                        dst=dst_stack,
                        connector_fill=conn_fill,
                        connector_width=conn_w,
                    )
                )

        # Second pass: if a layer has both an incoming and outgoing patch, enforce ordering + separation.
        # Incoming patch lives on layer j as dst_patch of edge (j-1). Outgoing patch lives on layer j as src_patch of edge j.
        for j in range(1, len(stacks) - 1):
            left_edge = edge_defs[j - 1]
            right_edge = edge_defs[j]
            if left_edge.get('type') == 'pyramid' and right_edge.get('type') == 'pyramid':
                layer_obj = stacks[j]
                lname = getattr(layer_obj.get('layer'), 'name', f'layer_{j}')
                rng_sep = random.Random(_stable_seed(seed, 'sep', j, lname))
                face = layer_obj['stack'].front_rect()
                inc = left_edge['dst_patch']
                out = right_edge['src_patch']
                inc2, out2 = _enforce_in_out_patch_separation(
                    face,
                    inc,
                    out,
                    rng=rng_sep,
                    x_eps=1.0,
                    v_gap=2.0,
                    margin=1.0,
                )
                left_edge['dst_patch'] = inc2
                right_edge['src_patch'] = out2

        # Final: instantiate connection objects.
        for ed in edge_defs:
            if ed['type'] == 'pyramid':
                connections.append(
                    PyramidConnection(
                        ed['src'],
                        ed['dst'],
                        src_patch=ed['src_patch'],
                        dst_patch=ed['dst_patch'],
                        connector_fill=ed['connector_fill'],
                        connector_width=ed['connector_width'],
                        patch_fill=ed['patch_fill'],
                        patch_outline=ed['patch_outline'],
                        draw_patches=ed.get('draw_patches', True),
                    )
                )
            elif ed['type'] == 'funnel':
                connections.append(
                    FunnelConnection(
                        ed['src'],
                        ed['dst'],
                        connector_fill=ed['connector_fill'],
                        connector_width=ed['connector_width'],
                    )
                )
            else:
                connections.append(
                    FullConnection(
                        ed['src'],
                        ed['dst'],
                        connector_fill=ed['connector_fill'],
                        connector_width=ed['connector_width'],
                    )
                )

    # Create canvas and draw
    img = Image.new("RGBA", (img_w, img_h), get_rgba_tuple(background_fill))

    # Pass 1: draw stacks (geometry + outlines)
    draw = aggdraw.Draw(img)
    for obj in stacks:
        obj["stack"].draw(draw)
    draw.flush()

    # Pass 1.5: paste optional face images onto each stack's front face
    _apply_face_images(img, stacks)

    # Pass 2: draw connections (polygons + patch boxes) over the faces
    draw = aggdraw.Draw(img)
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
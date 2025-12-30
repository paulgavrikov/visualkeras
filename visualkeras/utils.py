from typing import Any, Dict, Mapping, Union
from PIL import ImageColor, ImageDraw, Image
import aggdraw

def resolve_style(
    target: Any, 
    name: str, 
    styles: Mapping[Union[str, type], Dict[str, Any]], 
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generic style resolver.
    """
    final_style = defaults.copy()
    
    for cls in type(target).__mro__:
        if cls in styles:
            final_style.update(styles[cls])
    
    if name in styles:
        final_style.update(styles[name])
        
    return final_style


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: Any
    _outline: Any
    style: dict = None 

    def __init__(self):
        self.style = {}

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class Box(RectShape):
    de: int
    shade: int

    def draw(self, draw: ImageDraw, draw_reversed: bool = False):
        pen, brush = self._get_pen_brush()

        if hasattr(self, 'de') and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

            if draw_reversed:
                draw.line([self.x2 - self.de, self.y1 - self.de, self.x2 - self.de, self.y2 - self.de], pen)
                draw.line([self.x2 - self.de, self.y2 - self.de, self.x2, self.y2], pen)
                draw.line([self.x1 - self.de, self.y2 - self.de, self.x2 - self.de, self.y2 - self.de], pen)

                draw.polygon([self.x1, self.y1,
                              self.x1 - self.de, self.y1 - self.de,
                              self.x2 - self.de, self.y1 - self.de,
                              self.x2, self.y1
                              ], pen, brush_s1)

                draw.polygon([self.x1 - self.de, self.y1 - self.de,
                              self.x1, self.y1,
                              self.x1, self.y2,
                              self.x1 - self.de, self.y2 - self.de
                              ], pen, brush_s2)
            else:
                draw.line([self.x1 + self.de, self.y1 - self.de, self.x1 + self.de, self.y2 - self.de], pen)
                draw.line([self.x1 + self.de, self.y2 - self.de, self.x1, self.y2], pen)
                draw.line([self.x1 + self.de, self.y2 - self.de, self.x2 + self.de, self.y2 - self.de], pen)

                draw.polygon([self.x1, self.y1,
                              self.x1 + self.de, self.y1 - self.de,
                              self.x2 + self.de, self.y1 - self.de,
                              self.x2, self.y1
                              ], pen, brush_s1)

                draw.polygon([self.x2 + self.de, self.y1 - self.de,
                              self.x2, self.y1,
                              self.x2, self.y2,
                              self.x2 + self.de, self.y2 - self.de
                              ], pen, brush_s2)

        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Circle(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 1 * d, self.x1 + (w + d) / 2, self.y1 + 2 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 3 * d, self.x1 + (w + d) / 2, self.y1 + 4 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 5 * d, self.x1 + (w + d) / 2, self.y1 + 6 * d], pen, brush)


class ColorWheel:

    def __init__(self, colors: list = None):
        self._cache = dict()
        self.colors = colors if colors is not None else ["#ffd166", "#ef476f", "#118ab2", "#073b4c", "#842da1", "#ffbad4", "#fe9775", "#83d483", "#06d6a0", "#0cb0a9"]

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: Any) -> tuple:
    """

    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff, color >> 24 & 0xff)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def get_keys_by_value(d, v):
    for key in d.keys():  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple):
    """

    :param tensor_tuple:
    :return:
    """
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(im1: Image, im2: Image, background_fill: Any = 'white'):
    """
    Vertical concatenation of two PIL images.

    :param im1: top image
    :param im2: bottom image
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return: concatenated image
    """
    dst = Image.new('RGBA', (max(im1.width, im2.width), im1.height + im2.height), background_fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(images: list, max_width: int = -1, max_height: int = -1, horizontal: bool = True, padding: int = 0,
                  spacing: int = 0, background_fill: Any = 'white'):
    """
    Creates a linear layout of a passed list of images in horizontal or vertical orientation. The layout will wrap in x
    or y dimension if a maximum value is exceeded.

    :param images: List of PIL images
    :param max_width: Maximum width of the image. Only enforced in horizontal orientation.
    :param max_height: Maximum height of the image. Only enforced in vertical orientation.
    :param horizontal: If True, will draw images horizontally, else vertically.
    :param padding: Top, bottom, left, right border distance in pixels.
    :param spacing: Spacing in pixels between elements.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return:
    """
    coords = list()
    width = 0
    height = 0

    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                # make a new row
                x = padding
                y = height - padding + spacing
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                # make a new column
                x = width - padding + spacing
                y = padding
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            y += img.height + spacing

    layout = Image.new('RGBA', (width, height), background_fill)
    for img, coord in zip(images, coords):
        layout.paste(img, coord)

    return layout

class Ribbon:
    def __init__(self, x1, y1, x2, y2, de, width, color, shade_step):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.de = de
        self.width = width
        self.fill = get_rgba_tuple(color)
        self.shade = shade_step
        
        # Calculate depth sort key for layering ribbons correctly
        self.z_sort = (x1 + x2) / 2 + (y1 + y2) / 2

    def draw(self, draw: aggdraw.Draw):
        pen = aggdraw.Pen("black", 0.5) # Thin outline for crispness
        
        # Colors
        top_color = fade_color(self.fill, self.shade)
        side_color = fade_color(self.fill, 2 * self.shade)
        front_color = self.fill
        
        brush_top = aggdraw.Brush(top_color)
        brush_side = aggdraw.Brush(side_color)
        brush_front = aggdraw.Brush(front_color)

        # A horizontal ribbon is a rectangle of height 'width'
        # A vertical ribbon is a rectangle of width 'width'
        
        is_horizontal = abs(self.y1 - self.y2) < abs(self.x1 - self.x2)
        
        if is_horizontal:
            # Draw Horizontal Ribbon (Left -> Right)
            lx, rx = min(self.x1, self.x2), max(self.x1, self.x2)
            y = self.y1 
            w = self.width
            
            # 1. Back Face (Top)
            # 2. Top Face (Depth)
            # Polygon: (lx, y), (rx, y), (rx+de, y-de), (lx+de, y-de)
            draw.polygon([
                lx, y - w/2, 
                rx, y - w/2, 
                rx + self.de, y - w/2 - self.de, 
                lx + self.de, y - w/2 - self.de
            ], pen, brush_top)
            
            # 3. Front Face (The main line)
            draw.rectangle([lx, y - w/2, rx, y + w/2], pen, brush_front)
            
        else:
            # Draw Vertical Ribbon (Top -> Bottom)
            ty, by = min(self.y1, self.y2), max(self.y1, self.y2)
            x = self.x1
            w = self.width
            
            # 1. Side Face
            # Polygon: (x+w/2, ty), (x+w/2, by), (x+w/2+de, by-de), (x+w/2+de, ty-de)
            draw.polygon([
                x + w/2, ty,
                x + w/2, by,
                x + w/2 + self.de, by - self.de,
                x + w/2 + self.de, ty - self.de
            ], pen, brush_side)

            # 2. Top Face
            draw.polygon([
                x - w/2, ty,
                x + w/2, ty,
                x + w/2 + self.de, ty - self.de,
                x - w/2 + self.de, ty - self.de
            ], pen, brush_top)

            # 3. Front Face
            draw.rectangle([x - w/2, ty, x + w/2, by], pen, brush_front)
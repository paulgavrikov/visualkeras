from typing import Any
from PIL import ImageColor, ImageDraw, Image
import aggdraw


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: Any
    _outline: Any

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

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()

        if hasattr(self, 'de') and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

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
        self.colors = colors if colors is not None else ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#073b4c"]

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

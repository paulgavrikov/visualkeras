from typing import Any
from PIL import ImageColor


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    fill: Any
    outline: Any


class Box(RectShape):
    de: int


class Circle(RectShape):
    pass


class Ellipses(RectShape):
    pass


class ColorWheel:

    def __init__(self, colors: list = ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#073b4c"]):
        self._cache = dict()
        self.colors = colors

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def get_rgba_tuple(color) -> tuple:
    """

    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        return color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff, 255
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

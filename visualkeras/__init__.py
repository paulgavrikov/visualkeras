from typing import Any
from PIL import Image, ImageDraw
from tensorflow.python.keras.layers import Layer
from math import ceil

class Box:
    x1: int
    x2: int
    y1: int
    y2: int
    de: int
    fill: Any
    outline: Any


class FakeLayer(Layer):

    def __init__(self, padding: int = 50):
        super().__init__()
        self.padding = padding


# TODO
# default colormap
# optional: add layer groups
# optional: translucent layers


def cnn_arch(model, out_file: str = None, min_x: int = 20, min_yz: int = 20, max_x: int = 400,
             max_yz: int = 2000,
             scale_depth: float = 0.1, scale_height: float = 4, type_ignore: list = [], index_ignore: list = [],
             color_map: dict = {}, dense_orientation: str = 'x',
             background_fill: Any = 'white', draw_volume=True, padding=10,
             distance = 10, funnel: bool = True) -> Image:
    """
    Generates a architecture visualization for a given linear keras model (i.e. one input and output tensor for each layer) in style of a Convolutional Neural Network.

    :param model: A keras model that will be visualized.
    :param out_file: Path to the file to write the created image to. If the image does not exist yet it will be created, else overwritten. Image type is inferred from the file ending. Providing None will disable writing.
    :param min_x: Minimum x size a layer will have.
    :param min_yz: Minimum y and z size a layer will have.
    :param max_x: Maximum x size a layer will have.
    :param max_yz: Maximum y and z size a layer will have.
    :param scale_depth: Scalar multiplier for the height of each layer.
    :param scale_height: Scalar multiplier for the height of each layer.
    :param type_ignore: List of layer types in the keras model to ignore during drawing.
    :param index_ignore: List of layer indexes in the keras model to ignore during drawing.
    :param color_map: Dict defining fill and outline for each layer by class type. Will fallback to default values for not specified classes.
    :param dense_orientation: Axis on which one dimensional layers should be drawn. Can  be 'x', 'y' or 'z'.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :param draw_volume: Flag to switch between 3D volumetric view and 2D box view.
    :param padding: Distance in pixel before the first and after the last layer.
    :param distance: Distance in pixel between two layers
    :param funnel: If set to True, a funnel will be drawn between consecutive layers

    :return: Generated architecture image.
    """

    # Iterate over the model to compute bounds and generate boxes

    boxes = list()
    current_x = padding
    x_off = -1
    img_height = 0
    img_width = 0
    for index, layer in enumerate(model.layers):

        print(layer, layer.input_shape, layer.output_shape)

        if type(layer) in type_ignore or index in index_ignore:
            continue

        if type(layer) == FakeLayer:
            current_x += layer.padding
            continue

        w = min_yz
        h = min_yz
        d = min_x

        if isinstance(layer.output_shape, tuple):
            shape = layer.output_shape
        elif isinstance(layer.output_shape, list) and len(layer.output_shape) == 1:
            shape = layer.output_shape[0]
        else:
            raise RuntimeError(f"not supported tensor shape {layer.output_shape}")

        if len(shape) == 4:
            w = min(max(shape[1] * scale_height, w), max_yz)
            h = min(max(shape[2] * scale_height, h), max_yz)
            d = min(max(shape[3] * scale_depth, d), max_x)
        elif len(shape) == 3:
            w = min(max(shape[1] * scale_height, w), max_yz)
            h = min(max(shape[2] * scale_height, h), max_yz)
            d = min(max(d), max_x)
        elif len(shape) == 2:
            if dense_orientation == 'x':
                d = min(max(shape[1] * scale_depth, min_x), max_x)
            elif dense_orientation == 'y':
                h = min(max(shape[1] * scale_height, h), max_yz)
            elif dense_orientation == 'z':
                w = min(max(shape[1] * scale_height, w), max_yz)
            else:
                raise ValueError(f"unsupported orientation {dense_orientation}")
        else:
            raise RuntimeError(f"not supported tensor shape {layer.output_shape}")

        box = Box()

        box.de = 0
        if draw_volume:
            box.de = w / 3

        if x_off == -1:
            x_off = box.de / 2

        # top left coordinate
        box.x1 = current_x - box.de / 2
        box.y1 = box.de

        # bottom right coordinate
        box.x2 = box.x1 + d
        box.y2 = box.y1 + h

        box.fill = color_map.get(type(layer), {}).get('fill', 'orange')
        box.outline = color_map.get(type(layer), {}).get('outline', 'black')

        boxes.append(box)

        # Update image bounds
        hh = box.y2 - (box.y1 - box.de)
        if hh > img_height:
            img_height = hh
        img_width = box.x2 + box.de + x_off + padding

        current_x += d + distance

    # Generate image

    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)
    draw = ImageDraw.Draw(img)

    # draw.line([0, img.height / 2, img.width, img.height / 2], fill='red', width=5)
    # draw.line([img.width / 2, 0, img.width / 2, img.height], fill='red', width=5)

    # Draw created boxes

    last_box = None

    for box in boxes:
        fill = box.fill
        outline = box.outline
        y_off = (img.height - (box.y2 - (box.y1 - box.de))) / 2

        box.y1 += y_off
        box.y2 += y_off

        box.x1 += x_off
        box.x2 += x_off

        x1 = box.x1
        x2 = box.x2
        y1 = box.y1
        y2 = box.y2
        de = box.de

        if last_box is not None and funnel:
            draw.line([last_box.x2 + last_box.de, last_box.y1 - last_box.de,
                       x1 + de, y1 - de], fill=outline)

            draw.line([last_box.x2 + last_box.de, last_box.y2 - last_box.de,
                       x1 + de, y2 - de], fill=outline)

            draw.line([last_box.x2, last_box.y2,
                       x1, y2], fill=outline)

            draw.line([last_box.x2, last_box.y1,
                       x1, y1], fill=outline)


        draw.rectangle([x1, y1, x2, y2],
                       fill=fill,
                       outline=outline)

        draw.polygon([x1, y1,
                      x1 + de, y1 - de,
                      x2 + de, y1 - de,
                      x2, y1
                      ], fill=fill, outline=outline)

        draw.polygon([x2 + de, y1 - de,
                      x2, y1,
                      x2, y2,
                      x2 + de, y2 - de
                      ], fill=fill, outline=outline)

        # draw.line([x1 + de, y1 - de, x1 + de, y2 - de], fill=outline)
        # draw.line([x1 + de, y2 - de, x1, y2], fill=outline)
        # draw.line([x1 + de, y2 - de, x2 + de, y2 - de], fill=outline)

        last_box = box

    if out_file is not None:
        img.save(out_file)

    return img

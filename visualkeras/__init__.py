from typing import Any
from PIL import Image, ImageDraw
from tensorflow.python.keras.layers import Layer


class FakeLayer(Layer):

    def __init__(self, padding: int = 100):
        super().__init__()
        self.padding = padding


# TODO
# calculate and shift x offset
# calculate and set image size
# optional: add layer groups / group paddings
# optional: translucent layers


def cnn_arch(model, out_file: str = None, min_x: int = 20, min_yz: int = 20, max_x: int = 400,
             max_yz: int = 2000,
             scale_depth: float = 0.3, scale_height: float = 5, type_ignore: list = [], index_ignore: list = [],
             color_map: dict = {}, dense_orientation: str = 'x',
             background_fill: Any = 'gray', text_fill: Any = 'black', draw_volume=True) -> Image:
    img = Image.new('RGBA', (3000, max_yz), background_fill)
    draw = ImageDraw.Draw(img)

    draw.line([0, img.height / 2, img.width, img.height / 2], fill='red', width=5)
    draw.line([img.width / 2, 0, img.width / 2, img.height], fill='red', width=5)

    current_x = 373.3333333333333 / 2

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

        fill = color_map.get(type(layer), {}).get('fill', 'orange')
        outline = color_map.get(type(layer), {}).get('outline', 'black')

        if len(layer.output_shape) == 4:
            w = min(max(layer.output_shape[1] * scale_height, w), max_yz)
            h = min(max(layer.output_shape[2] * scale_height, h), max_yz)
            d = min(max(layer.output_shape[3] * scale_depth, d), max_x)
        elif len(layer.output_shape) == 3:
            w = min(max(layer.output_shape[1] * scale_height, w), max_yz)
            h = min(max(layer.output_shape[2] * scale_height, h), max_yz)
            d = min(max(d), max_x)
        elif len(layer.output_shape) == 2:
            if dense_orientation == 'x':
                d = min(max(layer.output_shape[1] * scale_depth, min_x), max_x)
            elif dense_orientation == 'y':
                h = min(max(layer.output_shape[1] * scale_height, h), max_yz)
            elif dense_orientation == 'z':
                w = min(max(layer.output_shape[1] * scale_height, w), max_yz)
            else:
                print(f"unsupported orientation {dense_orientation}")
        else:
            print(f"not supported tensor shape {layer.output_shape}")
            continue

        de = 0
        if draw_volume:
            de = w / 3

        # bottom left coordinate
        x1 = current_x - de / 2

        y1 = (img.height - h) / 2 + de / 2

        # top right coordinate
        x2 = x1 + d
        y2 = y1 + h

        draw.rectangle([x1, y1, x2, y2],
                       fill=fill,
                       outline=outline)

        draw.polygon([x1, y1,
                      x1 + de, y1 - de,
                      x1 + de + d, y1 - de,
                      x1 + d, y1
                      ], fill=fill, outline=outline)

        draw.polygon([x1 + de + d, y1 - de,
                      x1 + d, y1,
                      x2, y2,
                      x2 + de, y2 - de
                      ], fill=fill, outline=outline)

        # draw.line([x1 + de, y1 - de, x1 + de, y1 - de + h], fill=outline)
        # draw.line([x1 + de, y1 - de + h, x1, y1 + h], fill=outline)
        # draw.line([x1 + de, y1 - de + h, x2 + de, y2 - de], fill=outline)

        # draw.rectangle([x1 + de, y1 - de, x2 + de, y2 - de],
        #                fill=fill,
        #                outline=outline)

        label = 'x'.join(map(str, list(layer.output_shape[1:])))

        draw.text((x1, y2 + 10), label, fill=text_fill)  # todo magic value

        current_x += d

    if out_file is not None:
        img.save(out_file)

    return img

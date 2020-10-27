from typing import Any
import aggdraw
from PIL import Image, ImageDraw
from math import ceil
from .utils import *
from .layer_utils import *

try:
    from tensorflow.python.keras.layers import Dense, InputLayer
except:
    from keras.layers import Dense, InputLayer


class _DummyLayer:

    def __init__(self, units, name):
        # self.units = units
        self.name = name


def graph_view(model, to_file: str = None,
               color_map: dict = {}, node_size: int = 50,
               background_fill: Any = 'white', padding: int = 10,
               layer_spacing: int = 250, node_spacing: int = 10, connector_fill: Any = 'gray',
               connector_width: int = 1, ellipsize_after: int = 10,
               inout_as_tensor: bool = True, show_neurons: bool = True) -> Image:
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

    :return: Generated architecture image.
    """

    # Iterate over the model to compute bounds and generate boxes

    # TODO: assert len of input and outlabels

    layers = list()
    # layer_labels = list()
    layer_y = list()

    if inout_as_tensor:
        inputs = len(model.inputs)
        outputs = len(model.outputs)
    else:
        inputs = self_multiply(model.input_shape)
        outputs = self_multiply(model.output_shape)
    #
    # # Make labels if not provided
    #
    # if not input_labels:
    #     input_labels = [f"Input {i}" for i in range(inputs)]
    #
    # if not output_labels:
    #     output_labels = [f"Output {i}" for i in range(outputs)]
    #
    # # Measure label size
    #
    # fake_img = Image.new('RGB', (0, 0))
    # fake_draw = ImageDraw.Draw(fake_img)
    # input_label_size = max([fake_draw.textsize(text=s) for s in input_labels])
    # output_label_size = max([fake_draw.textsize(text=s) for s in output_labels])
    #
    # del fake_img, fake_draw

    # Attach helper layers

    id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    model_layers = model_to_hierarchy_lists(model, id_to_num_mapping, adj_matrix)

    # add fake output layers
    num_outs = len(model.outputs)
    adj_matrix = np.pad(adj_matrix, ((0, num_outs), (0, num_outs)), mode='constant', constant_values=0)
    dummy_outputs = [_DummyLayer(1, 'output_' + str(i + 1)) for i in range(num_outs)]
    model_layers.append(dummy_outputs)
    for dummy_output in dummy_outputs:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(dummy_outputs[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    # if not hasattr(model_layers[-1], 'filters') and not hasattr(model_layers[-1], 'units'):
    #     model_layers += [_DummyLayer(units=outputs, is_input=False)]

    # Create architecture

    current_x = padding  # + input_label_size[0] + text_padding

    id_to_node_list_map = dict()

    for index, layer_list in enumerate(model_layers):
        current_y = 0
        nodes = []
        for layer in layer_list:
            # label = layer.name
            #
            # if isinstance(layer, Dense):
            #     label += f"({layer.units}, {str(layer.activation.__name__)})"
            #
            # layer_labels.append(label)

            units = 1
            is_box = True
            if show_neurons:
                if hasattr(layer, 'units'):
                    is_box = False
                    units = layer.units
                elif hasattr(layer, 'filters'):
                    is_box = False
                    units = layer.filters

            n = min(units, ellipsize_after)
            layer_nodes = list()

            for i in range(n):
                scale = 1
                if not is_box:
                    if i != ellipsize_after - 2:
                        c = Circle()
                    else:
                        c = Ellipses()
                else:
                    c = Box()
                    scale = 3

                c.x1 = current_x
                c.y1 = current_y
                c.x2 = c.x1 + node_size
                c.y2 = c.y1 + node_size * scale

                current_y = c.y2 + node_spacing

                c.fill = color_map.get(type(layer), {}).get('fill', 'orange')
                c.outline = color_map.get(type(layer), {}).get('outline', 'black')

                layer_nodes.append(c)

            id_to_node_list_map[id(layer)] = layer_nodes
            nodes.extend(layer_nodes)
            current_y += 2 * node_size

        layer_y.append(current_y - node_spacing - 2 * node_size)
        layers.append(nodes)
        current_x += node_size + layer_spacing

    # Generate image

    img_width = len(layers) * node_size + (len(layers) - 1) * layer_spacing + 2 * padding \
               # + input_label_size[0] + output_label_size[0] + text_padding * 2
    img_height = max(*layer_y) + 2 * padding \
                # + text_padding + input_label_size[1]
    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)

    draw = aggdraw.Draw(img)

    # y correction (centering)
    for i, layer in enumerate(layers):
        y_off = (img.height - layer_y[i]) / 2
        # y_off += text_padding + input_label_size[1]
        for node in layer:
            node.y1 += y_off
            node.y2 += y_off

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

        # layer_label_w = draw.textsize(layer_labels[i])
        # draw.text((layer[0].x1 + (layer[0].x2 - layer[0].x1 - layer_label_w[0]) / 2, padding), layer_labels[i],
        #           fill=text_color)

        for node_index, node in enumerate(layer):

            # Draw labels

            # if i == 0 and not isinstance(node, Ellipses):
            #     cy = node.y1 + (node.y2 - node.y1 - input_label_size[1]) / 2
            #     text = input_labels[node_index] if node_index != len(layer) - 1 else input_labels[-1]
            #     draw.text((padding, cy), text, fill=text_color)
            #
            # if i == len(layers) - 1 and not isinstance(node, Ellipses):
            #     cy = node.y1 + (node.y2 - node.y1 - output_label_size[1]) / 2
            #     text = output_labels[node_index] if node_index != len(layer) - 1 else output_labels[-1]
            #     draw.text((node.x2 + text_padding, cy), text, fill=text_color)

            _draw_node(node, draw)

    draw.flush()

    if to_file is not None:
        img.save(to_file)

    return img


def _draw_connector(draw, start_node, end_node, color, width):
    pen = aggdraw.Pen(color, width)
    x1 = start_node.x2
    y1 = start_node.y1 + (start_node.y2 - start_node.y1) / 2
    x2 = end_node.x1
    y2 = end_node.y1 + (end_node.y2 - end_node.y1) / 2
    draw.line([x1, y1, x2, y2], pen)


def _draw_node(node: RectShape, draw: ImageDraw):
    pen = aggdraw.Pen(node.outline)
    brush = aggdraw.Brush(node.fill)
    if isinstance(node, Circle):
        draw.ellipse([node.x1, node.y1, node.x2, node.y2], pen, brush)
    elif isinstance(node, Box):
        draw.rectangle([node.x1, node.y1, node.x2, node.y2], pen, brush)
    elif isinstance(node, Ellipses):
        w = node.x2 - node.x1
        d = int(w / 7)
        draw.ellipse([node.x1 + (w - d) / 2, node.y1 + 1 * d, node.x1 + (w + d) / 2, node.y1 + 2 * d], pen, brush)
        draw.ellipse([node.x1 + (w - d) / 2, node.y1 + 3 * d, node.x1 + (w + d) / 2, node.y1 + 4 * d], pen, brush)
        draw.ellipse([node.x1 + (w - d) / 2, node.y1 + 5 * d, node.x1 + (w + d) / 2, node.y1 + 6 * d], pen, brush)

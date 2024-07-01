import numpy as np
from .utils import get_keys_by_value
from collections.abc import Iterable

try:
    from tensorflow.keras.layers import Layer
except ModuleNotFoundError:
    try:
        from keras.layers import Layer
    except ModuleNotFoundError:
        pass


class SpacingDummyLayer(Layer):

    def __init__(self, spacing: int = 50):
        super().__init__()
        self.spacing = spacing


def get_incoming_layers(layer):
    for i, node in enumerate(layer._inbound_nodes):
        if isinstance(node.inbound_layers, Iterable):
            for inbound_layer in node.inbound_layers:
                yield inbound_layer
        else:  # for tf 2.3
            yield node.inbound_layers


def get_outgoing_layers(layer):
    for i, node in enumerate(layer._outbound_nodes):
        yield node.outbound_layer


def model_to_adj_matrix(model):
    if hasattr(model, 'built'):
        if not model.built:
            model.build()
            
    layers = []
    if hasattr(model, '_layers'):
        layers = model._layers
    elif hasattr(model, '_self_tracked_trackables'):
        layers = model._self_tracked_trackables

    adj_matrix = np.zeros((len(layers), len(layers)))
    id_to_num_mapping = dict()

    for layer in layers:
        layer_id = id(layer)
        if layer_id not in id_to_num_mapping:
            id_to_num_mapping[layer_id] = len(id_to_num_mapping.keys())

        for inbound_layer in get_incoming_layers(layer):
            inbound_layer_id = id(inbound_layer)

            if inbound_layer_id not in id_to_num_mapping:
                id_to_num_mapping[inbound_layer_id] = len(id_to_num_mapping.keys())

            src = id_to_num_mapping[inbound_layer_id]
            tgt = id_to_num_mapping[layer_id]
            adj_matrix[src, tgt] += 1

    return id_to_num_mapping, adj_matrix


def find_layer_by_id(model, _id):
    layers = []
    if hasattr(model, '_layers'):
        layers = model._layers
    elif hasattr(model, '_self_tracked_trackables'):
        layers = model._self_tracked_trackables
    
    for layer in layers: # manually because get_layer does not access model._layers
            if id(layer) == _id:
                return layer
    return None


def find_layer_by_name(model, name):
    layers = []
    if hasattr(model, '_layers'):
        layers = model._layers
    elif hasattr(model, '_self_tracked_trackables'):
        layers = model._self_tracked_trackables
    
    for layer in layers:
        if layer.name == name:
            return layer
    return None


def find_input_layers(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    for i in np.where(np.sum(adj_matrix, axis=0) == 0)[0]:  # find all nodes with 0 inputs
        for key in get_keys_by_value(id_to_num_mapping, i):
            yield find_layer_by_id(model, key)


def find_output_layers(model):
    for name in model.output_names:
        yield model.get_layer(name=name)


def model_to_hierarchy_lists(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    hierarchy = [set(find_input_layers(model, id_to_num_mapping, adj_matrix))]
    prev_layers = set(hierarchy[0])
    finished = False

    while not finished:
        layer = list()
        finished = True
        for start_layer in hierarchy[-1]:
            start_layer_idx = id_to_num_mapping[id(start_layer)]
            for end_layer_idx in np.where(adj_matrix[start_layer_idx] > 0)[0]:
                finished = False
                for end_layer_id in get_keys_by_value(id_to_num_mapping, end_layer_idx):
                    end_layer = find_layer_by_id(model, end_layer_id)
                    incoming_to_end_layer = set(get_incoming_layers(end_layer))
                    intersection = set(incoming_to_end_layer).intersection(prev_layers)
                    if len(intersection) == len(incoming_to_end_layer):
                        if end_layer not in layer:
                            layer.append(end_layer)
                            prev_layers.add(end_layer)
        if not finished:
            hierarchy.append(layer)

    return hierarchy


def augment_output_layers(model, output_layers, id_to_num_mapping, adj_matrix):

    adj_matrix = np.pad(adj_matrix, ((0, len(output_layers)), (0, len(output_layers))), mode='constant', constant_values=0)

    for dummy_output in output_layers:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(output_layers[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    try:
        import tensorflow.python.keras.engine.input_layer.InputLayer
        if isinstance(layer, tensorflow.python.keras.engine.input_layer.InputLayer):
            return True
    except ModuleNotFoundError:
        pass

    try:
        import keras
        if isinstance(layer, keras.engine.input_layer.InputLayer):
            return True
    except ModuleNotFoundError:
        pass

    return False

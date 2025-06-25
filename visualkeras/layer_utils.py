import numpy as np
from .utils import get_keys_by_value
from collections.abc import Iterable
import warnings

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
        if hasattr(node, 'inbound_layers'):
            # Old Node class (TF 2.15 & Keras 2.15 and under)
            if isinstance(node.inbound_layers, Iterable):
                for inbound_layer in node.inbound_layers:
                    yield inbound_layer
            else:  # For older versions like TF 2.3
                yield node.inbound_layers
        else:
            # New Node class (TF 2.16 and Keras 3 and up)
            inbound_layers = [parent_node.operation for parent_node in node.parent_nodes]
            if isinstance(inbound_layers, Iterable):
                for inbound_layer in inbound_layers:
                    yield inbound_layer
            else:
                yield inbound_layers


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
    if hasattr(model, 'output_names'):
        # For older Keras versions (<3)
        for name in model.output_names:
            yield model.get_layer(name=name)
    else:
        # For newer Keras versions (>=3)
        for output in model.outputs:
            if hasattr(output, '_keras_history'):
                # Get the layer that produced the output
                layer = output._keras_history[0]
                yield layer


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
    # Treat any InputLayer class as internal input
    if layer.__class__.__name__ == 'InputLayer':
        return True
    # Detect tf.keras InputLayer (newer versions)
    try:
        from tensorflow.keras.layers import InputLayer as _TfInputLayer
        if isinstance(layer, _TfInputLayer):
            return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    # Detect standalone keras InputLayer
    try:
        from keras.layers import InputLayer as _KerasInputLayer
        if isinstance(layer, _KerasInputLayer):
            return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass
    # Fallback for older internal paths
    try:
        import tensorflow.python.keras.engine.input_layer.InputLayer
        if isinstance(layer, tensorflow.python.keras.engine.input_layer.InputLayer):
            return True
    except (ModuleNotFoundError, AttributeError):
        pass

    try:
        # From versions Keras 2.13+ the Keras module may store all its code in a src subfolder
        import tensorflow.python.keras.src.keras.engine.input_layer.InputLayer 
        if isinstance(layer, tensorflow.python.keras.src.engine.input_layer.InputLayer):
            return True
    except (ModuleNotFoundError, AttributeError):
        pass

    try:
        import keras
        if isinstance(layer, keras.engine.input_layer.InputLayer):
            return True
    except (ModuleNotFoundError, AttributeError):
        pass

    try:
        import keras
        if isinstance(layer, keras.src.engine.input_layer.InputLayer):
            return True
    except (ModuleNotFoundError, AttributeError):
        pass

    return False

def extract_primary_shape(layer_output_shape, layer_name: str = None) -> tuple:
    """
    Extract the primary shape from a layer's output shape to handle multi-output scenarios.
    
    This function addresses the issue where some layers (like TransformerBlock in ViT models)
    have multiple outputs, resulting in a tuple of shapes rather than a single shape tuple.
    For visualization purposes, we need to extract the primary/main output shape.
    
    Args:
        layer_output_shape: The output shape from a Keras layer. Can be:
            - A single shape tuple: (None, height, width, channels)
            - A tuple of shape tuples: ((None, 197, 1024), (None, 16, None, None))
            - A list of shape tuples: [(None, 197, 1024), (None, 16, None, None)]
    
    Returns:
        tuple: The primary shape tuple to use for visualization. Always returns a single
            shape tuple in the format (batch_size, dim1, dim2, ...).
    
    Examples:
        >>> # Single output case
        >>> extract_primary_shape((None, 224, 224, 3))
        (None, 224, 224, 3)
        
        >>> # Multi-output case (TransformerBlock)
        >>> extract_primary_shape(((None, 197, 1024), (None, 16, None, None)))
        (None, 197, 1024)
        
        >>> # List of outputs case
        >>> extract_primary_shape([(None, 197, 1024), (None, 16, None, None)])
        (None, 197, 1024)
    
    Notes:
        - For multi-output layers, the first output is considered the primary output
        - The function assumes the primary output contains the main feature representations
        - Secondary outputs (like attention weights) are ignored for visualization purposes
    """
    # Handle None or empty cases
    layer_info = f" (layer: {layer_name})" if layer_name else ""
    
    # Handle None case - warn and use default
    if layer_output_shape is None:
        warnings.warn(
            f"Layer output shape is None{layer_info}. This may indicate an unbuilt model "
            f"or invalid layer configuration. Using default shape (None, 1) for visualization.",
            UserWarning,
            stacklevel=3
        )
        return (None, 1)
    
    # Handle tuple case
    if isinstance(layer_output_shape, tuple):
        # Check if this is a multi-output scenario (tuple of tuples)
        if len(layer_output_shape) > 0 and isinstance(layer_output_shape[0], (tuple, list)):
            # Multi-output case
            warnings.warn(
                f"Multi-output layer detected{layer_info}. "
                f"Using primary output shape {layer_output_shape[0]} for visualization. "
                f"Secondary outputs {layer_output_shape[1:]} will be ignored.",
                UserWarning,
                stacklevel=3
            )
            return layer_output_shape[0]
        else:
            # Single output tuple
            return layer_output_shape
    
    # Handle list case
    elif isinstance(layer_output_shape, list):
        if len(layer_output_shape) == 1:
            # Single output
            return layer_output_shape[0]
        elif len(layer_output_shape) > 1:
            # Multi-output list
            warnings.warn(
                f"Multi-output layer detected{layer_info}. "
                f"Using primary output shape {layer_output_shape[0]} for visualization. "
                f"Secondary outputs {layer_output_shape[1:]} will be ignored.",
                UserWarning,
                stacklevel=3
            )
            return layer_output_shape[0]
        else:
            # Empty list
            warnings.warn(
                f"Layer output shape is an empty list{layer_info}. This indicates an invalid "
                f"layer configuration. Using default shape (None, 1) for visualization.",
                UserWarning,
                stacklevel=3
            )
            return (None, 1)
    
    # Unsupported format
    else:
        raise RuntimeError(
            f"Unsupported tensor shape format: {type(layer_output_shape).__name__} = {layer_output_shape}{layer_info}. "
            f"Expected tuple or list, but got {type(layer_output_shape).__name__}."
        )

def calculate_layer_dimensions(shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
                               one_dim_orientation='y', sizing_mode='accurate',
                               dimension_caps=None, relative_base_size=20):
    """
    Calculate layer dimensions for visualization with flexible sizing strategies.

    Args:
        shape (tuple): The layer shape (batch_size, dim1, dim2, ...)
        scale_z (float): Base scaling factor for z-dimension
        scale_xy (float): Base scaling factor for xy-dimensions
        max_z (int): Maximum z size in pixels
        max_xy (int): Maximum xy size in pixels
        min_z (int): Minimum z size in pixels
        min_xy (int): Minimum xy size in pixels
        one_dim_orientation (str): Orientation for 1D layers ('x','y','z')        sizing_mode (str): 'accurate', 'capped', 'balanced', 'logarithmic', or 'relative'
        dimension_caps (dict): Custom caps for 'channels','sequence','general'
        relative_base_size (int): Base size in pixels for relative scaling mode. 
            Represents the visual size (in pixels) that a dimension of size 1 would have.
            All dimensions scale proportionally: visual_size = dimension × relative_base_size.
            For example, if relative_base_size=5:
            - A layer with 64 units gets visual size 64×5=320 pixels
            - A layer with 32 units gets visual size 32×5=160 pixels (exactly half)
            - A layer with 16 units gets visual size 16×5=80 pixels (exactly half of 32)
            This maintains true proportional relationships between all layers. (default: 20)

    Returns:
        tuple: (x, y, z) pixel dimensions for the layer box
          Sizing Modes:
        - 'accurate': Use actual dimensions with scaling (may create very large visualizations)
        - 'balanced': Smart scaling that balances accuracy with visual clarity
        - 'capped': Cap dimensions at specified limits while preserving ratios
        - 'logarithmic': Use logarithmic scaling for very large dimensions
        - 'relative': True proportional scaling where visual_size = dimension × relative_base_size.
                     Each layer's visual size is directly proportional to its dimension count.
                     If one layer has 2x the dimensions of another, it will be 2x the visual size.
    """

    import math
    from .utils import self_multiply

    # Extract numeric dims (skip batch dim)
    dims = [d for d in shape[1:] if isinstance(d, int) and d is not None]

    # Default fallback
    if not dims:
        return (min_xy, min_xy, min_z)

    # Setup caps
    channel_cap = dimension_caps.get('channels', max_z) if dimension_caps else max_z
    sequence_cap = dimension_caps.get('sequence', max_xy) if dimension_caps else max_xy
    general_cap = dimension_caps.get('general', max(max_z, max_xy)) if dimension_caps else max(max_z, max_xy)

    def smart_scale(value, base_scale, min_val, cap_val):
        """
        Smart scaling that maintains relative proportions while preventing extremely large visualizations.
        This provides the relative scaling functionality that was promised by the relative_scaling parameter.
        """
        if value <= 64:
            # Small dimensions: use full scaling to make them visible
            return min(max(value * base_scale, min_val), cap_val)
        elif value <= 512:
            # Medium dimensions: reduce scaling to balance visibility and proportion
            return min(max(value * base_scale * 0.6, min_val), cap_val)
        elif value <= 2048:
            # Large dimensions: further reduce scaling but maintain relative differences
            return min(max(value * base_scale * 0.3, min_val), cap_val)
        else:
            # Very large dimensions: use logarithmic scaling but still maintain relativity
            log_scale = math.log10(value) * base_scale * 15
            return min(max(log_scale, min_val), cap_val)

    def log_scale(value, base_scale, min_val, cap_val):
        if value <= 1:
            return min_val
        log_val = math.log10(value) * base_scale * 20
        return min(max(log_val, min_val), cap_val)

    # Accurate mode: mirror original block
    if sizing_mode == 'accurate':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = min(max(dims[0] * scale_xy, min_xy), max_xy)
                return (min_xy, y, min_z)
            else:
                z = min(max(dims[0] * scale_z, min_z), max_z)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(dims[1] * scale_z, min_z), max_z)
            return (x, y, z)
        else:
            x = min(max(dims[0] * scale_xy, min_xy), max_xy)
            y = min(max(dims[1] * scale_xy, min_xy), max_xy)
            z = min(max(self_multiply(dims[2:]) * scale_z, min_z), max_z)
            return (x, y, z)

    # Capped mode
    elif sizing_mode == 'capped':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
                return (min_xy, y, min_z)
            else:
                z = max(min(dims[0] * scale_z, channel_cap), min_z)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = max(min(dims[0] * scale_xy, sequence_cap), min_xy)
            y = max(min(dims[1] * scale_xy, sequence_cap), min_xy)
            z = max(min(dims[2] * scale_z if len(dims)>2 else min_z, channel_cap), min_z)
            return (x, y, z)

    # Balanced mode
    elif sizing_mode == 'balanced':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return (min_xy, int(y), min_z)
            else:
                z = smart_scale(dims[0], scale_z, min_z, channel_cap)
                return (min_xy, min_xy, int(z))
        else:
            x = smart_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = smart_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = smart_scale(dims[2] if len(dims)>2 else 1, scale_z, min_z, channel_cap)
            return (int(x), int(y), int(z))    # Logarithmic mode
    elif sizing_mode == 'logarithmic':
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
                return (min_xy, int(y), min_z)
            else:
                z = log_scale(dims[0], scale_z, min_z, channel_cap)
                return (min_xy, min_xy, int(z))
        else:
            x = log_scale(dims[0], scale_xy, min_xy, sequence_cap)
            y = log_scale(dims[1], scale_xy, min_xy, sequence_cap)
            z = log_scale(dims[2] if len(dims)>2 else 1, scale_z, min_z, channel_cap)
            return (int(x), int(y), int(z))

    # Relative mode - True proportional scaling where each layer's size is directly proportional to its dimension
    elif sizing_mode == 'relative':
        def proportional_scale(dimension, relative_base_size, min_val, max_val):
            """
            Scale dimension proportionally where relative_base_size represents 
            the visual size for dimension=1.
            
            Args:
                dimension (int): The dimension value to scale
                relative_base_size (int): Visual size (in pixels) for dimension=1
                min_val (int): Minimum allowed scaled value
                max_val (int): Maximum allowed scaled value
            
            Returns:
                int: Scaled dimension with true proportional relationships
                
            Formula: visual_size = dimension * relative_base_size
            """
            if dimension <= 0:
                return min_val
            
            # True proportional scaling: dimension * base_size
            scaled = dimension * relative_base_size
            
            # Apply min/max constraints while preserving proportionality as much as possible
            return max(min_val, min(scaled, max_val))
        
        if len(dims) == 1:
            if one_dim_orientation == 'y':
                y = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
                return (min_xy, y, min_z)
            else:
                z = proportional_scale(dims[0], relative_base_size, min_z, channel_cap)
                return (min_xy, min_xy, z)
        elif len(dims) == 2:
            x = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(dims[1], relative_base_size, min_xy, sequence_cap)
            # For 2D layers, use the second dimension for z-scaling as well
            z = proportional_scale(dims[1], relative_base_size, min_z, channel_cap)
            return (x, y, z)
        else:
            # 3D+ layers: handle spatial dimensions and channels separately
            x = proportional_scale(dims[0], relative_base_size, min_xy, sequence_cap)
            y = proportional_scale(dims[1], relative_base_size, min_xy, sequence_cap)
            # For channels (typically dims[2:]), use product for z-dimension
            channel_product = self_multiply(dims[2:]) if len(dims) > 2 else 1
            z = proportional_scale(channel_product, relative_base_size, min_z, channel_cap)
            return (x, y, z)
    else:
        warnings.warn(
            f"Unknown sizing mode '{sizing_mode}'. Defaulting to accurate.",
            UserWarning,
            stacklevel=3
        )

        # Recursive call to handle accurate sizing
        return calculate_layer_dimensions(
            shape, scale_z, scale_xy, max_z, max_xy, min_z, min_xy,
            one_dim_orientation=one_dim_orientation, sizing_mode='accurate',
            dimension_caps=dimension_caps, relative_base_size=relative_base_size
        )

    # Fallback
    return (min_xy, min_xy, min_z)
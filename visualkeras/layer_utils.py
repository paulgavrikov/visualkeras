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
        class Layer:  # type: ignore[no-redef]
            def __init__(self, *args, **kwargs):
                super().__init__()


class SpacingDummyLayer(Layer):
    """Placeholder layer used only to create visual gaps in a diagram.

    This layer has no semantic effect on the underlying model architecture. It
    exists so that renderers can detect intentional spacing between model
    sections without requiring a separate external layout description.
    """

    def __init__(self, spacing: int = 50):
        super().__init__()
        self.spacing = spacing


def get_layers(model):
    """Return the sequence of layers tracked by a Keras model.

    This helper normalizes the internal layer container differences between
    older and newer Keras or TensorFlow builds. It is used throughout the
    renderers whenever they need a stable ordered list of layers.

    Parameters
    ----------
    model : Any
        Keras or TensorFlow model instance.

    Returns
    -------
    list
        Sequence of layers tracked by the model.

    Raises
    ------
    RuntimeError
        If the model does not expose a supported internal layer container.
    """
    if hasattr(model, '_layers'):
        return model._layers
    if hasattr(model, '_self_tracked_trackables'):
        return model._self_tracked_trackables
    raise RuntimeError('Model does not expose _layers or _self_tracked_trackables')


def get_incoming_layers(layer):
    """Yield the layers that feed directly into ``layer``.

    This helper abstracts over the legacy and modern Keras node APIs so the
    rest of the package can treat inbound edges uniformly.

    Parameters
    ----------
    layer : Any
        Keras or TensorFlow layer instance whose parents should be discovered.

    Yields
    ------
    Any
        Layer objects that connect directly into ``layer``.
    """
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
    """Yield the layers that receive output directly from ``layer``.

    Parameters
    ----------
    layer : Any
        Keras or TensorFlow layer instance whose children should be discovered.

    Yields
    ------
    Any
        Layer objects that consume the output of ``layer``.
    """
    for i, node in enumerate(layer._outbound_nodes):
        yield node.outbound_layer


def model_to_adj_matrix(model):
    """Build an adjacency matrix describing model connectivity.

    The returned matrix records directed layer-to-layer edges using the model's
    internal graph representation. This is a foundational helper for renderers
    that need to reason about graph structure, hierarchy levels, or terminal
    nodes.

    Parameters
    ----------
    model : Any
        Keras or TensorFlow model instance.

    Returns
    -------
    tuple
        Two-item tuple ``(id_to_num_mapping, adj_matrix)`` where
        ``id_to_num_mapping`` maps ``id(layer)`` to a row or column index and
        ``adj_matrix`` is a square NumPy array counting directed edges.
    """
    if hasattr(model, 'built'):
        if not model.built:
            model.build()
            
    layers = get_layers(model)

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
    """Return a model layer by its Python object id.

    Parameters
    ----------
    model : Any
        Keras or TensorFlow model instance to search.
    _id : int
        Result of calling ``id(layer)`` for the target layer.

    Returns
    -------
    Any or None
        Matching layer instance, or ``None`` if no layer has the requested id.
    """
    for layer in get_layers(model):  # manually because get_layer may not access model._layers
        if id(layer) == _id:
            return layer
    return None


def find_layer_by_name(model, name):
    """Return a model layer by its ``name`` attribute.

    Parameters
    ----------
    model : Any
        Keras or TensorFlow model instance to search.
    name : str
        Layer name to match.

    Returns
    -------
    Any or None
        Matching layer instance, or ``None`` if no layer has the requested
        name.
    """
    for layer in get_layers(model):
        if layer.name == name:
            return layer
    return None


def find_input_layers(model, id_to_num_mapping=None, adj_matrix=None):
    """Yield graph input layers for a model.

    Inputs are identified as layers with zero in-degree in the model adjacency
    matrix. Precomputed graph structures may be supplied when this helper is
    used inside a larger connectivity pipeline.

    Parameters
    ----------
    model : Any
        Model whose input layers should be discovered.
    id_to_num_mapping : dict, optional
        Precomputed mapping from ``id(layer)`` to adjacency-matrix index.
    adj_matrix : numpy.ndarray, optional
        Precomputed adjacency matrix.

    Yields
    ------
    Any
        Layers that behave as graph inputs.
    """
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    for i in np.where(np.sum(adj_matrix, axis=0) == 0)[0]:  # find all nodes with 0 inputs
        for key in get_keys_by_value(id_to_num_mapping, i):
            yield find_layer_by_id(model, key)


def find_output_layers(model):
    """Yield graph output layers for a model.

    This helper supports both older and newer Keras APIs by using
    ``model.output_names`` when available and falling back to output tensor
    provenance when necessary.

    Parameters
    ----------
    model : Any
        Model whose output-producing layers should be discovered.

    Yields
    ------
    Any
        Layers that produce model outputs.
    """
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
    """Group model layers into topological hierarchy levels.

    Starting from graph inputs, this helper collects each successive wave of
    layers whose inbound dependencies have already been satisfied. The result is
    useful for renderers that organize diagrams by rank or stage.

    Parameters
    ----------
    model : Any
        Model whose layers should be grouped.
    id_to_num_mapping : dict, optional
        Precomputed mapping from ``id(layer)`` to adjacency-matrix index.
    adj_matrix : numpy.ndarray, optional
        Precomputed adjacency matrix.

    Returns
    -------
    list of list
        Layers grouped by hierarchy level from inputs to outputs.
    """
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
    """Append synthetic output layers to an existing adjacency graph.

    Some renderers benefit from explicit terminal nodes even when the original
    model ends on real layers. This helper extends both the adjacency matrix and
    the id mapping so those synthetic sinks can be treated like ordinary nodes.

    Parameters
    ----------
    model : Any
        Source model whose output layers should be connected to synthetic sinks.
    output_layers : sequence
        Synthetic output-layer objects to append.
    id_to_num_mapping : dict
        Existing mapping from ``id(layer)`` to adjacency-matrix index.
    adj_matrix : numpy.ndarray
        Existing adjacency matrix to extend.

    Returns
    -------
    tuple
        Updated ``(id_to_num_mapping, adj_matrix)`` pair.
    """

    adj_matrix = np.pad(adj_matrix, ((0, len(output_layers)), (0, len(output_layers))), mode='constant', constant_values=0)

    for dummy_output in output_layers:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(output_layers[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    """Return whether ``layer`` should be treated as an internal input layer.

    Keras and TensorFlow have exposed input-layer classes from several import
    paths over time. This helper centralizes the compatibility checks so the
    renderers can treat all of them consistently.

    Parameters
    ----------
    layer : Any
        Layer instance to classify.

    Returns
    -------
    bool
        ``True`` when the layer behaves like an internal input node.
    """
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
    """Return the primary output shape for visualization.

    Some layers expose multiple output shapes, but most renderers need a single
    representative tensor shape. This helper selects the first output as the
    primary one and emits a warning when information is being discarded.

    Parameters
    ----------
    layer_output_shape : tuple or list
        Output shape reported by a Keras layer. This may be a single shape, a
        tuple of shapes, or a list of shapes.
    layer_name : str, optional
        Layer name used to provide more informative warnings.

    Returns
    -------
    tuple
        Single shape tuple to use for visualization.

    Raises
    ------
    RuntimeError
        If the shape uses an unsupported container format.
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
    """Calculate rendered box dimensions for a layer shape.

    This helper converts a tensor shape into the pixel dimensions used by
    layered and functional renderers. The exact conversion depends on the
    selected sizing mode and on the scaling, minimum, and maximum bounds that
    accompany it.

    Parameters
    ----------
    shape : tuple
        Tensor shape to convert. The batch dimension is ignored when present.
    scale_z : float
        Multiplier applied to depth-like dimensions before clamping.
    scale_xy : float
        Multiplier applied to width and height dimensions before clamping.
    max_z : int
        Upper bound for the rendered depth dimension.
    max_xy : int
        Upper bound for rendered width and height.
    min_z : int
        Lower bound for the rendered depth dimension.
    min_xy : int
        Lower bound for rendered width and height.
    one_dim_orientation : {'y', 'z'}, default='y'
        Axis used when rendering one-dimensional layers.
    sizing_mode : {'accurate', 'capped', 'balanced', 'logarithmic', 'relative'}, default='accurate'
        Strategy used to transform tensor dimensions into pixels.
    dimension_caps : mapping, optional
        Optional per-dimension caps used by sizing modes that support them.
    relative_base_size : int, default=20
        Base pixel size used by ``relative`` mode.

    Returns
    -------
    tuple of int
        Three-item tuple ``(x, y, z)`` describing the rendered dimensions.

    Notes
    -----
    See the layer-utils API page for a fuller discussion of the sizing modes
    and their tradeoffs.

    Full documentation:
    https://visualkeras.readthedocs.io/en/latest/api/layer_utils_details.html
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
            return (int(x), int(y), int(z))
    
    # Logarithmic mode
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

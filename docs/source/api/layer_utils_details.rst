===========
Layer Utils
===========

Utility functions and classes for working with layers.

API Reference
=============

.. autoclass:: visualkeras.layer_utils.SpacingDummyLayer

.. autofunction:: visualkeras.layer_utils.get_layers

.. autofunction:: visualkeras.layer_utils.get_incoming_layers

.. autofunction:: visualkeras.layer_utils.get_outgoing_layers

.. autofunction:: visualkeras.layer_utils.model_to_adj_matrix

.. autofunction:: visualkeras.layer_utils.find_layer_by_id

.. autofunction:: visualkeras.layer_utils.find_layer_by_name

.. autofunction:: visualkeras.layer_utils.find_input_layers

.. autofunction:: visualkeras.layer_utils.find_output_layers

.. autofunction:: visualkeras.layer_utils.model_to_hierarchy_lists

.. autofunction:: visualkeras.layer_utils.augment_output_layers

.. autofunction:: visualkeras.layer_utils.is_internal_input

.. autofunction:: visualkeras.layer_utils.extract_primary_shape

.. autofunction:: visualkeras.layer_utils.calculate_layer_dimensions

SpacingDummyLayer
=================

Special layer for adding visual spacing in sequential models.

Purpose
-------

``SpacingDummyLayer`` is a no-op layer that serves purely for visualization purposes. It creates visual breaks between sections of your model, helping organize the diagram.

Basic Usage
-----------

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Conv2D(32, (3, 3)),
        visualkeras.SpacingDummyLayer(),  # Visual separator

        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.Conv2D(64, (3, 3)),
    ])

When visualizing, typically ignore these layers to keep the diagram clean:

.. code-block:: python

    visualkeras.layered_view(
        model,
        type_ignore=[visualkeras.SpacingDummyLayer]
    ).show()

Custom Spacing
--------------

Control the size of the spacing:

.. code-block:: python

    visualkeras.SpacingDummyLayer(spacing=100)  # Custom spacing size

The ``spacing`` parameter controls how much space is added. Default is 50 pixels.

When to Use
-----------

Use ``SpacingDummyLayer`` to:

- Separate feature extraction from classification sections
- Group convolutional blocks visually
- Organize encoder and decoder sections
- Divide architectural stages

It has no effect on model training or inference -- it's purely for visualization.

Why Useful
----------

In large complex models, visual organization helps readers understand the architecture at a glance. Compare:

.. code-block:: python

    # Without spacing - hard to see structure
    model = Sequential([
        Conv2D(...),  # Block 1
        Conv2D(...),
        MaxPooling2D(...),
        Conv2D(...),  # Block 2
        Conv2D(...),
        MaxPooling2D(...),
        Flatten(),
        Dense(...),   # Classification
        Dense(...)
    ])

    # With spacing - clear structure
    model = Sequential([
        Conv2D(...),
        Conv2D(...),
        MaxPooling2D(...),
        SpacingDummyLayer(),  # Break between blocks

        Conv2D(...),
        Conv2D(...),
        MaxPooling2D(...),
        SpacingDummyLayer(),  # Break before classification

        Flatten(),
        Dense(...),
        Dense(...)
    ])

Real-World Example
------------------

VGG-style network with clear section breaks:

.. code-block:: python

    model = keras.Sequential([
        # Block 1
        Conv2D(64, (3, 3), padding='same'),
        Conv2D(64, (3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        SpacingDummyLayer(),

        # Block 2
        Conv2D(128, (3, 3), padding='same'),
        Conv2D(128, (3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        SpacingDummyLayer(),

        # Block 3
        Conv2D(256, (3, 3), padding='same'),
        Conv2D(256, (3, 3), padding='same'),
        Conv2D(256, (3, 3), padding='same'),
        MaxPooling2D((2, 2)),
        SpacingDummyLayer(),

        # Classification
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])

    visualkeras.layered_view(
        model,
        type_ignore=[SpacingDummyLayer]
    ).show()

See Also
--------

- :doc:`../examples/sequential_models` for LeNet with spacing examples
- :doc:`../examples/advanced_features` for filtering and visualization techniques

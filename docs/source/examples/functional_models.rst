==================
Functional Models
==================

Complex models with multiple paths and connections.

Multi-Input Model
=================

A model with multiple input branches:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    # Define inputs
    input1 = keras.Input(shape=(32, 32, 3), name='image')
    input2 = keras.Input(shape=(10,), name='features')

    # Branch 1: Process images
    x1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input1)
    x1 = keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = keras.layers.Flatten()(x1)

    # Branch 2: Process features
    x2 = keras.layers.Dense(64, activation='relu')(input2)

    # Merge
    merged = keras.layers.Concatenate()([x1, x2])
    outputs = keras.layers.Dense(10, activation='softmax')(merged)

    model = keras.Model(inputs=[input1, input2], outputs=outputs)
    visualkeras.graph_view(model).show()

Residual Connection Model
==========================

A model with skip/residual connections:

.. code-block:: python

    inputs = keras.Input(shape=(224, 224, 3))

    x = keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Residual block
    residual = x
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.Add()([x, residual])
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    visualkeras.graph_view(model).show()

Multi-Output Model
==================

A model that produces multiple outputs:

.. code-block:: python

    inputs = keras.Input(shape=(224, 224, 3))

    x = keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Classification output
    classification = keras.layers.Dense(10, activation='softmax', name='classification')(x)

    # Regression output
    regression = keras.layers.Dense(1, name='regression')(x)

    model = keras.Model(inputs=inputs, outputs=[classification, regression])
    visualkeras.graph_view(model).show()

See Also
========

- :doc:`cnn_models` for CNN examples
- :doc:`sequential_models` for simple models
- :doc:`../tutorials/index` for tutorials

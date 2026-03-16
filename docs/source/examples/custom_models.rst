==============
Custom Models
==============

Advanced and specialized model architectures.

Inception-Style Module
======================

A model inspired by GoogLeNet's Inception module:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    inputs = keras.Input(shape=(224, 224, 3))

    # 1x1 convolution
    branch1 = keras.layers.Conv2D(96, (1, 1), activation='relu')(inputs)

    # 1x1 -> 3x3
    branch2 = keras.layers.Conv2D(96, (1, 1), activation='relu')(inputs)
    branch2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)

    # 1x1 -> 5x5
    branch3 = keras.layers.Conv2D(16, (1, 1), activation='relu')(inputs)
    branch3 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(branch3)

    # Max pooling
    branch4 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch4 = keras.layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate all branches
    output = keras.layers.Concatenate()([branch1, branch2, branch3, branch4])

    model = keras.Model(inputs=inputs, outputs=output)
    visualkeras.graph_view(model).show()

Dense Inception Stack
=====================

Multiple Inception-like modules stacked together:

.. code-block:: python

    def inception_module(x, name=None):
        """Simple Inception module"""
        branch1 = keras.layers.Conv2D(96, (1, 1), activation='relu')(x)
        branch2 = keras.layers.Conv2D(96, (1, 1), activation='relu')(x)
        branch2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)
        branch3 = keras.layers.Conv2D(16, (1, 1), activation='relu')(x)
        branch3 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(branch3)
        return keras.layers.Concatenate()([branch1, branch2, branch3])

    inputs = keras.Input(shape=(224, 224, 3))
    x = inception_module(inputs, name='inception_1')
    x = inception_module(x, name='inception_2')
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    visualkeras.graph_view(model).show()

Encoder-Decoder Architecture
=============================

An encoder-decoder model for tasks like semantic segmentation:

.. code-block:: python

    inputs = keras.Input(shape=(256, 256, 3))

    # Encoder
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Decoder
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Output
    outputs = keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    visualkeras.graph_view(model).show()

See Also
========

- :doc:`functional_models` for other complex models
- :doc:`../tutorials/index` for tutorials

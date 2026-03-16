==================
Sequential Models
==================

Simple, layer-by-layer model examples.

LeNet
=====

A classic LeNet architecture for digit recognition. This is the foundational CNN architecture from the 1990s:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(6, (5, 5), activation='tanh'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='tanh'),
        keras.layers.Dense(84, activation='tanh'),
        keras.layers.Dense(10, activation='softmax')
    ])

    visualkeras.layered_view(model).show()

LeNet with Visual Spacing
==========================

Add ``SpacingDummyLayer`` to create visual breaks between architectural components:

.. code-block:: python

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        # Feature extraction blocks
        keras.layers.Conv2D(6, (5, 5), activation='tanh'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        keras.layers.MaxPooling2D((2, 2)),
        visualkeras.SpacingDummyLayer(),  # Visual break

        # Classification section
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='tanh'),
        keras.layers.Dense(84, activation='tanh'),
        keras.layers.Dense(10, activation='softmax')
    ])

    visualkeras.layered_view(
        model,
        type_ignore=[visualkeras.SpacingDummyLayer]
    ).show()

Autoencoder
===========

A simple autoencoder for unsupervised learning:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        # Encoder
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        # Bottleneck
        keras.layers.Dense(32, activation='relu'),
        # Decoder
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(784, activation='sigmoid')
    ])

    visualkeras.layered_view(model).show()

Dense Network
=============

A simple fully-connected dense network for classification:

.. code-block:: python

    model = keras.Sequential([
        keras.layers.Input(shape=(30,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    visualkeras.layered_view(model).show()

See Also
========

- :doc:`cnn_models` for convolutional architectures
- :doc:`functional_models` for complex multi-path models
- :doc:`lenet_view` for LeNet-style feature map visualizations
- :doc:`../tutorials/index` for tutorials

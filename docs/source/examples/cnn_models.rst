===========
CNN Models
===========

Convolutional Neural Network examples - visualizations of popular CNN architectures.

VGG-Style Architecture
======================

A classic VGG-style convolutional architecture:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(1000, activation='softmax')
    ])

    visualkeras.layered_view(model).show()

Custom CNN
==========

A simple custom CNN for image classification:

.. code-block:: python

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
    ])

    visualkeras.layered_view(model).show()

See Also
========

- :doc:`sequential_models` for simple layer-by-layer models
- :doc:`functional_models` for complex multi-path architectures
- :doc:`../tutorials/index` for detailed tutorials

====================================
Tutorial 1: Basic Visualization
====================================

*Estimated time: 15 minutes*

In this tutorial, you'll learn the basics of visualizing neural network architectures using visualkeras.

Overview
========

By the end of this tutorial, you'll be able to:

- Understand the difference between layered view and graph view
- Create your first visualization
- Know when to use each visualization style
- Customize basic appearance

Prerequisites
=============

- visualkeras installed (see :doc:`../installation`)
- Basic knowledge of Keras/TensorFlow models
- Jupyter notebook or Python IDE

Visualization Styles
====================

visualkeras provides two main visualization approaches:

**Layered View**
    - Shows layers stacked on top of each other
    - Great for Convolutional Neural Networks (CNNs)
    - Shows tensor dimensions flowing through layers
    - Can become cluttered with very deep networks

**Graph View**
    - Shows model as a computational graph
    - Works with any model type (Sequential, Functional, Subclassed)
    - Shows connections between layers clearly
    - Best for understanding complex architectures

Creating Your First Visualization
==================================

Let's start with a simple Sequential model:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    # Define a simple CNN
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Display the model
    image = visualkeras.layered_view(model)
    image.show()

This creates a visual representation showing each layer and how data flows through the network.

When to Use Each Style
======================

Use **Layered View** when:
    - Your model is a CNN or image processing network
    - You want to emphasize layer stacking
    - The model has relatively few layers (< 30)
    - You're presenting to people unfamiliar with neural networks

Use **Graph View** when:
    - Your model is complex with multiple paths
    - You have skip connections or parallel layers
    - You want a compact representation
    - You're visualizing a Functional or Subclassed model

Quick Comparison
================

.. code-block:: python

    import visualkeras

    # Layered view (usually better for CNNs)
    visualkeras.layered_view(model).show()

    # Graph view (usually better for complex architectures)
    visualkeras.graph_view(model).show()

Common Issues
=============

**Blank visualization**
    Make sure your model has an Input layer:

    .. code-block:: python

        # With Input layer (works)
        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, (3, 3)),
        ])

**Tensor shapes not showing**
    Build the model explicitly:

    .. code-block:: python

        model.build((None, 28, 28, 1))
        visualkeras.layered_view(model).show()

Next Steps
==========

Now you understand the basics! Move on to:

- :doc:`tutorial_02_styling_customization` - Learn how to customize colors and appearance
- :doc:`tutorial_03_advanced_usage` - Explore advanced techniques
- :doc:`../quickstart` - Quick reference guide
- :doc:`../examples/index` - See example galleries

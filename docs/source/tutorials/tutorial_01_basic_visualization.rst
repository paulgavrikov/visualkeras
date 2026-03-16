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

Visualization Modes
===================

visualkeras provides **four visualization modes**, all accessed through the unified ``show()`` API:

**mode='layered'** — Stacked layers
    - Shows layers as 3D blocks stacked vertically
    - Great for CNNs and image processing networks
    - Emphasizes tensor dimensions flowing through layers
    - Best for understanding layer-by-layer transformations

**mode='graph'** — Computational graph
    - Shows model as a connected graph
    - Works with any model type (Sequential, Functional, Subclassed)
    - Clear visualization of connections and branching
    - Best for complex models with multiple paths

**mode='functional'** — Graph-aware layering
    - Combines best of both: layered appearance with graph awareness
    - Shows connections while maintaining layer structure
    - Ideal for CNNs with skip connections or multi-branch architectures
    - Supports collapse rules to merge repeated layer sequences

**mode='lenet'** — Feature map stack
    - Classic "feature map stacks" visualization style
    - Shows channels as individual map visualizations
    - Perfect for publishing CNN architectures in papers
    - Ideal for detailed architectural diagrams

Creating Your First Visualization
==================================

Let's start with a simple CNN and see all four modes:

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

    # The unified show() API with mode parameter
    # Try different modes to see what works best for your use case
    
    # Layered view
    image = visualkeras.show(model, mode='layered')
    image.show()

The unified ``show()`` API makes it easy to try different visualization styles without changing function calls.

Comparing All Four Modes
========================

Here's how to visualize your model in all four modes:

.. code-block:: python

    import visualkeras

    # Try each visualization mode
    layered = visualkeras.show(model, mode='layered')
    graph = visualkeras.show(model, mode='graph')
    functional = visualkeras.show(model, mode='functional')
    lenet = visualkeras.show(model, mode='lenet')

Each mode reveals different aspects of your architecture. Experiment to find what works best for your presentation or publication.

When to Use Each Mode
=====================

**Use Layered** when:
    - Visualizing CNNs or image processing networks
    - Want to emphasize layer stacking and tensor shapes
    - Have relatively simple sequential architectures
    - Presenting to people new to neural networks

**Use Graph** when:
    - Working with complex models with many branches
    - Have skip connections or parallel layers
    - Want the most compact representation
    - Visualizing Functional or Subclassed models

**Use Functional** when:
    - Have a CNN with non-sequential elements (skip connections)
    - Want layered appearance but with graph connections shown
    - Need something between pure layered and pure graph
    - Working with complex feature extraction pipelines

**Use LeNet** when:
    - Publishing architectures in academic papers
    - Want classic "feature maps on paper" style
    - Need detailed per-channel visualization
    - Designing presentation slides with detailed diagrams

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

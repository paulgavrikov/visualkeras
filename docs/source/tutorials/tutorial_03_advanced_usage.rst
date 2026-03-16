=====================================
Tutorial 3: Advanced Usage
=====================================

*Estimated time: 25 minutes*

Explore advanced techniques and patterns for visualizing complex models.

Overview
========

In this tutorial, you'll learn:

- Visualizing complex multi-path architectures
- Working with Functional models
- Creating publication-ready figures
- Best practices and tips

Complex Model Architectures
============================

visualkeras works great for visualizing complex models:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    # Create a more complex Functional model
    inputs = keras.Input(shape=(224, 224, 3))

    # Branch 1: Large convolutions
    x1 = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x1 = keras.layers.MaxPooling2D((2, 2))(x1)

    # Branch 2: Small convolutions
    x2 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)

    # Merge branches
    merged = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(merged)
    outputs = keras.layers.Dense(10, activation='softmax')(keras.layers.Flatten()(x))

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Visualize with graph view for complex architectures
    visualkeras.graph_view(model).show()

Graph View for Complex Models
==============================

The graph view is ideal for multi-input, multi-output, and residual models:

.. code-block:: python

    # Works great for Functional models with complex connections
    visualkeras.graph_view(model).show()

    # Can also work with layered view for simpler visualization
    visualkeras.layered_view(model).show()

Publication-Ready Figures
=========================

Create figures suitable for papers and presentations:

.. code-block:: python

    from visualkeras import options

    # High-quality output
    image = visualkeras.layered_view(
        model,
        color_map={
            keras.layers.Conv2D: '#1f77b4',
            keras.layers.MaxPooling2D: '#2ca02c',
            keras.layers.Dense: '#d62728',
        },
        scale_xy=40,
        scale_z=40
    )

    # Save at high resolution
    image.save('model_architecture_highres.png')

Best Practices
==============

**For CNNs, prefer layered view**:

.. code-block:: python

    # Layered view is intuitive for CNNs
    visualkeras.layered_view(model).show()

**For complex models, use graph view**:

.. code-block:: python

    # Graph view handles complexity better
    visualkeras.graph_view(model).show()

**Adjust scale for clarity**:

.. code-block:: python

    # If too crowded, increase scale
    visualkeras.layered_view(
        model,
        scale_xy=20  # Reduce size
    ).show()

    # If too small, decrease scale
    visualkeras.layered_view(
        model,
        scale_xy=50  # Increase size
    ).show()

Tips & Tricks
=============

**Visualize model stages**:

.. code-block:: python

    # Visualize just the feature extraction part
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # Before final Dense layers
    )
    visualkeras.layered_view(feature_extractor).show()

**Compare architectures**:

.. code-block:: python

    # Create multiple visualizations for comparison
    model1 = # ... your model 1
    model2 = # ... your model 2

    img1 = visualkeras.layered_view(model1)
    img2 = visualkeras.layered_view(model2)

    img1.show()  # View first
    img2.show()  # View second

**Batch processing multiple models**:

.. code-block:: python

    models = [model1, model2, model3]

    for i, model in enumerate(models):
        image = visualkeras.layered_view(model)
        image.save(f'model_{i}.png')

Next Steps
==========

You've mastered the fundamentals! Now:

- Explore the :doc:`../examples/index` for real-world use cases
- Check the :doc:`../api/index` for detailed API reference
- See the :doc:`../installation` for troubleshooting

Have a suggestion or found a bug? Visit the `GitHub repository <https://github.com/paulgavrikov/visualkeras>`_.

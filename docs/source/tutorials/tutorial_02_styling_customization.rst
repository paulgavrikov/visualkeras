==========================================
Tutorial 2: Styling & Customization
==========================================

*Estimated time: 20 minutes*

Learn how to customize the appearance of your visualizations.

Overview
========

In this tutorial, you'll learn how to:

- Customize layer colors
- Change font sizes and styles
- Adjust zoom and scaling
- Add custom layer configurations
- Export visualizations in different formats

Color Customization
===================

The easiest way to customize appearances is through the ``color_map`` parameter:

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dense(10),
    ])

    # Custom colors for different layer types
    color_map = {
        keras.layers.Conv2D: 'blue',
        keras.layers.MaxPooling2D: 'green',
        keras.layers.Dense: 'red',
        keras.layers.Flatten: 'yellow',
    }

    visualkeras.layered_view(model, color_map=color_map).show()

Styling Options
===============

The ``visualkeras.options`` module provides styling configurations:

.. code-block:: python

    from visualkeras import options

    # Get default styling options
    style = options.layered_style_options()

    # Modify styling
    visualkeras.layered_view(
        model,
        color_map=color_map,
        one_dim_orientation='x'  # Orientation for 1D layers
    ).show()

Scale and Zoom
==============

Control the size and zoom of your visualization:

.. code-block:: python

    # Smaller visualization
    visualkeras.layered_view(
        model,
        scale_xy=15,  # Default is usually 30
        scale_z=15
    ).show()

    # Larger visualization
    visualkeras.layered_view(
        model,
        scale_xy=50,
        scale_z=50
    ).show()

Exporting Visualizations
=========================

Save your visualizations to files:

.. code-block:: python

    # Create visualization
    image = visualkeras.layered_view(model, color_map=color_map)

    # Save as PNG
    image.save('my_model.png')

    # Save as other formats
    image.save('my_model.jpg')  # JPEG
    image.save('my_model.bmp')  # BMP

Advanced Customization
======================

For more control, explore the ``options`` module:

.. code-block:: python

    from visualkeras.options import Linestyle, Activation
    import visualkeras

    # Create custom options
    visualkeras.layered_view(
        model,
        color_map=color_map,
        one_dim_orientation='x',
        draw_volume=True
    ).show()

Common Patterns
===============

**Presentation slides colors** (light background):

.. code-block:: python

    light_colors = {
        keras.layers.Conv2D: '#1f77b4',
        keras.layers.MaxPooling2D: '#2ca02c',
        keras.layers.Dense: '#d62728',
    }
    visualkeras.layered_view(model, color_map=light_colors).show()

**Publication quality** (grayscale):

.. code-block:: python

    bw_colors = {
        keras.layers.Conv2D: '#555555',
        keras.layers.MaxPooling2D: '#888888',
        keras.layers.Dense: '#cccccc',
    }
    visualkeras.layered_view(model, color_map=bw_colors).show()

Next Steps
==========

Ready to dive deeper?

- :doc:`tutorial_03_advanced_usage` - Advanced techniques
- :doc:`../examples/index` - Example gallery
- :doc:`../api/index` - Complete API reference

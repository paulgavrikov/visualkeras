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

Customize layer appearance using the unified ``show()`` API with proper color dictionaries.

Color values must specify both ``fill`` and ``outline`` colors for each layer type:

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

    # Define colors with fill and outline for each layer type
    color_map = {
        keras.layers.Conv2D: {'fill': '#3498db', 'outline': '#2980b9'},
        keras.layers.MaxPooling2D: {'fill': '#2ecc71', 'outline': '#27ae60'},
        keras.layers.Dense: {'fill': '#e74c3c', 'outline': '#c0392b'},
        keras.layers.Flatten: {'fill': '#f39c12', 'outline': '#d68910'},
    }

    # Use show() with mode parameter for clean, unified API
    image = visualkeras.show(model, mode='layered', color_map=color_map)
    image.show()

Using Options Objects
======================

For more control, bundle related settings with ``LayeredOptions``:

.. code-block:: python

    from visualkeras.options import LayeredOptions, LAYERED_PRESETS
    import visualkeras

    # Use preset as base, then customize
    my_options = LayeredOptions(
        **LAYERED_PRESETS['presentation'].__dict__,  # Start with presentation preset
        one_dim_orientation='x',  # Customize: orientation for 1D layers
    )

    # Apply to model
    image = visualkeras.show(model, mode='layered', options=my_options, color_map=color_map)
    image.show()

Scale and Zoom
==============

Control the size and zoom using ``scale_xy`` and ``scale_z`` parameters:

.. code-block:: python

    # Compact visualization
    small_image = visualkeras.show(model, mode='layered', scale_xy=3.0, scale_z=1.0)
    small_image.show()

    # Large, detailed visualization
    large_image = visualkeras.show(model, mode='layered', scale_xy=6.0, scale_z=2.0)
    large_image.show()

    # Or use presets for curated scaling
    compact_image = visualkeras.show(model, mode='layered', preset='compact')
    presentation_image = visualkeras.show(model, mode='layered', preset='presentation')

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

For fine-grained control, use ``LayeredOptions`` with detailed configuration:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    import visualkeras

    # Create fully custom options
    custom_options = LayeredOptions(
        color_map=color_map,
        one_dim_orientation='x',
        draw_volume=True,
        spacing=15,  # Gap between layers
        padding=20,  # Border padding
        legend=True,  # Show layer type legend
        show_dimension=True,  # Display tensor shapes
    )

    image = visualkeras.show(model, mode='layered', options=custom_options)
    image.show()

Common Patterns
===============

**Presentation slides** (high contrast colors):

.. code-block:: python

    presentation_colors = {
        keras.layers.Conv2D: {'fill': '#1f77b4', 'outline': '#0d47a1'},
        keras.layers.MaxPooling2D: {'fill': '#2ca02c', 'outline': '#1b5e20'},
        keras.layers.Dense: {'fill': '#d62728', 'outline': '#b71c1c'},
    }
    image = visualkeras.show(
        model,
        mode='layered',
        preset='presentation',  # Use curated preset
        color_map=presentation_colors
    )
    image.show()

**Publication quality** (grayscale with professional borders):

.. code-block:: python

    grayscale_colors = {
        keras.layers.Conv2D: {'fill': '#555555', 'outline': '#000000'},
        keras.layers.MaxPooling2D: {'fill': '#888888', 'outline': '#333333'},
        keras.layers.Dense: {'fill': '#cccccc', 'outline': '#666666'},
    }
    image = visualkeras.show(
        model,
        mode='layered',
        color_map=grayscale_colors,
        background_fill='white'
    )
    image.show()

Next Steps
==========

Ready to dive deeper?

- :doc:`tutorial_03_advanced_usage` - Advanced techniques
- :doc:`../examples/index` - Example gallery
- :doc:`../api/index` - Complete API reference

================================
Advanced Features and Techniques
================================

This page covers advanced customization options, special features, and techniques available across all visualization types.

Using the Unified Show Function
================================

The highest level API is the ``show()`` function, which lets you select a renderer by mode:

.. code-block:: python

    import visualkeras

    # Use layered view
    img = visualkeras.show(model, mode='layered')

    # Use graph view
    img = visualkeras.show(model, mode='graph')

    # Use functional view
    img = visualkeras.show(model, mode='functional')

    # Use lenet view
    img = visualkeras.show(model, mode='lenet')

Benefits of ``show()``
======================

The ``show()`` function provides:

- Single entry point for all renderers
- Support for presets across all types
- Options object support for cleaner code
- Unified parameter overrides

Using Options Objects
=====================

For cleaner, more maintainable code, use options objects instead of individual parameters:

.. code-block:: python

    from visualkeras.options import LayeredOptions, GraphOptions

    # Define your configuration once
    options = LayeredOptions(
        scale_xy=5.0,
        spacing=15,
        padding=20,
        legend=True,
        text_callable='name_shape'
    )

    # Reuse for multiple models
    img1 = visualkeras.layered_view(model1, options=options)
    img2 = visualkeras.layered_view(model2, options=options)

This is cleaner than passing many parameters repeatedly.

Presets for Quick Configuration
================================

All renderers include curated presets for common scenarios. Available for all renderer types:

- ``'default'`` - Balanced settings for general use
- ``'compact'`` - Minimal spacing, tight layout
- ``'presentation'`` - Large, detailed output for talks and papers

.. code-block:: python

    # Quick preset use
    img = visualkeras.show(model, mode='layered', preset='presentation')

Filtering Layers
================

Control which layers appear in your visualization.

By Type
-------

Ignore specific layer types:

.. code-block:: python

    from tensorflow.keras.layers import Dropout, BatchNormalization

    visualkeras.layered_view(
        model,
        type_ignore=[Dropout, BatchNormalization]
    ).show()

By Index
--------

Ignore layers by their position in the model:

.. code-block:: python

    visualkeras.layered_view(
        model,
        index_ignore=[3, 5, 8]  # Skip layers at positions 3, 5, 8
    ).show()

Adding Visual Spacing
=====================

Use ``SpacingDummyLayer`` to add visual breaks between layer groups:

.. code-block:: python

    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Conv2D(32, (3, 3)),
        visualkeras.SpacingDummyLayer(),  # Visual separator

        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.Conv2D(64, (3, 3)),
        visualkeras.SpacingDummyLayer(),  # Another separator

        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    visualkeras.layered_view(
        model,
        type_ignore=[visualkeras.SpacingDummyLayer]
    ).show()

When visualizing, the spacing layers appear as separators but you typically ignore them to keep the diagram clean.

Text Annotations
================

Add layer information below or above each layer using text callables.

Built-in Options
-----------------

Available text callables in layered view:

- ``'name'`` - Shows layer class name
- ``'type'`` - Shows full type information
- ``'shape'`` - Shows output tensor shape
- ``'name_shape'`` - Shows both name and shape

.. code-block:: python

    from visualkeras.options import LAYERED_TEXT_CALLABLES

    visualkeras.layered_view(
        model,
        text_callable=LAYERED_TEXT_CALLABLES['name_shape']
    ).show()

Or use the preset that includes annotations:

.. code-block:: python

    visualkeras.layered_view(model, preset='presentation').show()

Custom Text Callables
-----------------------

Define your own text function:

.. code-block:: python

    def my_text_callable(layer_index, layer):
        name = layer.__class__.__name__
        text = f"{name} (#{layer_index})"
        above = False  # Show below the layer
        return (text, above)

    visualkeras.layered_view(
        model,
        text_callable=my_text_callable
    ).show()

The function receives the layer index and layer object, and returns a tuple of (text, above).

Dimension Information
=====================

Show output dimensions in the legend:

.. code-block:: python

    visualkeras.layered_view(
        model,
        legend=True,
        show_dimension=True
    ).show()

This displays the output shape for each layer in the legend.

Sizing Strategies
=================

Control how layer sizes are calculated from dimensions.

Available Modes
---------------

- ``'accurate'`` - Raw dimensions with scaling (may be very large)
- ``'balanced'`` - Smart scaling balancing accuracy and clarity (recommended for modern models)
- ``'capped'`` - Cap dimensions at limits while preserving ratios
- ``'logarithmic'`` - Use logarithmic scaling for very large dimensions
- ``'relative'`` - Proportional scaling where size matches actual dimension ratios

.. code-block:: python

    visualkeras.layered_view(
        model,
        sizing_mode='balanced'
    ).show()

Relative Sizing in Detail
----------------------------

The ``'relative'`` mode maintains true proportional relationships:

.. code-block:: python

    visualkeras.layered_view(
        model,
        sizing_mode='relative',
        relative_base_size=10  # Each dimension unit = 10 pixels
    ).show()

With ``relative_base_size=10``:
- A layer with 64 units gets 64 × 10 = 640 pixels
- A layer with 32 units gets 32 × 10 = 320 pixels (exactly half)

This ensures consistent proportions across all layers.

2D and 3D Visualization
=======================

Layered view supports both 3D volumetric and 2D flat rendering.

3D Volumetric (Default)
------------------------

.. code-block:: python

    visualkeras.layered_view(
        model,
        draw_volume=True
    ).show()

2D Flat View
------------

For a simpler 2D appearance:

.. code-block:: python

    visualkeras.layered_view(
        model,
        draw_volume=False
    ).show()

Control 3D Direction
--------------------

For 3D views, reverse the drawing direction:

.. code-block:: python

    visualkeras.layered_view(
        model,
        draw_volume=True,
        draw_reversed=True  # Front-right to back-left instead of back-left to front-right
    ).show()

One Dimensional Layer Orientation
==================================

Control how 1D layers (Dense, Flatten) are rendered:

.. code-block:: python

    # Draw 1D layers along Z axis (tall)
    visualkeras.layered_view(model, one_dim_orientation='z').show()

    # Draw 1D layers along X axis (wide)
    visualkeras.layered_view(model, one_dim_orientation='x').show()

    # Draw 1D layers along Y axis
    visualkeras.layered_view(model, one_dim_orientation='y').show()

Advanced Color Mapping
======================

Map colors to specific layer types or names:

.. code-block:: python

    from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

    color_map = {
        Conv2D: {
            'fill': '#1f77b4',      # Layer fill color
            'outline': '#000000'    # Border color
        },
        Dense: {
            'fill': '#2ca02c'
        },
        MaxPooling2D: {
            'fill': '#ff7f0e'
        }
    }

    visualkeras.layered_view(model, color_map=color_map).show()

Saving to File
==============

All renderers can save directly to file:

.. code-block:: python

    # Automatically detect format from extension
    visualkeras.layered_view(
        model,
        to_file='my_model.png'
    ).show()

Supported formats include PNG, JPG, BMP, and others supported by Pillow.

Image Fitting Options
=====================

When visualizations include images, control how they fit:

.. code-block:: python

    visualkeras.layered_view(
        model,
        image_fit='contain'  # Fit within bounds
    ).show()

Available options depend on the renderer (consult API docs for details).

Legends and Annotations
=======================

Add a legend showing layer types and output shapes:

.. code-block:: python

    from PIL import ImageFont

    font = ImageFont.truetype("arial.ttf", 14)

    visualkeras.layered_view(
        model,
        legend=True,
        show_dimension=True,
        font=font
    ).show()

Tips and Tricks
===============

Comparing Renderers
--------------------

Quickly compare different visualization styles:

.. code-block:: python

    import visualkeras

    # Show all renderers side-by-side for comparison
    for mode in ['layered', 'graph', 'functional', 'lenet']:
        img = visualkeras.show(model, mode=mode)
        img.save(f'model_{mode}.png')

Creating Consistent Series
------------------------------

Use the same options for multiple models:

.. code-block:: python

    from visualkeras.options import LayeredOptions

    opts = LayeredOptions(
        spacing=12,
        padding=15,
        legend=True,
        text_callable='name_shape'
    )

    for model in [model1, model2, model3]:
        img = visualkeras.layered_view(model, options=opts)

Batch Processing
------------------

Process multiple models in a loop:

.. code-block:: python

    import visualkeras
    from pathlib import Path

    models = [model1, model2, model3]
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    for i, model in enumerate(models):
        img = visualkeras.show(model, preset='presentation')
        img.save(output_dir / f'model_{i}.png')

Performance Optimization
-------------------------

For large models:

1. Reduce scale factors to make rendering faster
2. Use ``sizing_mode='balanced'`` instead of 'accurate'
3. Ignore non-essential layer types
4. Try graph view for complex architectures

.. code-block:: python

    visualkeras.layered_view(
        huge_model,
        scale_xy=2.0,  # Smaller scale
        sizing_mode='balanced',
        type_ignore=[BatchNormalization, Dropout]
    ).show()

See Also
========

- :doc:`../tutorials/index` for step-by-step guides
- :doc:`../api/index` for complete API reference
- :doc:`lenet_view` for feature map stack diagrams

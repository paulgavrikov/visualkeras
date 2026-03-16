======
Layered
======

Layered view visualization for neural networks.

Overview
========

The layered renderer displays neural network architectures as stacked 3D layer blocks, making it ideal for understanding CNNs and other models where tensor shape changes are important.

**When to use:**
    - Visualizing Convolutional Neural Networks (CNNs)
    - Emphasizing tensor dimensions flowing through layers
    - Creating readable diagrams for relatively simple sequential models
    - Understanding layer-by-layer data transformations

**When NOT to use:**
    - Complex branching architectures (use graph or functional mode instead)
    - Very deep models (>100 layers) without layer collapse
    - Models where computational graph structure is the focus

API Reference
=============

.. automodule:: visualkeras.layered
   :members:
   :undoc-members:
   :show-inheritance:

Key Parameters
==============

**Core Parameters**

- ``model``: Keras model instance
- ``to_file``: Path to save output (optional)
- ``preset``: Use preset configuration ('default', 'compact', 'presentation')
- ``options``: LayeredOptions object for bundled configuration

**Layout Control**

- ``padding``: Border space (default: 10)
- ``spacing``: Gap between layers (default: 10)
- ``min_z``, ``max_z``: Min/max layer depth in pixels
- ``min_xy``, ``max_xy``: Min/max layer width/height in pixels
- ``scale_z``, ``scale_xy``: Scale multipliers (default: 1.5, 4.0)

**Styling**

- ``color_map``: Dict mapping layer types to ``{'fill': '...', 'outline': '...'}``
- ``background_fill``: Background color (default: 'white')
- ``font_color``: Text color (default: 'black')
- ``draw_volume``: Show 3D effect (default: True)
- ``draw_funnel``: Show tapering for shape changes (default: True)

**Content Control**

- ``text_callable``: Function or string key controlling layer labels
- ``show_dimension``: Display tensor shapes (default: False)
- ``type_ignore``: Layer types to skip
- ``index_ignore``: Layer indices to skip
- ``legend``: Show layer type legend (default: False)

**Advanced**

- ``layered_groups``: Visually group layers with backgrounds
- ``logos_legend``: Show logo/group legend
- ``styles``: Per-layer style overrides
- ``sizing_mode``: How to calculate layer sizes ('accurate', 'balanced', 'logarithmic', 'relative')

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

    import visualkeras
    
    image = visualkeras.show(model, mode='layered')
    image.show()

With Presets
------------

.. code-block:: python

    # Presentation quality
    image = visualkeras.show(model, mode='layered', preset='presentation')
    
    # Compact for slides
    image = visualkeras.show(model, mode='layered', preset='compact')

Custom Colors
--------------

.. code-block:: python

    image = visualkeras.show(
        model,
        mode='layered',
        color_map={
            keras.layers.Conv2D: {'fill': '#1976d2', 'outline': '#0d47a1'},
            keras.layers.Dense: {'fill': '#388e3c', 'outline': '#1b5e20'},
        }
    )

With Options
--------------

.. code-block:: python

    from visualkeras.options import LayeredOptions, LAYERED_TEXT_CALLABLES
    
    options = LayeredOptions(
        spacing=15,
        padding=20,
        legend=True,
        text_callable=LAYERED_TEXT_CALLABLES['name_shape'],
        show_dimension=True,
    )
    
    image = visualkeras.show(model, mode='layered', options=options)

See Also
========

- :doc:`../examples/cnn_models` for CNN examples
- :doc:`../examples/sequential_models` for simple model examples
- :doc:`../tutorials/tutorial_01_basic_visualization` for tutorial
- :doc:`../tutorials/tutorial_02_styling_customization` for styling guide
- :doc:`options` for LayeredOptions reference

=====
Graph
=====

Graph view visualization for neural networks.

Overview
========

The graph renderer displays neural network architectures as computational graphs, with nodes representing layers and edges showing data flow. This mode is ideal for understanding model topology and works with any Keras model type.

**When to use:**
    - Models with complex topology (branches, merges, skip connections)
    - Understanding computational structure is important
    - Functional or Subclassed models
    - Any model where connections between non-adjacent layers matter
    - Comparative topology analysis

**When NOT to use:**
    - Simple sequential models (layered, graph, or functional all work, but layered is simplest)
    - Models where tensor shape progression is important (use layered or functional)
    - Very wide models with many parallel paths (may be too dense to read)

API Reference
=============

.. autofunction:: visualkeras.graph.graph_view

Configuration Options
=====================

Use :py:class:`visualkeras.options.GraphOptions` when you want to bundle graph
layout, node styling, connector behavior, and related settings for reuse.
Curated presets are available through
:py:data:`visualkeras.options.GRAPH_PRESETS`.

Key Parameters
==============

**Core Parameters**

- ``model``: Keras model instance
- ``to_file``: Path to save output (optional)
- ``preset``: Use preset configuration ('default', 'compact', 'presentation')
- ``options``: GraphOptions object for bundled configuration

**Layout Control**

- ``layer_spacing``: Space between layers (default: 250)
- ``node_spacing``: Space between nodes (default: 10)
- ``padding``: Border space (default: 10)

**Styling**

- ``color_map``: Dict mapping layer types to ``{'fill': '...', 'outline': '...'}``
- ``connector_fill``: Connection line color (default: 'gray')
- ``connector_width``: Line thickness (default: 1)
- ``node_size``: Node size in pixels (default: 50)
- ``background_fill``: Background color (default: 'white')
- ``font_color``: Text color (default: 'black')

**Content Control**

- ``type_ignore``: Layer types to skip
- ``index_ignore``: Layer indices to skip

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

    import visualkeras
    
    image = visualkeras.show(model, mode='graph')
    image.show()

With Custom Node Size
---------------------

.. code-block:: python

    image = visualkeras.show(
        model,
        mode='graph',
        node_size=75,
        layer_spacing=300
    )

With Color Customization
------------------------

.. code-block:: python

    image = visualkeras.show(
        model,
        mode='graph',
        color_map={
            keras.layers.Conv2D: {'fill': '#1976d2', 'outline': '#0d47a1'},
            keras.layers.Dense: {'fill': '#388e3c', 'outline': '#1b5e20'},
        },
        preset='presentation'
    )

See Also
========

- :doc:`../examples/functional_models` for examples
- :doc:`../tutorials/tutorial_01_basic_visualization` for tutorial
- :py:class:`visualkeras.options.GraphOptions` for the full options API
- :doc:`options` for presets and shared options documentation

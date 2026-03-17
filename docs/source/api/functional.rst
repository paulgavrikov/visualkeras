==========
Functional
==========

Functional view visualization for neural networks.

Overview
========

The functional renderer combines the best of layered and graph visualization: it maintains layer-level structure while showing computational connections. This makes it ideal for Keras Functional models with skip connections, multi-branch architectures, and complex topologies.

**When to use:**
    - Keras Functional API models with skip connections
    - Multi-input or multi-output architectures
    - Models with parallel branches (e.g., Inception modules)
    - ResNet-style architectures with residual connections
    - Models where both layer structure and connections matter

**When NOT to use:**
    - Simple sequential models (use layered mode)
    - Models where pure computational graph is important (use graph mode)
    - Very wide parallel models (graph mode may be clearer)

API Reference
=============

.. autofunction:: visualkeras.functional.functional_view

Configuration Options
=====================

Use :py:class:`visualkeras.options.FunctionalOptions` when you want to bundle
functional layout, connector routing, collapse behavior, and related styling in
one reusable object. Curated presets are available through
:py:data:`visualkeras.options.FUNCTIONAL_PRESETS`.

Key Parameters
==============

**Core Parameters**

- ``model``: Keras Functional model instance
- ``to_file``: Path to save output (optional)
- ``preset``: Use preset configuration ('default', 'compact', 'presentation')
- ``options``: FunctionalOptions object for bundled configuration

**Layout Control**

- ``column_spacing``: Space between columns (default: 60)
- ``row_spacing``: Space between rows (default: 30)
- ``component_spacing``: Space between components (default: 100)
- ``padding``: Border space (default: 10)

**Styling**

- ``color_map``: Dict mapping layer types to ``{'fill': '...', 'outline': '...'}``
- ``connector_fill``: Connection line color (default: 'gray')
- ``connector_width``: Line thickness (default: 1)
- ``background_fill``: Background color (default: 'white')
- ``font_color``: Text color (default: 'black')

**Content Control**

- ``text_callable``: Function or string key controlling layer labels
- ``type_ignore``: Layer types to skip
- ``index_ignore``: Layer indices to skip

**Advanced**

- ``collapse_rules``: Automatically collapse repeated layer sequences
- ``sizing_mode``: How to calculate layer sizes ('accurate', 'balanced', 'logarithmic')
- ``styles``: Per-layer style overrides

Collapse Rules
===============

Automatically collapse sequences of repeated layers:

.. code-block:: python

    collapse_rules = [
        {
            'kind': 'layer',
            'selector': keras.layers.Conv2D,
            'repeat_count': 3,
            'label': '3x Conv2D',
            'annotation_position': 'above',
        }
    ]
    
    image = visualkeras.show(
        model,
        mode='functional',
        collapse_rules=collapse_rules
    )

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

    import visualkeras
    
    image = visualkeras.show(model, mode='functional')
    image.show()

With Presets
------------

.. code-block:: python

    # Presentation quality
    image = visualkeras.show(model, mode='functional', preset='presentation')
    
    # Compact for slides
    image = visualkeras.show(model, mode='functional', preset='compact')

With Collapsed Layers
---------------------

.. code-block:: python

    collapse_rules = [
        {
            'kind': 'block',
            'selector': [keras.layers.Conv2D, keras.layers.BatchNormalization],
            'repeat_count': 2,
            'label': 'Conv Block (2x)',
        }
    ]
    
    image = visualkeras.show(
        model,
        mode='functional',
        collapse_rules=collapse_rules,
        preset='presentation'
    )

See Also
========

- :doc:`../examples/functional_models` for functional model examples
- :doc:`../tutorials/tutorial_01_basic_visualization` for tutorial
- :py:class:`visualkeras.options.FunctionalOptions` for the full options API
- :doc:`options` for presets and shared options documentation

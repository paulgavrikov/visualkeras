=======
Layered
=======

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

.. autofunction:: visualkeras.layered.layered_view

Configuration Options
=====================

Use :py:class:`visualkeras.options.LayeredOptions` when you want to bundle a
layered configuration and reuse it across multiple renders. Curated presets are
available through :py:data:`visualkeras.options.LAYERED_PRESETS`.

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
- :py:class:`visualkeras.options.LayeredOptions` for the full options API
- :doc:`options` for presets and shared options documentation

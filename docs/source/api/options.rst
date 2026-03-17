=======
Options
=======

Configuration objects and presets for all renderers.

API Reference
=============

.. autoclass:: visualkeras.options.LayeredOptions
   :members: to_kwargs

.. autoclass:: visualkeras.options.GraphOptions
   :members: to_kwargs

.. autoclass:: visualkeras.options.FunctionalOptions
   :members: to_kwargs

.. autoclass:: visualkeras.options.LenetOptions
   :members: to_kwargs

.. autodata:: visualkeras.options.LAYERED_TEXT_CALLABLES

.. autodata:: visualkeras.options.LAYERED_PRESETS

.. autodata:: visualkeras.options.GRAPH_PRESETS

.. autodata:: visualkeras.options.FUNCTIONAL_PRESETS

.. autodata:: visualkeras.options.LENET_PRESETS

Using Options Objects
=====================

Instead of passing many individual parameters, you can use options objects for cleaner code:

.. code-block:: python

    from visualkeras.options import LayeredOptions, LAYERED_TEXT_CALLABLES

    # Define once
    my_options = LayeredOptions(
        spacing=15,
        padding=20,
        legend=True,
        text_callable=LAYERED_TEXT_CALLABLES['name_shape']
    )

    # Reuse for all models
    img = visualkeras.layered_view(model1, options=my_options)
    img = visualkeras.layered_view(model2, options=my_options)

Available Presets
=================

All renderers include three curated presets:

**'default'** - Balanced defaults for general use

**'compact'** - Minimal spacing for tight layouts

**'presentation'** - Large, detailed output for publications and talks

.. code-block:: python

    # Use a preset
    img = visualkeras.layered_view(model, preset='presentation')

    # Or with the unified show() function
    img = visualkeras.show(model, mode='layered', preset='presentation')

Available Options Classes
==========================

- :py:class:`~visualkeras.options.LayeredOptions` - Configuration for layered_view()
- :py:class:`~visualkeras.options.GraphOptions` - Configuration for graph_view()
- :py:class:`~visualkeras.options.FunctionalOptions` - Configuration for functional_view()
- :py:class:`~visualkeras.options.LenetOptions` - Configuration for lenet_view()

Text Callables
==============

Pre-defined text callable functions for annotating layers in layered view:

.. code-block:: python

    from visualkeras.options import LAYERED_TEXT_CALLABLES

    # Available callables:
    # 'name' - Layer class name
    # 'type' - Full type information
    # 'shape' - Output tensor shape
    # 'name_shape' - Layer name and shape

    visualkeras.layered_view(
        model,
        text_callable=LAYERED_TEXT_CALLABLES['name_shape']
    ).show()

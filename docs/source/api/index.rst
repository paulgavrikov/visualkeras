==============
API Reference
==============

Complete API documentation for all visualkeras modules.

Main Functions
==============

The main entry points for visualkeras:

- :py:func:`visualkeras.show` - Unified interface for all renderers (recommended for most use cases)
- :py:func:`visualkeras.layered_view` - Visualize models in layered style (great for CNNs)
- :py:func:`visualkeras.graph_view` - Visualize models as graphs (works for all types)
- :py:func:`visualkeras.functional_view` - Visualize functional models with graph-aware layering
- :py:func:`visualkeras.lenet_view` - Classic feature map stack visualization (LeNet style)

Utility Classes
===============

- :py:class:`visualkeras.SpacingDummyLayer` - Add visual spacing between layer groups

Module Documentation
=====================

.. toctree::
   :maxdepth: 2

   layered
   functional
   graph
   lenet_style
   layer_utils_details
   options
   advanced_styling_reference
   utils_details

Quick Reference
===============

**Unified Interface (Recommended)**

.. code-block:: python

    import visualkeras

    # Select renderer by mode
    img = visualkeras.show(model, mode='layered')

**Layered View**

.. code-block:: python

    visualkeras.layered_view(model).show()

**Graph View**

.. code-block:: python

    visualkeras.graph_view(model).show()

**Functional View**

.. code-block:: python

    visualkeras.functional_view(model).show()

**LeNet View**

.. code-block:: python

    visualkeras.lenet_view(model).show()

**With Options**

.. code-block:: python

    from visualkeras.options import LayeredOptions

    opts = LayeredOptions(spacing=15, legend=True)
    visualkeras.layered_view(model, options=opts).show()

Getting Help
============

- :doc:`../quickstart` - Getting started quickly
- :doc:`../tutorials/index` - Step-by-step guides
- :doc:`../examples/index` - Example gallery
- :doc:`../examples/advanced_features` - Advanced techniques


Quick Reference
===============

**Layered View** - Best for CNNs
   .. code-block:: python

      import visualkeras
      visualkeras.layered_view(model).show()

**Graph View** - Works for all models
   .. code-block:: python

      visualkeras.graph_view(model).show()

**Customization** - Modify appearance
   .. code-block:: python

      from visualkeras import options
      visualkeras.layered_view(
          model,
          color_map={...},
          **options.layered_style_options()
      ).show()

Getting Help
============

- 📖 **Learning?** Start with :doc:`../quickstart` or :doc:`../tutorials/index`
- 🖼️ **Examples?** Browse the :doc:`../examples/index`
- ❓ **Questions?** Check `GitHub Discussions <https://github.com/paulgavrikov/visualkeras/discussions>`_
- 🐛 **Issues?** Visit `GitHub Issues <https://github.com/paulgavrikov/visualkeras/issues>`_

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

.. automodule:: visualkeras.layered
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Reference
===================

This reference covers the keyword arguments accepted by ``visualkeras.layered_view`` and the fields exposed by
``LayeredOptions``. When you use multiple configuration mechanisms together, values are applied in this order

1. preset defaults
2. ``options`` values
3. explicit keyword arguments

Required Input
--------------

``model``
  The Keras model to render. Layered view is designed for sequential or effectively linear architectures where layer
  order is the main story. It can render some functional models, but graph or functional mode is usually clearer for
  branching topologies.

Output and Reuse
----------------

``to_file``
  Optional path to write the rendered image to disk. Pillow infers the output format from the file extension. The image
  object is still returned, so you can both save and keep working with the result.

``preset``
  Optional preset name. Layered mode currently ships with ``default``, ``compact``, and ``presentation``. Use a preset
  when you want a sensible starting point and then override only a few details.

``options``
  Optional ``LayeredOptions`` instance or a plain mapping. This is the cleanest way to bundle settings for reuse across
  multiple models or examples. Explicit keyword arguments override anything provided here.

Size and Layout
---------------

``min_z`` and ``max_z``
  Lower and upper bounds for the rendered depth of a layer box. These limits prevent extremely narrow or extremely deep
  channel dimensions from producing unreadable output.

``min_xy`` and ``max_xy``
  Lower and upper bounds for the rendered width and height of a layer box. These matter most when your model mixes very
  small tensors with very large ones.

``scale_z``
  Global multiplier applied to depth before clamping. Increase it when channel depth should read more strongly in the
  visualization.

``scale_xy``
  Global multiplier applied to width and height before clamping. Reduce it when the diagram grows too large. Increase
  it when layers feel too small or compressed.

``padding``
  Outer margin around the full image. Increase it when text, legends, or grouped overlays feel crowded near the edges.

``spacing``
  Horizontal gap between consecutive rendered layers. Use this for overall breathing room. If you need selective
  separation between stages, combine it with ``SpacingDummyLayer`` or ``layered_groups``.

``one_dim_orientation``
  Controls how one dimensional outputs such as dense or flattened layers are drawn. Accepted values are ``'x'``,
  ``'y'``, and ``'z'``. This is useful when vector layers would otherwise dominate one axis visually.

``index_2D``
  Sequence of layer indices to force into flat 2D rendering even when ``draw_volume`` is enabled. This is useful for
  selectively simplifying dense heads, summary blocks, or other layers that do not benefit from 3D depth.

Layer Inclusion and Annotation
------------------------------

``type_ignore``
  Sequence of layer classes to skip entirely. This is the simplest way to remove utility layers such as dropout or
  padding layers from the diagram without changing the model itself.

``index_ignore``
  Sequence of layer positions to skip. Use this when you want precise control over individual layers rather than whole
  layer types.

``text_callable``
  Optional callable that receives ``(layer_index, layer)`` and returns ``(text, above)``. Use it to place custom
  labels above or below individual layers. Built in callables are available through ``LAYERED_TEXT_CALLABLES``.

``text_vspacing``
  Vertical spacing between lines emitted by ``text_callable``. Increase it when multiline labels feel cramped.

``legend``
  If ``True``, add a legend showing rendered layer types and their colors. This is most useful when the visualization
  uses a custom color palette or many repeated layer classes.

``legend_text_spacing_offset``
  Extra space reserved for legend labels. Increase it if legend entries are clipping or if long names need more room.

``font``
  Optional Pillow font used for legend and annotation text. If omitted, visualkeras falls back to the default Pillow
  font.

``font_color``
  Text color for legend and annotation output.

``show_dimension``
  If ``True``, include layer output dimensions in the legend. This only affects legend output. It does not replace
  ``text_callable`` labels drawn next to layers.

Rendering Behavior
------------------

``background_fill``
  Background color for the final image. This can be a named color string or an RGBA tuple.

``draw_volume``
  If ``True``, draw boxes with 3D depth cues. If ``False``, render layers as flat 2D rectangles. Flat mode is often
  easier to read for small models or documentation examples.

``draw_reversed``
  Reverses the 3D viewing direction when ``draw_volume`` is enabled. This is useful for decoder style networks or
  diagrams where the standard front face orientation feels visually backwards.

``draw_funnel``
  Draws tapered transitions between consecutive layers. This can make size changes more obvious, but it also adds
  visual complexity. Turn it off for a cleaner, more schematic look.

``shade_step``
  Controls how much darker the shaded faces become in 3D mode. Larger values produce a stronger depth effect.

``connector_fill``
  Color used for connector and transition elements between layers.

``connector_width``
  Line width used for connector and transition elements. Increase it for presentation graphics or when the default
  strokes feel too light.

Sizing Strategy
---------------

``sizing_mode``
  Strategy used to convert tensor shapes into pixels. Layered mode supports ``accurate``, ``balanced``, ``capped``,
  ``logarithmic``, and ``relative``.

  ``accurate``
    Uses the raw tensor dimensions after scaling and clamping. This is the most literal option.

  ``balanced``
    Applies additional normalization so large and small layers remain readable in the same figure.

  ``capped``
    Respects ``dimension_caps`` so large dimensions do not dominate the output.

  ``logarithmic``
    Compresses large ranges of values. This is useful when the model mixes tiny tensors and very large tensors.

  ``relative``
    Sizes layers directly from their dimensions using ``relative_base_size``. This is the most proportional mode.

``dimension_caps``
  Optional mapping used by ``capped`` mode to limit specific dimension groups. The supported keys are ``channels``,
  ``sequence``, and ``general``. Use this when a few very large layers distort the overall layout.

``relative_base_size``
  Base pixel unit used by ``relative`` mode. A larger value makes all layers grow proportionally while preserving their
  relative differences.

Color and Style Control
-----------------------

``color_map``
  Coarse styling map keyed by layer class. Each value should usually provide ``fill`` and optionally ``outline``. Use
  this when you want consistent colors per layer type without managing per layer overrides.

``image_fit``
  Global default fit mode for images injected through ``styles``. Individual style entries can override it. Common
  values are ``fill``, ``contain``, ``cover``, and ``match_aspect``.

``image_axis``
  Global default axis used when rendering per layer images in 3D mode. Individual style entries can override it.
  Accepted values are ``x``, ``y``, and ``z``.

``styles``
  Fine grained per layer style overrides keyed by layer name or layer class. This is the most flexible styling system
  in layered mode. It can control fills, outlines, text, images, grouping behavior, and other advanced renderer
  details. See :doc:`advanced_styling_reference` for supported keys and examples.

Advanced Grouping and Overlays
------------------------------

``layered_groups``
  Sequence of group definitions used to draw labeled background regions behind related layers. This is useful for
  separating architectural stages such as encoder, bottleneck, and classifier blocks. See
  :doc:`advanced_styling_reference` for the expected structure.

``logo_groups``
  Sequence of logo placement definitions. Use this to add icons or small branded overlays to selected layers. This is
  mainly useful for presentation figures or strongly annotated diagrams. See :doc:`advanced_styling_reference`.

``logos_legend``
  If set to ``True`` or to a legend configuration mapping, render a legend for the logos supplied through
  ``logo_groups``. This is only meaningful when you are already using logo overlays.

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

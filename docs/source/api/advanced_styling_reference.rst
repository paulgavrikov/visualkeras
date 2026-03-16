============================
Advanced Styling Reference
============================

Complete reference for all advanced styling and configuration options in visualkeras.

This page documents the powerful but less-commonly-used customization features available across all visualization modes.

Overview
========

The options system in visualkeras supports:

- **Grouped highlights**: Visually group related layers with backgrounds and labels
- **Logos and brand elements**: Add logos and create legend entries
- **Per-layer styling**: Customize individual layers with fine-grained control
- **Image embedding**: Embed custom images or textures in visualizations
- **Text callables**: Control what text appears in visualizations
- **Dimension configuration**: Fine-tune how tensor shapes are displayed

All features are accessed through options objects and the ``styles`` parameter.

Grouped Highlights (layered_groups)
====================================

Group related layers with colored backgrounds and labels.

Purpose
-------

Use grouped highlights to:

- Visually separate architectural components (e.g., "Feature Extraction", "Classification")
- Highlight specific stages of processing
- Create visual hierarchy in complex models
- Make diagrams easier to understand at a glance

Basic Syntax
------------

Groups are defined via the ``layered_groups`` parameter in ``LayeredOptions``:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    import visualkeras

    options = LayeredOptions(
        layered_groups=[
            {
                'name': 'Feature Extraction',
                'type': 'block',
                'indices': [0, 1, 2, 3],  # Layer indices to group
                'fill': '#e8f4f8',         # Background color
                'outline': '#4a90e2',      # Border color
            },
            {
                'name': 'Classification',
                'type': 'block',
                'indices': [4, 5, 6],
                'fill': '#f8e8e8',         # Different color
                'outline': '#e24a4a',
            }
        ]
    )

    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

Parameter Details
------------------

Each group definition is a dictionary with:

============== ========== ===================================================================================================
Parameter      Type       Description
============== ========== ===================================================================================================
``name``       str        Label for the group (displayed above the block)
``type``       str        Must be ``'block'`` (groups consecutive layers into a visual block)
``indices``    list/tuple Layer indices to include in the group (0-indexed from model.layers)
``fill``       color      Background fill color (hex string like ``'#e8f4f8'`` or CSS color name)
``outline``    color      Border color around the group
============== ========== ===================================================================================================

Practical Example: ResNet-style Model
--------------------------------------

.. code-block:: python

    model = keras.Sequential([
        # Input/preprocessing layers
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(64, (7, 7), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPooling2D((3, 3)),
        
        # Feature extraction block
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        
        # Classification block
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256),
        keras.layers.ReLU(),
        keras.layers.Dense(10, activation='softmax'),
    ])

    options = LayeredOptions(
        layered_groups=[
            {
                'name': 'Input & Preprocessing',
                'type': 'block',
                'indices': [0, 1, 2, 3, 4],
                'fill': '#e3f2fd',      # Blue
                'outline': '#1976d2',
            },
            {
                'name': 'Feature Extraction',
                'type': 'block',
                'indices': [5, 6, 7, 8],
                'fill': '#f3e5f5',      # Purple
                'outline': '#7b1fa2',
            },
            {
                'name': 'Classification Head',
                'type': 'block',
                'indices': [9, 10, 11, 12],
                'fill': '#e8f5e9',      # Green
                'outline': '#388e3c',
            }
        ]
    )

    image = visualkeras.show(model, mode='layered', options=options, preset='presentation')
    image.show()

Logos and Branding (logo_groups, logos_legend)
================================================

Add logos and create visual legends for your diagrams.

Purpose
-------

Use logos to:

- Add company or institutional branding
- Create visual legends explaining color coding
- Add informational graphics
- Improve publication figures

Basic Logo Syntax
------------------

Define logos via the ``logo_groups`` and ``logos_legend`` parameters:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    
    options = LayeredOptions(
        logo_groups=[
            {
                'indices': [0, 1, 2],      # Layers to label
                'fill': '#3498db',         # Color
                'outline': '#2980b9',      # Border
                'text': 'Conv Block 1',    # Logo/label text
            }
        ],
        logos_legend=True  # Show legend explaining logos
    )

    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

Logo Parameter Details
-----------------------

============== ============ ============================================================================================================================
Parameter      Type         Description
============== ============ ============================================================================================================================
``text``       str          Text label for the logo
``indices``    list/tuple   Layer indices to apply this logo to
``fill``       color        Background color for logo
``outline``    color        Border color for logo
``positions``  str          When to show: ``'top'``, ``'bottom'``, ``'above'``, ``'below'`` (depends on context)
============== ============ ============================================================================================================================

``logos_legend`` can be:
    - ``True/False`` to enable/disable legend
    - A dict with legend customization options

Per-Layer Styling (styles Parameter)
=====================================

Fine-grained control over individual layers.

Purpose
-------

Use per-layer styling to:

- Customize specific layers differently from others
- Embed images in particular layers
- Apply layer-specific colors or effects
- Create highly customized publication figures

Basic Syntax
------------

Styles are passed via the ``styles`` parameter in LayeredOptions:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    import visualkeras

    styles = {
        0: {                              # Layer 0 (by index)
            'fill': '#1f77b4',
            'outline': '#0d47a1',
        },
        'conv_1': {                       # Layer by name
            'fill': '#ff7f0e',
            'outline': '#d84315',
        },
        keras.layers.Dense: {             # By layer type (applies to all Dense layers)
            'fill': '#2ca02c',
            'outline': '#1b5e20',
        }
    }

    options = LayeredOptions(styles=styles)
    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

Style Keys Reference
---------------------

Available style parameters per layer:

============= ======== ======================================================================================
Key           Type     Description
============= ======== ======================================================================================
``fill``      color    Layer background/fill color
``outline``   color    Layer border color
``line_width``int      Border thickness in pixels
``text``      str      Custom text to display in/for the layer
============= ======== ======================================================================================

Image Embedding (face_image)
=============================

Embed custom images or textures in layer visualizations (LeNet mode).

Purpose
-------

Use image embedding to:

- Show kernel visualizations
- Display activation maps
- Add custom graphics to layers
- Create distinctive, publication-quality diagrams

Basic Syntax (LeNet Mode)
---------------------------

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    styles = {
        0: {           # Layer 0
            'face_image': '/path/to/kernel_viz.png',
            'face_image_fit': 'cover',   # How to fit image
            'face_image_alpha': 200,     # Transparency (0-255)
            'face_image_inset': 2,       # Border inset in pixels
        }
    }
    
    options = LenetOptions(styles=styles)
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

Image Options Reference
------------------------

================= ======== =========================================================================
Key                Type     Description
================= ======== =========================================================================
``face_image``    path/str Path to image file or PIL.Image object
``face_image_fit`` str      Fit mode: ``'cover'``, ``'contain'``, ``'fill'``, ``'stretch'``
``face_image_alpha``int     Transparency 0 (transparent) to 255 (opaque). Default: 255
``face_image_inset``int     Border/inset in pixels. Default: 0
================= ======== =========================================================================

Fit Mode Behavior
------------------

- ``'cover'``: Fill entire space, crop image if necessary
- ``'contain'``: Fit entire image, may have empty space (letterboxing)
- ``'fill'``: Stretch to fill (may distort)
- ``'stretch'``: Same as ``'fill'``
- ``'scale-down'``: Use largest fitting size without upscaling

Image-Fitting Options (image_axis, image_fit)
==============================================

Control how images are displayed in layered visualizations.

Purpose
-------

These options control global image behavior in layered mode:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    
    options = LayeredOptions(
        image_axis='z',      # Which axis represents image depth (batch processing)
        image_fit='fill',    # How to fit images in layer blocks
    )

ImageAxis Options
------------------

==== ============================================
 Axis Description
==== ============================================
``'x'``  Image axis along width
``'y'``  Image axis along height
``'z'``  Image axis along depth/channels (default)
==== ============================================

ImageFit Options
-----------------

Same options as ``face_image_fit``:  ``'cover'``, ``'contain'``, ``'fill'``, ``'stretch'``, ``'scale-down'``

Text Callables and Labels
==========================

Customize what text appears in visualizations.

Using Built-in Callables
--------------------------

.. code-block:: python

    from visualkeras.options import LAYERED_TEXT_CALLABLES, LayeredOptions
    import visualkeras

    # Use a built-in text callable
    options = LayeredOptions(
        text_callable=LAYERED_TEXT_CALLABLES['name_shape']
    )
    
    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

Available built-in callables:
    - ``'name'`` - Layer name only
    - ``'type'`` - Layer type/class name
    - ``'shape'`` - Output tensor shape
    - ``'name_shape'`` - Both name and shape (recommended for presentation)

Custom Text Callables
----------------------

Define your own text function:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    
    def custom_label(layer_index, layer):
        """Return (text_string, is_above_layer)"""
        name = layer.__class__.__name__
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'unknown'
        text = f"{name}\n{output_shape}"
        above = False  # Show below the layer
        return (text, above)
    
    options = LayeredOptions(
        text_callable=custom_label
    )
    
    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

LeNet Label Callables
-----------------------

LeNet mode has special label callables:

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    def top_info(layer, render_shape):
        """Text above each layer"""
        return f"{render_shape.dH}×{render_shape.dW}×{render_shape.dZ}"  # h x w x channels
    
    def bottom_info(layer, render_shape):
        """Text below each layer"""
        return layer.__class__.__name__
    
    options = LenetOptions(
        top_label_callable=top_info,
        bottom_label_callable=bottom_info,
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

RenderShape Properties (for LeNet)
-----------------------------------

When using label callables in LeNet mode, the ``RenderShape`` object provides:

==== ========================================
 Attr  Description
==== ========================================
``dH``  Height of feature map visualization (pixels)
``dW``  Width of feature map visualization (pixels)
``dZ``  Number of channels (depth)
==== ========================================

Dimension Display Options
==========================

Control how tensor dimensions are shown.

ShowDimension
--------------

Enable dimension display on layers:

.. code-block:: python

    from visualkeras.options import LayeredOptions
    
    options = LayeredOptions(
        show_dimension=True,      # Show tensor shapes on layers
        dimension_caps=None,      # Custom dimension display
    )
    
    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

DimensionCaps
---------------

Limit dimension display formatting:

.. code-block:: python

    options = LayeredOptions(
        show_dimension=True,
        dimension_caps={
            'max_width': 100,     # Max characters for width display
            'max_height': 100,    # Max characters for height display
            'decimal_places': 2,  # Decimal precision
        }
    )

SizingMode Strategies
=======================

Control how layer sizes are calculated.

Available Modes
----------------

=========== ====================================================================
 Mode       Description
=========== ====================================================================
``'accurate'`` Use actual tensor dimensions (default, most precise)
``'balanced'`` Balance readability with accuracy
``'logarithmic'`` Logarithmic scaling (helps visualize large range of sizes)
``'relative'`` Relative to a base size (controlled by relative_base_size)
=========== ====================================================================

Example
--------

.. code-block:: python

    from visualkeras.options import LayeredOptions
    
    # Logarithmic scaling for models with extreme size variations
    options = LayeredOptions(
        sizing_mode='logarithmic',
        relative_base_size=20,  # Reference size for relative mode
    )
    
    image = visualkeras.show(model, mode='layered', options=options)
    image.show()

Complete Advanced Example
===========================

Combining multiple advanced features:

.. code-block:: python

    from visualkeras.options import LayeredOptions, LAYERED_TEXT_CALLABLES
    import visualkeras

    # Define advanced options
    options = LayeredOptions(
        # Layout
        spacing=20,
        padding=30,
        
        # Styling
        background_fill='white',
        font_color='black',
        
        # Groups
        layered_groups=[
            {
                'name': 'Feature Extraction',
                'type': 'block',
                'indices': [0, 1, 2, 3],
                'fill': '#e3f2fd',
                'outline': '#1976d2',
            },
            {
                'name': 'Dense Classifier',
                'type': 'block',
                'indices': [4, 5, 6],
                'fill': '#e8f5e9',
                'outline': '#388e3c',
            }
        ],
        
        # Per-layer styling
        styles={
            keras.layers.Conv2D: {'fill': '#1976d2'},
            keras.layers.Dense: {'fill': '#388e3c'},
        },
        
        # Text
        text_callable=LAYERED_TEXT_CALLABLES['name_shape'],
        
        # Dimensions
        show_dimension=True,
        sizing_mode='balanced',
        
        # Legends
        legend=True,
        logos_legend=True,
    )

    image = visualkeras.show(
        model,
        mode='layered',
        options=options,
        preset='presentation'
    )
    image.save('advanced_diagram.png')

Tips and Best Practices
=========================

1. **Start with presets**: Begin with ``preset='presentation'`` and customize from there
2. **Use grouped highlights**: Improve readability of complex models with visual grouping
3. **Test fit modes**: Try different ``image_fit`` values to find what looks best
4. **Combine overlays**: Use colors + groups + text for clearest communication
5. **Save high-DPI**: For publication, save PNG with sufficient resolution
6. **Limit complexity**: Too many style customizations can make diagrams harder to parse

See Also
========

- :doc:`../tutorials/index` for step-by-step guides
- :doc:`options` for options classes reference
- :doc:`../examples/index` for more examples

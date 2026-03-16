==============
LeNet View
==============

The LeNet style visualization renders models using classic "feature map stack" diagrams, where each layer is shown as a 2D representation of its output channels. This style is inspired by the original LeNet paper and produces distinctive, publication-quality architectural diagrams.

Best For
========

LeNet view excels for:

- **Academic papers**: Classic feature-map diagrams expected in computer vision research
- **Technical documentation**: Showing detailed channel-level architecture
- **Educational materials**: Clear visualization of how channels flow through layers
- **Presentations**: Distinctive, professional-looking architecture diagrams
- **Detailed architecture analysis**: Understanding how many channels each layer outputs

LeNet view is particularly effective for CNNs because it shows the actual channel count progression visually, making the architecture's growth/reduction patterns immediately obvious.

When to Use LeNet vs. Other Modes
=================================

Choose ``mode='lenet'`` over other modes when:

✅ **DO** use LeNet for:
    - CNN architectures you're publishing in papers/presentations
    - Models where channel progression is important to understand
    - Creating distinctive architecture diagrams
    - Detailed architectural analysis
    - Models with interesting channel dynamics (e.g., bottlenecks)

❌ **DON'T** use LeNet for:
    - Models highlighting computational graph structure: use graph mode
    - Models with complex branching or skip connections: use functional mode
    - Simple sequential models for learning: use layered mode
    - Very deep models (>100 layers): becomes impractical unless heavily customized

Basic Example
=============

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])

    # Default LeNet view
    image = visualkeras.show(model, mode='lenet')
    image.show()

The visualization shows each layer as stacked feature maps, with the height/color/size representing the channel count changing through the network.

Layout Customization
====================

Control spacing and sizing:

.. code-block:: python

    from visualkeras.options import LenetOptions, LENET_PRESETS
    
    # Custom layout
    options = LenetOptions(
        layer_spacing=50,      # Space between layers (larger = more spread out)
        map_spacing=5,         # Space between individual channels
        scale_xy=5.0,          # Scale width/height of feature maps
        max_visual_channels=16 # Limit channels shown (default 12)
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

Using Presets
=============

LeNet view includes three curated presets:

.. code-block:: python

    from visualkeras.options import LENET_PRESETS
    
    # Compact: minimal spacing, tight layout
    compact = visualkeras.show(
        model, 
        mode='lenet', 
        preset='compact'
    )
    
    # Presentation: large, detailed, publication-ready
    presentation = visualkeras.show(
        model,
        mode='lenet',
        preset='presentation'
    )
    
    # Default: balanced for general use
    default = visualkeras.show(
        model,
        mode='lenet',
        preset='default'
    )

Color and Style Customization
=============================

Visual appearance options:

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    options = LenetOptions(
        background_fill='white',      # Background color
        connector_fill='#333333',      # Connection line color
        connector_width=2,             # Line thickness
        font_color='black',            # Label color
        draw_connections=True,         # Show layer connections
        draw_patches=True,             # Show patch visualizations
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

Layer-Type Color Styling
------------------------

Customize colors per layer type:

.. code-block:: python

    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
    
    color_map = {
        Conv2D: {'fill': '#1f77b4', 'outline': '#0d47a1'},
        MaxPooling2D: {'fill': '#ff7f0e', 'outline': '#d84315'},
        Dense: {'fill': '#2ca02c', 'outline': '#1b5e20'},
    }
    
    image = visualkeras.show(
        model,
        mode='lenet',
        color_map=color_map
    )
    image.show()

Advanced: Text Labels and Callables
====================================

Control what text appears above and below each layer.

Built-in Text Options
-----------------------

By default, LeNet shows layer names below and shapes above. You can customize this with label callables:

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    # Define custom top label (above each layer)
    def top_label(layer, shape):
        """Show shape info above each layer"""
        return f"{shape.dH}×{shape.dW}×{shape.dZ}"  # height x width x channels
    
    # Define custom bottom label (below each layer)
    def bottom_label(layer, shape):
        """Show layer name below each layer"""
        return layer.__class__.__name__
    
    options = LenetOptions(
        top_label_callable=top_label,
        bottom_label_callable=bottom_label,
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

The ``RenderShape`` object provides:
    - ``dH``, ``dW``: Height and width of feature maps
    - ``dZ``: Number of channels (depth)

Practical Label Example
------------------------

Show parameter counts and activation info:

.. code-block:: python

    def info_label(layer, shape):
        """Show layer type and parameters"""
        layer_type = layer.__class__.__name__
        activation = getattr(layer, 'activation', None)
        if activation:
            return f"{layer_type} ({activation.__name__})"
        return layer_type
    
    options = LenetOptions(
        bottom_label_callable=info_label
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

Advanced: Embedding Images in Feature Maps
===========================================

Display custom images or textures within feature map visualizations using the ``styles`` parameter.

Basic Face Image
-----------------

Embed an image in a specific layer's visualization:

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    # Style: embed image in specific layer
    styles = {
        'layer_2': {  # Apply to layer at index 2
            'face_image': '/path/to/your/texture.png',
            'face_image_fit': 'cover',      # How to fit image in frame
            'face_image_alpha': 200,        # Transparency (0-255)
            'face_image_inset': 2,          # Border inset in pixels
        }
    }
    
    options = LenetOptions(styles=styles)
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

``face_image_fit`` options:
    - ``'cover'``: Fill entire feature map (may crop image)
    - ``'contain'``: Fit entire image (may have letterboxing)
    - ``'fill'``: Stretch to fill (may distort image)
    - ``'stretch'``: Same as fill
    - ``'scale-down'``: Largest fit that doesn't upscale

Complex Styling with Images
----------------------------

Combine image embedding with other style parameters:

.. code-block:: python

    styles = {
        'conv_1': {
            'face_image': 'kernel_viz.png',
            'face_image_fit': 'contain',
            'face_image_alpha': 180,
        },
        'conv_2': {
            'face_image': 'activation_viz.png',
            'face_image_fit': 'cover',
            'face_image_alpha': 200,
        }
    }
    
    options = LenetOptions(
        styles=styles,
        padding=30,
        layer_spacing=60,
        preset='presentation'
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.save('detailed_architecture.png')

Filtering Layers
================

Control which layers appear:

.. code-block:: python

    from tensorflow.keras.layers import BatchNormalization, Dropout
    from visualkeras.options import LenetOptions
    
    # Skip layers by type
    options = LenetOptions(
        type_ignore=[BatchNormalization, Dropout]
    )
    
    # Or skip by index (layer position)
    options = LenetOptions(
        index_ignore=[3, 5, 8]  # Skip layers 3, 5, and 8
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

Practical Examples
==================

**Classic CNN for Publication**

.. code-block:: python

    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    
    # High-quality publication figure
    image = visualkeras.show(
        model,
        mode='lenet',
        preset='presentation',
        color_map={
            keras.layers.Conv2D: {'fill': '#1f77b4'},
            keras.layers.MaxPooling2D: {'fill': '#ff7f0e'},
            keras.layers.Dense: {'fill': '#2ca02c'},
        }
    )
    image.save('architecture.png', dpi=300)

**Compact Diagram for Slides**

.. code-block:: python

    # Space-efficient, still readable
    image = visualkeras.show(
        model,
        mode='lenet',
        preset='compact',
        max_visual_channels=8  # Limit channels shown
    )
    image.show()

Reproducible Randomization
===========================

Control randomness in patch visualization:

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    options = LenetOptions(
        seed=42  # Fixed seed = reproducible randomization
    )
    
    # Same seed produces identical visualization every time
    image = visualkeras.show(model, mode='lenet', options=options)
    image.show()

See Also
========

- :doc:`../tutorials/index` for step-by-step guides
- :doc:`../api/lenet_style` for complete API reference
- :doc:`../examples/index` for other visualization types

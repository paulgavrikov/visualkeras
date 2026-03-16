==============
LeNet View
==============

The LeNet style visualization renders models using classic "feature map stack" diagrams. This style is inspired by the original LeNet paper and shows layers as stacked feature maps flowing left to right.

Best For
========

LeNet view works well for:

- Classic Convolutional Neural Networks
- Models where you want a stylized, publication-ready look
- Diagrams emphasizing the feature map progression
- Educational presentations

When to Use
===========

LeNet view differs from other visualization styles. Choose it when you want a distinctive, retro-inspired appearance rather than the standard layered or graph layouts.

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

    visualkeras.lenet_view(model).show()

Customization
=============

Key parameters for LeNet view:

**Layout Control**

- ``layer_spacing`` - Space between layers (default 40 pixels)
- ``map_spacing`` - Space between feature maps (default 4 pixels)
- ``padding`` - Border padding (default 20 pixels)

**Size Control**

- ``scale_xy`` - Scale factor for width and height (default 4.0)
- ``min_xy`` - Minimum layer size (default 20 pixels)
- ``max_xy`` - Maximum layer size (default 220 pixels)
- ``max_visual_channels`` - Limit how many channels to draw (default 12)

**Visual Style**

- ``background_fill`` - Background color (default 'black')
- ``connector_fill`` - Connection line color (default 'gray')
- ``connector_width`` - Connection line width (default 1 pixel)
- ``patch_fill`` - Patch visualization color (default '#7db7ff')
- ``patch_alpha_on_image`` - Patch transparency (default 140)

**Toggling Features**

- ``draw_connections`` - Draw layer connections (default True)
- ``draw_patches`` - Draw patch visualizations (default True)
- ``top_label`` - Show labels above layers (default True)
- ``bottom_label`` - Show labels below layers (default True)

Using Presets
=============

LeNet view includes three curated presets for different use cases.

.. code-block:: python

    # Compact layout for tight spaces
    visualkeras.lenet_view(model, preset='compact').show()

    # Presentation style for talks and papers
    visualkeras.lenet_view(model, preset='presentation').show()

    # Default balanced style
    visualkeras.lenet_view(model, preset='default').show()

Styling Layers
==============

Use the ``color_map`` parameter to customize individual layer colors:

.. code-block:: python

    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

    color_map = {
        Conv2D: {'fill': '#1f77b4'},
        MaxPooling2D: {'fill': '#ff7f0e'},
        Dense: {'fill': '#2ca02c'},
    }

    visualkeras.lenet_view(model, color_map=color_map).show()

Advanced Features
=================

Ignoring Layers
---------------

Skip specific layers during visualization:

.. code-block:: python

    from tensorflow.keras.layers import Dropout

    # Ignore by type
    visualkeras.lenet_view(
        model,
        type_ignore=[Dropout]
    ).show()

    # Ignore by index (layer position)
    visualkeras.lenet_view(
        model,
        index_ignore=[5, 7]  # Skip layers 5 and 7
    ).show()

Custom Seeds and Randomization
-------------------------------

Control randomized patch placement:

.. code-block:: python

    visualkeras.lenet_view(
        model,
        seed=42  # Reproducible randomization
    ).show()

Common Patterns
===============

**Dark theme (publication quality)**

.. code-block:: python

    visualkeras.lenet_view(
        model,
        background_fill='black',
        font_color='white',
        connector_fill='gray',
        preset='presentation'
    ).show()

**Compact spacing**

.. code-block:: python

    visualkeras.lenet_view(
        model,
        preset='compact'
    ).show()

**Large feature maps**

.. code-block:: python

    visualkeras.lenet_view(
        model,
        max_xy=300,
        scale_xy=6.0
    ).show()

See Also
========

- :doc:`../tutorials/index` for step-by-step guides
- :doc:`../api/index` for complete API reference
- Other visualization types in :doc:`index`

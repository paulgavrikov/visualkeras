===========
LeNet Style
===========

LeNet-style visualization for neural networks.

Overview
========

The LeNet renderer displays neural networks using classic "feature map stack" diagrams. Each layer is rendered as a 2D representation of its output channels flowing left to right, creating distinctive, publication-quality architectural diagrams.

**When to use:**
    - Academic papers and technical publications
    - CNNs where channel progression is important to visualize
    - Creating distinctive, retro-styled architecture diagrams
    - Detailed architectural analysis at the channel level
    - Presentations emphasizing classical CNN concepts

**When NOT to use:**
    - Understanding computational structure (use graph or functional)
    - Simple learning/teaching (use layered mode)
    - Models with complex non-sequential components
    - Deep models (>100 layers) without customization

API Reference
=============

.. automodule:: visualkeras.lenet
   :members:
   :undoc-members:
   :show-inheritance:

Key Parameters
==============

**Core Parameters**

- ``model``: Keras model instance (typically Sequential or simple Functional CNNs)
- ``to_file``: Path to save output (optional)
- ``preset``: Use preset configuration ('default', 'compact', 'presentation')
- ``options``: LenetOptions object for bundled configuration

**Layout Control**

- ``layer_spacing``: Space between layers (default: 40)
- ``map_spacing``: Space between individual channels (default: 4)
- ``scale_xy``: Width/height scale multiplier (default: 4.0)
- ``min_xy``, ``max_xy``: Min/max feature map size (pixels)
- ``max_visual_channels``: Limit channels shown (default: 12)
- ``padding``: Border space (default: 20)

**Styling**

- ``background_fill``: Background color (default: 'black')
- ``connector_fill``: Connection line color (default: 'gray')
- ``connector_width``: Line thickness (default: 1)
- ``patch_fill``: Feature map color (default: '#7db7ff')
- ``patch_alpha_on_image``: Transparency (default: 140)
- ``color_map``: Dict mapping layer types to ``{'fill': '...', ...}``

**Content Control**

- ``top_label_callable``: Function controlling text above layers
- ``bottom_label_callable``: Function controlling text below layers
- ``draw_connections``: Show connections between layers (default: True)
- ``draw_patches``: Show patch visualizations (default: True)
- ``top_label``, ``bottom_label``: Show/hide label positions (default: True)

**Advanced**

- ``styles``: Per-layer style overrides including ``face_image``
- ``seed``: Fixed seed for reproducible randomization
- ``type_ignore``: Layer types to skip
- ``index_ignore``: Layer indices to skip

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

    import visualkeras
    
    image = visualkeras.show(model, mode='lenet')
    image.show()

With Presets
------------

.. code-block:: python

    # Publication quality
    image = visualkeras.show(model, mode='lenet', preset='presentation')
    
    # Compact for slides
    image = visualkeras.show(model, mode='lenet', preset='compact')

Custom Label Functions
----------------------

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    def top_label(layer, shape):
        return f"{shape.dH}×{shape.dW}×{shape.dZ}"
    
    def bottom_label(layer, shape):
        return layer.__class__.__name__
    
    options = LenetOptions(
        top_label_callable=top_label,
        bottom_label_callable=bottom_label,
    )
    
    image = visualkeras.show(model, mode='lenet', options=options)

With Embedded Images
--------------------

.. code-block:: python

    from visualkeras.options import LenetOptions
    
    styles = {
        0: {
            'face_image': 'kernel_viz.png',
            'face_image_fit': 'cover',
            'face_image_alpha': 200,
        }
    }
    
    options = LenetOptions(styles=styles)
    image = visualkeras.show(model, mode='lenet', options=options)

See Also
========

- :doc:`../examples/lenet_view` for detailed LeNet examples
- :doc:`../tutorials/tutorial_01_basic_visualization` for tutorial
- :doc:`advanced_styling_reference` for advanced styling
- :doc:`options` for LenetOptions reference

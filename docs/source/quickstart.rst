===========
Quickstart
===========

Get visualkeras running in just 5 minutes!

Installation
============

Install visualkeras using pip:

.. code-block:: bash

   pip install visualkeras

Your First Visualization
========================

Let's create a simple neural network and visualize it using the unified ``show()`` API:

.. code-block:: python

   import tensorflow as tf
   from tensorflow import keras
   import visualkeras

   # Create a simple Sequential model
   model = keras.Sequential([
       keras.layers.Input(shape=(28, 28, 1)),
       keras.layers.Conv2D(32, (3, 3), activation='relu'),
       keras.layers.MaxPooling2D((2, 2)),
       keras.layers.Conv2D(64, (3, 3), activation='relu'),
       keras.layers.MaxPooling2D((2, 2)),
       keras.layers.Conv2D(64, (3, 3), activation='relu'),
       keras.layers.Flatten(),
       keras.layers.Dense(64, activation='relu'),
       keras.layers.Dense(10)
   ])

   # Display the model architecture using show() with mode parameter
   image = visualkeras.show(model, mode='layered')
   image.show()

What Just Happened?
===================

visualkeras generated a **layered view** of your model (using ``mode='layered'``), showing:

- Each layer as a colored block
- Tensor shapes flowing through the network
- How data is transformed at each step

The ``show()`` function is the unified API that supports four visualization modes.

This layered view is great for understanding **Convolutional Neural Networks (CNNs)**.

Using Presets
==============

visualkeras includes curated presets for different use cases:

.. code-block:: python

   # Compact layout (minimal spacing)
   compact = visualkeras.show(model, mode='layered', preset='compact')

   # Presentation style (large, detailed, suitable for papers/slides)
   presentation = visualkeras.show(model, mode='layered', preset='presentation')

   # Default balanced style
   default = visualkeras.show(model, mode='layered', preset='default')

Trying Different Modes
======================

The ``show()`` API makes it easy to experiment with different visualization styles:

.. code-block:: python

   # These all work with the same function - just change the mode
   
   layered_viz = visualkeras.show(model, mode='layered')        # Layer stacks
   graph_viz = visualkeras.show(model, mode='graph')            # Computational graph
   functional_viz = visualkeras.show(model, mode='functional')  # Layered + graph hybrid
   lenet_viz = visualkeras.show(model, mode='lenet')             # Classic feature map style

Try each mode to see which best represents your model!

Next Steps
==========

Ready to explore more? Check out:

- :doc:`installation` - Detailed installation and troubleshooting
- :doc:`tutorials/index` - Step-by-step tutorials
- :doc:`examples/index` - Gallery of examples
- :doc:`api/index` - Complete API reference

For more control over styling, see :doc:`tutorials/index`.

Common Customizations
=====================

Changing Colors and Styles
---------------------------

Use the unified ``show()`` API with color customization:

.. code-block:: python

   # Define colors with fill and outline for each layer type
   image = visualkeras.show(
       model,
       mode='layered',
       color_map={
           keras.layers.Conv2D: {'fill': '#3498db', 'outline': '#2980b9'},
           keras.layers.MaxPooling2D: {'fill': '#2ecc71', 'outline': '#27ae60'},
           keras.layers.Dense: {'fill': '#e74c3c', 'outline': '#c0392b'}
       }
   )
   image.show()

Using Different Visualization Modes
-------------------------------------

visaalkeras supports four visualization modes. Each mode is useful for different model types:

.. code-block:: python

   # Layered view: Great for CNNs, shows stacked feature maps
   layered = visualkeras.show(model, mode='layered')
   
   # Graph view: Shows computational structure for any model type
   graph = visualkeras.show(model, mode='graph')
   
   # Functional view: Graph-aware layering (bridges both approaches)
   functional = visualkeras.show(model, mode='functional')
   
   # LeNet view: Classic feature map stack visualization
   lenet = visualkeras.show(model, mode='lenet')
   
   # Use presets for publication-ready output
   publication = visualkeras.show(model, mode='layered', preset='presentation')

Saving to File
--------------

.. code-block:: python

   image = visualkeras.layered_view(model)
   image.save('model_architecture.png')

Troubleshooting
===============

**The visualization is too crowded**
  - Try using ``graph_view()`` instead of ``layered_view()``
  - Reduce the scale factor: ``visualkeras.layered_view(model, scale_xy=20)``

**Shapes aren't displaying correctly**
  - Make sure your model has an Input layer defined
  - Call ``model.build((batch_size, *input_shape))`` if needed

**Getting an error?**
  - Check the :doc:`installation` page for common issues
  - See the full :doc:`tutorials/index` for detailed examples

Need Help?
==========

- 📖 Check the :doc:`tutorials/index`
- 🖼️ Browse the :doc:`examples/index`
- 🐛 `Report an issue on GitHub <https://github.com/paulgavrikov/visualkeras/issues>`_
- 💬 `Start a discussion <https://github.com/paulgavrikov/visualkeras/discussions>`_

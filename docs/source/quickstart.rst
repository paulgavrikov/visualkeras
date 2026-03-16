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

Let's create a simple neural network and visualize it:

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

   # Display the model architecture
   visualkeras.layered_view(model).show()

What Just Happened?
===================

visualkeras generated a **layered view** of your model, showing:

- Each layer as a colored block
- Tensor shapes flowing through the network
- How data is transformed at each step

This layered view is great for understanding **Convolutional Neural Networks (CNNs)**.

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

Changing Colors
---------------

.. code-block:: python

   visualkeras.layered_view(model, color_map={
       keras.layers.Conv2D: 'Blue',
       keras.layers.MaxPooling2D: 'Green',
       keras.layers.Dense: 'Red'
   }).show()

Using Graph View Instead
-------------------------

For more complex models, use graph view:

.. code-block:: python

   visualkeras.graph_view(model).show()

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

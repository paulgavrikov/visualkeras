=======================
How to File a Good Bug
=======================

Found an issue? We'd love to hear about it! Filing a good bug report helps us fix problems faster and makes visualkeras better for everyone.

Quick Checklist
===============

Before filing a bug, check off these items:

- ✅ You're using the latest version: ``pip install --upgrade visualkeras``
- ✅ The issue reproduces consistently
- ✅ You've searched `existing issues <https://github.com/paulgavrikov/visualkeras/issues>`_ 
- ✅ The issue is specifically with visualkeras (not TensorFlow/Keras itself)

Where to Report
================

File bugs on GitHub: `paulgavrikov/visualkeras/issues <https://github.com/paulgavrikov/visualkeras/issues>`_

Essential Information
=====================

Every good bug report includes these 5 things:

1. **What Version of visualkeras?**
   
   .. code-block:: python
   
       import visualkeras
       print(visualkeras.__version__)
   
   Or check via pip:
   
   .. code-block:: bash
   
       pip show visualkeras

2. **What Python Version?**
   
   .. code-block:: bash
   
       python --version

3. **Environment Details**
   
   Include:
   - Operating System (Windows/macOS/Linux)
   - TensorFlow/Keras version: ``pip show tensorflow``
   - Keras standalone or tf.keras?
   - Virtual environment (conda/venv/Poetry)?

4. **Minimal Reproducible Example**
   
   Provide the **shortest possible code** that shows the problem:
   
   .. code-block:: python
   
       import tensorflow as tf
       from tensorflow import keras
       import visualkeras
       
       # Create minimal model that shows the issue
       model = keras.Sequential([
           keras.layers.Input(shape=(28, 28, 1)),
           keras.layers.Conv2D(32, (3, 3)),
           keras.layers.Dense(10),
       ])
       
       # This causes the bug:
       image = visualkeras.show(model, mode='layered')
       # Error: ... (paste full error here)
   
   **Do:**
   - Use the smallest model that reproduces it
   - Use standard layers (Conv2D, Dense, etc.)
   - Include the exact error message/traceback
   - Copy-paste code directly (no screenshots of code)
   
   **Don't:**
   - Share entire model files
   - Use obscure custom layers if not necessary
   - Describe the problem in words instead of showing code

5. **What Happened vs. What You Expected**
   
   **Actual behavior:**
   ```
   When I call visualkeras.show(model, mode='functional'), I get a TypeError:
   'NoneType' object is not subscriptable at line 245 of functional.py
   ```
   
   **Expected behavior:**
   ```
   The function should return a PIL Image showing the model visualization.
   ```
   
   **Screenshots** (optional but helpful):
   - Screenshot of error message
   - Screenshot of expected vs. actual output

Example Bug Report
==================

**Title:** ``show()` with mode='functional' crashes on models with skip connections``

**Description:**

When visualizing a model with skip connections using ``mode='functional'``, visualkeras crashes with a TypeError.

**Version Info:**
- visualkeras: 2.4.1
- Python: 3.11.4
- TensorFlow: 2.17.0 (tf.keras)
- OS: Ubuntu 22.04

**Minimal Reproducible Example:**

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    import visualkeras
    
    # Simple model with skip connection
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Conv2D(64, (3, 3))(inputs)
    x = keras.layers.Conv2D(64, (3, 3))(x)
    
    # Skip connection
    residual = inputs
    x = keras.layers.Add()([x, residual])  # ← Fails here
    
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # This crashes:
    img = visualkeras.show(model, mode='functional')

**Error Message:**

.. code-block:: text

    Traceback (most recent call last):
      File "reproduce_bug.py", line 20, in <module>
        img = visualkeras.show(model, mode='functional')
      File ".../visualkeras/show.py", line 115, in show
        return functional_view(model, preset=preset, options=options, **overrides)
      File ".../visualkeras/functional.py", line 245, in functional_view
        dims = shape_dict[layer.name]
    TypeError: 'NoneType' object is not subscriptable

**Expected Behavior:**

Should return a PIL Image with the functional visualization showing the skip connection.

**Actual Behavior:**

Crashes with TypeError. The issue doesn't occur with ``mode='layered'`` or ``mode='graph'``.

---

Common Issues and Solutions
============================

Before Filing, Try These First
-------------------------------

**"ModuleNotFoundError: No module named 'aggdraw'"**

.. code-block:: bash

    # Install missing dependencies
    pip install aggdraw pillow numpy

**"visualkeras.show() opens no image window"**

You need an image viewer configured:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install imagemagick display
    
    # macOS
    brew install imagemagick
    
    # Or just save to file instead
    image = visualkeras.show(model)
    image.save('model.png')

**Model visualization looks wrong or empty**

Check these:

.. code-block:: python

    # Make sure model has Input layer
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),  # ← Required!
        keras.layers.Dense(10),
    ])
    
    # Build model if needed
    model.build((None, 28, 28, 1))
    
    # Check model summary
    model.summary()

**"This works on the latest GitHub version but not PyPI"**

Install from main branch:

.. code-block:: bash

    pip install git+https://github.com/paulgavrikov/visualkeras

**Color/styling parameters not working**

Make sure you're passing correct format:

.. code-block:: python

    # ✅ CORRECT - dict with fill and outline
    visualkeras.show(
        model,
        color_map={
            keras.layers.Conv2D: {'fill': '#1976d2', 'outline': '#0d47a1'}
        }
    )
    
    # ❌ WRONG - plain strings
    visualkeras.show(
        model,
        color_map={keras.layers.Conv2D: 'blue'}  # This won't work!
    )

See Also
========

- :doc:`installation` - Troubleshooting installation issues
- :doc:`quickstart` - Getting started guide
- `GitHub Issues <https://github.com/paulgavrikov/visualkeras/issues>`_ - View all reported bugs

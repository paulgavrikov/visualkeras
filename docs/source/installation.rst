=============
Installation
=============

This guide covers how to install visualkeras and troubleshoot common issues.

Quick Installation
==================

The easiest way to install visualkeras is via pip:

.. code-block:: bash

   pip install visualkeras

This installs visualkeras and all required dependencies.

System Requirements
===================

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimal (< 100 MB)

Dependencies
============

visualkeras requires:

- **pillow** >= 6.2.0 - Image processing
- **numpy** >= 1.18.1 - Numerical computing
- **aggdraw** >= 1.3.11 - Anti-aliased drawing
- **TensorFlow** or **Keras** (not required for basic imports, but needed for actual visualization)

Optional Dependencies
=====================

- **TensorFlow 2.17+** (with tf.keras)
- **Standalone keras package** (keras >= 3)

Compatibility Table
===================

The following table summarizes visualkeras compatibility with various TensorFlow and Keras configurations:

.. list-table::
   :header-rows: 1
   :widths: 15 15 25 25 20

   * - Feature
     - Support
     - Python
     - TensorFlow/Keras
     - OS

   * - Core package (import visualkeras)
     - ✅ Supported
     - 3.9-3.12
     - Not required
     - All

   * - Visualization with tf.keras
     - ✅ Supported
     - 3.11
     - TensorFlow 2.17+
     - Linux (primarily)

   * - Standalone keras package
     - ⚠️ Best effort
     - 3.11-3.12
     - keras >= 3
     - Linux (primarily)

Platform-Specific Installation
==============================

Linux (Ubuntu/Debian)
---------------------

.. code-block:: bash

   # Install system dependencies
   sudo apt-get install python3-dev libfreetype6-dev

   # Install visualkeras
   pip install visualkeras

macOS
-----

.. code-block:: bash

   # Using Homebrew (recommended for aggdraw)
   brew install freetype

   # Install visualkeras
   pip install visualkeras

**Note**: If you encounter issues with aggdraw installation on macOS, try:

.. code-block:: bash

   pip install --upgrade --force-reinstall aggdraw

Windows
-------

.. code-block:: bash

   pip install visualkeras

**Troubleshooting on Windows**: If aggdraw fails to install, download pre-built wheels from `Unofficial Windows Binaries <https://www.lfd.uci.edu/~gohlke/pythonlibs/#aggdraw>`_.

Development Installation
=========================

To install visualkeras from source for development:

.. code-block:: bash

   git clone https://github.com/paulgavrikov/visualkeras.git
   cd visualkeras
   pip install -e ".[dev]"

This installs the package in editable mode with development dependencies.

Verifying Installation
======================

To verify visualkeras is installed correctly:

.. code-block:: python

   import visualkeras
   print(visualkeras.__version__)

You should see the version number printed.

Troubleshooting
===============

ImportError: No module named 'visualkeras'
-------------------------------------------

**Problem**: Python can't find visualkeras after installation.

**Solution**:
  1. Verify installation: ``pip list | grep visualkeras``
  2. If not listed, reinstall: ``pip install --upgrade visualkeras``
  3. If using a virtual environment, ensure it's activated
  4. Try restarting your Python kernel/IDE

"aggdraw" installation fails
-----------------------------

**Problem**: ``pip install visualkeras`` fails with aggdraw errors.

**Solution** (in order of preference):

1. **Update pip and wheels**:

   .. code-block:: bash

      pip install --upgrade pip setuptools wheel
      pip install visualkeras

2. **Pre-built wheels** (macOS/Windows):

   .. code-block:: bash

      pip install --upgrade --force-reinstall aggdraw

3. **From conda** (if available):

   .. code-block:: bash

      conda install -c conda-forge aggdraw
      pip install visualkeras

4. **Manual wheel installation** (Windows):

   Download from `Unofficial Windows Binaries <https://www.lfd.uci.edu/~gohlke/pythonlibs/#aggdraw>`_ and:

   .. code-block:: bash

      pip install aggdraw-<version>-<python>-<arch>.whl
      pip install visualkeras

"ModuleNotFoundError: No module named 'tensorflow'" or "No module named 'keras'"
-------------------------------------------------------------------------------

**Problem**: You're trying to visualize a model but TensorFlow/Keras isn't installed.

**Solution**: Install TensorFlow:

.. code-block:: bash

   pip install tensorflow  # For tf.keras
   # or
   pip install keras      # For standalone Keras

ImportError: cannot import name 'layered_view'
-----------------------------------------------

**Problem**: Can't import specific functions from visualkeras.

**Solution**:

.. code-block:: python

   # Wrong:
   from visualkeras import layered_view

   # Correct:
   import visualkeras
   visualkeras.layered_view(model)

   # Or:
   from visualkeras import layered_view

Rendering/Output Issues
-----------------------

**Problem**: Visualization is blank, corrupted, or shows errors.

**Solutions**:

1. **Ensure model has Input layer**:

   .. code-block:: python

      # Add Input layer explicitly:
      model = keras.Sequential([
          keras.layers.Input(shape=(28, 28, 1)),
          # ... rest of layers
      ])

2. **Build the model first**:

   .. code-block:: python

      model.build((None, 28, 28, 1))  # Specify batch_size and input shape
      visualkeras.layered_view(model).show()

3. **Try graph view instead**:

   .. code-block:: python

      visualkeras.graph_view(model).show()

Still Having Issues?
====================

If you can't find a solution:

1. Check the :doc:`../quickstart` for basic usage
2. Browse the :doc:`../tutorials/index` for step-by-step examples
3. Search `existing GitHub issues <https://github.com/paulgavrikov/visualkeras/issues>`_
4. `Open a new issue <https://github.com/paulgavrikov/visualkeras/issues/new>`_ with:

   - Your operating system and Python version
   - The error message (full traceback)
   - A minimal code example to reproduce the issue
   - Your TensorFlow/Keras version

Getting Help
============

- 📖 **Tutorials**: :doc:`../tutorials/index`
- 🖼️ **Examples**: :doc:`../examples/index`
- 📚 **API Reference**: :doc:`../api/index`
- 🐛 **Issue Tracker**: `GitHub Issues <https://github.com/paulgavrikov/visualkeras/issues>`_
- 💬 **Discussions**: `GitHub Discussions <https://github.com/paulgavrikov/visualkeras/discussions>`_

====================
visualkeras
====================

**Architecture visualization of Keras/TensorFlow models**

.. image:: https://img.shields.io/pypi/v/visualkeras.svg
   :alt: Latest Version
   :target: https://pypi.python.org/pypi/visualkeras

.. image:: https://img.shields.io/pypi/dm/visualkeras.svg
   :alt: Download Count
   :target: https://pypi.python.org/pypi/visualkeras

.. image:: https://img.shields.io/badge/tests-170%2F170%20passed%20(100%25)-brightgreen
   :alt: Test Pass Rate

.. image:: https://img.shields.io/badge/coverage-95.09%25-brightgreen
   :alt: Coverage

.. raw:: html

   <br><br>

visualkeras is a Python package to help visualize Keras (either standalone or included in TensorFlow) neural network architectures. It provides a unified, high-level API for creating publication-quality visualizations of your models.

Key Features
============

Visualkeras provides several ways to visualize Keras and TensorFlow models. It includes layered diagrams for
sequential CNN style architectures, graph based views for model topology, a functional view for more complex
structures, and a LeNet inspired mode for feature map style presentations.

The library also gives you control over how the result looks. You can adjust spacing, colors, labels, sizing, and
other rendering options without changing the model itself. Images can be shown directly in Python or saved for papers,
slides, and documentation.

Quick Links
===========

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: 🚀 Quickstart
      :link: quickstart
      :link-type: doc

      Get visualkeras running in 5 minutes with a simple example.

   .. grid-item-card:: 📖 Installation
      :link: installation
      :link-type: doc

      Installation instructions and troubleshooting guide.

   .. grid-item-card:: 🎓 Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step guides for mastering visualkeras.

   .. grid-item-card:: 🖼️ Examples
      :link: examples/index
      :link-type: doc

      Gallery of examples and use cases.

   .. grid-item-card:: 📚 API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation for all modules.

   .. grid-item-card:: 🔗 GitHub
      :link: https://github.com/paulgavrikov/visualkeras

      View source code and contribute on GitHub.

When to Use Which Visualization Style
======================================

When to Use Which Visualization Style
  =====================================

  .. list-table::
     :header-rows: 1
     :widths: 20 40 40

     * - Style
       - Best For
       - Pros & Cons

     * - **Layered View**
       - Convolutional neural networks and models where shape progression matters
       - | Pros
         | Intuitive for CNN style architectures
         |
         | Shows tensor size changes clearly
         |
         | Cons
         | Limited for complex branching models

     * - **Graph View**
       - General model topology, branching models, and connection focused diagrams
       - | Pros
         | Works with many model types
         |
         | Shows connections clearly
         |
         | Cons
         | Can become dense for large models

     * - **Functional View**
       - Functional models with multiple paths, merges, inputs, or outputs
       - | Pros
         | Better suited to complex functional structures
         |
         | Preserves more layer level structure than a pure graph view
         |
         | Cons
         | Less direct than layered view for simple sequential models

     * - **LeNet View**
       - Feature map style diagrams and presentation figures for CNNs
       - | Pros
         | Useful for classic CNN style visualizations
         |
         | Good fit for publication and teaching material
         |
         | Cons
         | Not ideal for highly irregular or non sequential architectures

Table of Contents
=================

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart
   installation
   tutorials/index
   examples/index
   api/index
   bug_reporting

Support & Community
===================

- 🐛 **Found a bug?** `Open an issue on GitHub <https://github.com/paulgavrikov/visualkeras/issues>`_ — see :doc:`bug_reporting` for tips on filing a great bug report
- 📝 **Want to contribute?** See our `contributing guide <https://github.com/paulgavrikov/visualkeras/blob/master/CONTRIBUTING.MD>`_

Citation
========

If you find visualkeras helpful in your research, please cite it:

.. code-block:: bibtex

   @misc{Gavrikov2020VisualKeras,
     author = {Gavrikov, Paul and Patapati, Santosh},
     title = {visualkeras},
     year = {2020},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/paulgavrikov/visualkeras}},
   }

License
=======

visualkeras is licensed under the MIT License. See the `LICENSE <https://github.com/paulgavrikov/visualkeras/blob/master/LICENSE>`_ file for details.

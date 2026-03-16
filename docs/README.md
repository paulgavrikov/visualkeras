# visualkeras Documentation

This directory contains the source code for the visualkeras documentation built with Sphinx.

## Quick Start

### Building Documentation Locally

**Option 1: Using the build script**

```bash
cd docs
./build_docs.sh
```

**Option 2: Using make (requires make to be installed)**

```bash
cd docs
make html
```

**Option 3: Using Sphinx directly**

```bash
cd docs
sphinx-build -b html source build/html
```

### Viewing Documentation

After building, view the documentation locally:

```bash
cd docs
python -m http.server 8000 --directory build/html
```

Then open <http://localhost:8000> in your browser.

## Directory Structure

```
docs/
├── source/                      # Documentation source files
│   ├── conf.py                 # Sphinx configuration
│   ├── index.rst               # Landing page
│   ├── installation.rst        # Installation and troubleshooting
│   ├── quickstart.rst          # 5-minute quick start
│   ├── tutorials/              # Step-by-step tutorials
│   │   ├── index.rst
│   │   ├── tutorial_01_basic_visualization.rst
│   │   ├── tutorial_02_styling_customization.rst
│   │   └── tutorial_03_advanced_usage.rst
│   ├── examples/               # Example gallery
│   │   ├── index.rst
│   │   ├── cnn_models.rst
│   │   ├── sequential_models.rst
│   │   ├── functional_models.rst
│   │   └── custom_models.rst
│   ├── api/                    # Auto-generated API documentation
│   │   ├── index.rst
│   │   ├── layered.rst
│   │   ├── functional.rst
│   │   ├── graph.rst
│   │   └── options.rst
│   ├── _static/                # Static assets (images, CSS, etc.)
│   └── _templates/             # Custom templates
├── build/                       # Generated HTML files (git ignored)
├── Makefile                     # Build automation (Unix/Linux/macOS)
└── requirements.txt             # Documentation build dependencies
```

## Dependencies

Documentation dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Main Dependencies

- **sphinx**: Documentation generator
- **furo**: Modern Sphinx theme
- **sphinx-autodoc-typehints**: Better type hints in API docs
- **sphinx-design**: Grid and card components for layouts

## Contributing to Documentation

### Adding a New Page

1. Create a new `.rst` file in the appropriate directory
2. Add it to the `toctree` directive in the parent index file
3. Rebuild the documentation to verify it works

### Updating API Documentation

API documentation is automatically generated from docstrings in the Python source code. To update:

1. Edit docstrings in `visualkeras/*.py`
2. Rebuild the documentation:

   ```bash
   ./build_docs.sh
   ```

3. Check that the API docs render correctly

### Writing reStructuredText (RST)

The documentation uses reStructuredText format. Quick reference:

```rst
# Headings (various levels)
Main Title
==========

Subtitle
--------

Subsubtitle
^^^^^^^^^^^

# Bold and italic
**bold text**
*italic text*

# Code blocks
.. code-block:: python

    import visualkeras
    model = keras.Sequential([...])

# Links
`Link text <https://example.com>`_

# Lists
- Item 1
- Item 2
  - Nested item

# Tables
.. list-table::
   :header-rows: 1

   * - Column 1
     - Column 2
   * - Data 1
     - Data 2
```

### Sphinx Directives

Common Sphinx directives used:

```rst
# Code highlighting
.. code-block:: python

    code here

# Admonitions
.. warning::
   This is important

.. note::
   This is a note

.. tip::
   Helpful tip

# Cross-references
:doc:`relative/path/to/file`
:py:func:`function_name`
:py:class:`ClassName`
:py:mod:`module_name`

# Grids (from sphinx-design)
.. grid:: 2

   .. grid-item-card:: Title
      
      Content here
```

## Configuration

Documentation configuration is in `source/conf.py`:

- **Theme**: Furo (modern, clean design)
- **Extensions**: autodoc, napoleon, intersphinx, sphinx-design
- **Auto-API docs**: Enabled for all visualkeras modules

## Building for Production

For ReadTheDocs deployment, the configuration is in `.readthedocs.yml` in the project root.

See [READTHEDOCS_SETUP.md](READTHEDOCS_SETUP.md) for detailed instructions on setting up automatic builds and hosting.

## Continuous Integration and Deployment

The project uses GitHub Actions and ReadTheDocs to maintain documentation quality and availability.

### GitHub Actions

Automatically builds and validates documentation on every commit and pull request. See [CI_CD_GUIDE.md](CI_CD_GUIDE.md) for details.

### ReadTheDocs

Hosts documentation publicly and rebuilds on every push. Free hosting with automatic versioning and search functionality.

See [READTHEDOCS_SETUP.md](READTHEDOCS_SETUP.md) for complete setup instructions.

## Checking Documentation Quality

### Check for broken links

```bash
sphinx-build -b linkcheck -d build/doctrees source build/linkcheck
```

### Check doctests

```bash
sphinx-build -b doctest -d build/doctrees source build/doctest
```

## Common Issues

### "Theme 'furo' not found"

Install furo: `pip install furo`

### "Can't import visualkeras module"

Ensure the project root is in Python path. The `conf.py` adds the parent directory automatically.

### Warnings about missing documents

Create placeholder `.rst` files for any referenced documents in toctrees.

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Furo Theme](https://pradyunsg.me/furo/)
- [Napoleon for NumPy-style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

## Help & Support

- Edit `.rst` files directly for content changes
- Rebuild with `./build_docs.sh` to preview
- Check warnings in build output
- Open HTML files in `build/html/` to review

---

For more information, see the main [README.md](../README.md) in the project root.

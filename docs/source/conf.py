# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the parent directory to the path so we can import visualkeras
sys.path.insert(0, os.path.abspath('../../'))

# Project information
project = 'visualkeras'
copyright = '2020-2024, Paul Gavrikov'
author = 'Paul Gavrikov'
version = '0.2.0'
release = '0.2.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source configuration
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
pygments_style = 'monokai'

# HTML output configuration
html_theme = 'furo'
html_theme_options = {
    'announcement': 'visualkeras is an open-source project. Feedback is welcome!',
    'body_max_width': '100%',
    'sidebar_hide_name': False,
    'top_of_page_button': 'edit',
    'source_repository': 'https://github.com/paulgavrikov/visualkeras',
    'source_branch': 'master',
    'source_directory': 'docs/source/',
}

html_static_path = ['_static']
html_logo = None
html_favicon = None

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon configuration (for Google-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'keras': ('https://www.tensorflow.org/api_docs/python', None),
}

# Type hints configuration
autodoc_typehints = 'description'
typehints_defaults = 'comma'

# Additional options
html_use_smartquotes = True
html_show_sphinx = True
html_show_copyright = True
add_module_names = True

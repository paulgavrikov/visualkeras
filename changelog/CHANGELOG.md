# Changelog
All notable changes to this project will be documented in this file, starting from version 0.0.1.

## 0.1.1 (unreleased)

Fourth release: More compatibility and documentation fixes. Not yet in Github repo.

Bug fixes:
- Changed README.md file to use links to images rather than paths - this will allow the images to be displayed on PyPI.
- Added compatability for Keras version 1.x.x

## 0.1.0 (2024-06-30)

Third release: More customization options and downwards compatability with older versions of packages.

Features:
- Draw reversed: Added the option to draw 3D networks from the back of each layer. Good for decoder-like architectures.
- Text callable: Added a callable parameter to determine if and how text is drawn on the network.
- Vertical padding: Added the option to add padding above and below the network.
- Padding left: Added the option to add padding to the left of the network.
- Bigger color palette: Expanded the default color palette to include six more colors.
- Index 2D: Added option to specify certain layers to be drawn in 2D in a 3D network.
- Legend dimensions: Added option to display output dimensions of all layers in the legend.
- Vertical spacing: Added option to specify the vertical spacing between newlines of text.

Bug fixes:
- Downwards compatability with newer versions of Pillow.
- Downards compatability with newer versions of Keras - compatibility with `tensorflow.keras`, `tensorflow.python.keras`, and `keras` in versions greater than 2.
- Removed features of Pillow which require [libraqm](https://github.com/HOST-Oman/libraqm). This was done because libraqm requires custom installation and is not available for automatic installation as a dependency.

Developer changes:
- Switched to cleaner method for downwards compatibility with older versions of packages - now using `hasattr` instead of `try` and `except` blocks.
- Updated documentation and docstrings for better grammar and clarity.

*Note: We recognize this is a very large update and may, as a result, cause dependency conflicts. In future updates, we will adhere to [Semantic Versioning](https://semver.org/)*

## 0.0.2 (2021-04-21)

Second release: Expanded the visualization options.

Features:
- Added graph_view as an alternative visualization file.
- Added the option to have a legend in layer_view.

## 0.0.1 (2020-10-05)

Initial release.
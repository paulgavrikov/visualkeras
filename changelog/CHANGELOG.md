# Changelog
All notable changes to this project will be documented in this file, starting from version 0.0.1.

## 0.1.4 (2024-11-24)
Seventh release: Compatability with newer versions of packages

Bug fixes:
- Added compatability with Keras versions >=3 (Tensorflow >= 2.16) for `graph_view`, see issue [#79](https://github.com/paulgavrikov/visualkeras/issues/79)
- Added compatability with pillow versions >= 10 for legend text, see issue [#77](https://github.com/paulgavrikov/visualkeras/issues/77)

## 0.1.3 (2024-07-27)

Sixth release: Bug fix.

Bug fixes:
- Applied full fix to bug which caused text in legend to be cut off.

Deprecations:
- Began deprecation of `legend_text_spacing_offset` parameter in `layered_view` function. It will be deprecated in a future release.

## 0.1.2 (2024-07-20)
Fifth release: Bug fix.

Bug fixes:
- Made temporary fix to bug which caused text in legend to be cut off (added `legend_text_spacing_offset` parameter `layered_view` function). The fix applied in version 0.1.1 only worked for some cases. This fix should work for all cases.

## 0.1.1 (2024-07-19)

Fourth release: Bug fixes and documentation update.

Bug fixes:
- Made temporary fix to bug which caused text in legend to be cut off.

Documentation:
- Changed README.md file to use links to images rather than paths - this will allow the images to be displayed on PyPI.

## 0.1.0 (2024-06-30)

Third release: More customization options and backwards compatability with older versions of packages.

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
- Backwards compatability with newer versions of Pillow.
- Backwards compatability with newer versions of Keras - compatibility with `tensorflow.keras`, `tensorflow.python.keras`, and `keras` in versions greater than 2.
- Removed features of Pillow which require [libraqm](https://github.com/HOST-Oman/libraqm). This was done because libraqm requires custom installation and is not available for automatic installation as a dependency.

Developer changes:
- Switched to cleaner method for backwards compatibility with older versions of packages - now using `hasattr` instead of `try` and `except` blocks.
- Updated documentation and docstrings for better grammar and clarity.

*Note: We recognize this is a very large update and may, as a result, cause dependency conflicts. In future updates, we will adhere to [Semantic Versioning](https://semver.org/)*

## 0.0.2 (2021-04-21)

Second release: Expanded the visualization options.

Features:
- Added graph_view as an alternative visualization file.
- Added the option to have a legend in layer_view.

## 0.0.1 (2020-10-05)

Initial release.

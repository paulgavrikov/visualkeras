# Changelog

All notable changes to this project will be documented in this file, starting from version 0.0.1.

## Unreleased (to be version 0.3.0)

Tenth release. This update is a major expansion of the library when compared
with `v0.2.0`. It adds two new renderers, a unified high level API, a much
richer styling system, and a broad set of rendering improvements across the
existing visualization modes.

Features and library changes

- Added `visualkeras.show(...)` as a unified entry point for rendering model
  visualizations. It selects a renderer by mode and supports presets, options
  objects, and explicit keyword overrides.
- Added top level exports for `show`, `functional_view`, `lenet_view`,
  `FunctionalOptions`, `LenetOptions`, `FUNCTIONAL_PRESETS`, and
  `LENET_PRESETS` so they can be imported directly from `visualkeras`.
- Added a new `functional_view` renderer for functional Keras models. This
  renderer supports branch aware layout, multi input and multi output models,
  rank based graph placement, routed connectors, disconnected components, and
  optional volumetric boxes.
- Added a new `lenet_view` renderer for classic feature map stack diagrams.
  This mode renders CNN style models as offset map stacks with dedicated
  connection primitives and label controls.
- Added `FunctionalOptions` and `LenetOptions` dataclasses and extended the
  structured options system beyond `LayeredOptions` and `GraphOptions`.
- Added curated presets for functional and LeNet style rendering through
  `FUNCTIONAL_PRESETS` and `LENET_PRESETS`.
- Expanded `LAYERED_TEXT_CALLABLES` so common caption formats can be selected
  by name instead of rewriting annotation callbacks.

Layered renderer

- Added stylesheet style per-layer overrides through the `styles` argument.
  These overrides can be keyed by layer class or layer name and provide
  fine grained control beyond `color_map`.
- Added per-layer image support for layered diagrams.
- Added `image_fit` and `image_axis` controls so images can be resized and
  placed on the intended layer face in volumetric mode.
- Added connector styling with `connector_fill` and `connector_width`.
- Added grouped stage highlighting through `layered_groups`.
- Added group captions for layered group overlays.
- Added automatic canvas expansion when group captions or text would otherwise
  overflow the rendered image.
- Added logo overlays and logo legends through `logo_groups` and
  `logos_legend`.

Graph renderer

- Added stylesheet style per-layer overrides through the `styles` argument.
- Added embedded node images with configurable fit behavior through
  `image_fit`.
- Added `circular_crop` for image based node rendering.
- Added grouped highlights and group captions through `layered_groups`.
- Improved handling of output nodes when `inout_as_tensor=False` so graph
  diagrams avoid duplicated terminal layers.
- Improved graph layout stability around connectors and spacing.

Functional renderer

- Added optional volumetric rendering for functional models.
- Added stylesheet style per-layer overrides through the `styles` argument.
- Added embedded images for functional diagrams.
- Added `image_fit` and `image_axis` controls for image placement on functional
  boxes.
- Added grouped highlights and group captions through `layered_groups`.
- Added logo overlays and legends through `logo_groups` and `logos_legend`.
- Added `simple_text_visualization` for compact text first functional diagrams.
- Added `simple_text_label_mode` to control where labels appear in simple text
  mode.
- Added block and layer collapsing through `collapse_enabled`,
  `collapse_rules`, and `collapse_annotations`.
- Added annotation placement and overflow handling for collapsed blocks and
  group labels.

LeNet style renderer

- Added `lenet_view` as a new renderer for LeNet inspired feature map stack
  diagrams.
- Added support for spatial stacks, vector stacks, and dedicated connector
  types between them.
- Added top and bottom label callables so stack annotations can be customized.
- Added per-side label toggles and padding controls.
- Added per-layer face images through style keys such as `face_image`,
  `face_image_fit`, `face_image_alpha`, and `face_image_inset`.
- Added patch styling controls including `patch_fill`, `patch_outline`,
  `patch_scale`, and `patch_alpha_on_image`.
- Added `max_visual_channels` to cap the number of maps drawn for high channel
  layers.
- Added `seed` for reproducible randomized patch placement.
- Improved connection routing so LeNet style connectors render more cleanly.

Options and presets

- Expanded `LayeredOptions` with support for connector styling, image handling,
  grouped overlays, logo overlays, legends for logos, and stylesheet style
  overrides.
- Expanded `GraphOptions` with support for stylesheet style overrides, image
  fit mode, circular image crops, and grouped overlays.
- Added `FunctionalOptions` to cover functional layout, connector routing,
  sizing, annotation, logo, image, text, and collapse features.
- Added `LenetOptions` to cover LeNet style layout, connector behavior, patch
  styling, label controls, and per-layer style overrides.
- Extended the `preset=` and `options=` workflow to the new renderers and kept
  explicit keyword arguments as the final override layer.

Bug fixes and behavior improvements

- Fixed blank spaces and connector jogging issues in graph renderings.
- Fixed connector jogging issues in volumetric functional renderings.
- Fixed the right and left axis handling in volumetric functional
  visualizations.
- Fixed text rendering issues in graph diagrams.
- Prevented embedded graph images from being drawn on ellipsis markers.
- Improved circular crop quality for graph node images.
- Improved the handling of special characters in rendered text.
- Improved 3D box rotation and dimension handling.
- Improved the placement of captions and labels when vertical or horizontal
  overflow occurs.
- Improved compatibility with modern Keras and TensorFlow internals for layer
  discovery, output name handling, and model graph traversal.

## 0.2.0 (2025-10-13)

Ninth release: Easier preset options and structured API.

Features:

- Added typed `LayeredOptions` and `GraphOptions` dataclasses with preset arguments for `layered_view` and `graph_view`
- Added presets for `text_callable` for common use cases with regards to captions under layers
- Extended `layered_view` and `graph_view` to accept `options=` and `preset=` keyword arguments so that the new dataclasses can be used without breaking backwards compatibility.
- We now raise `RuntimeError` for unsupported model structures to avoid confusing tracebacks.
- Created a `.help()` docstring on the initialization file to allow easier understanding on certain IDEs.

Developer changes:

- Re-exported options and preset defaults from the package root to make the structured API easy to import.
- Added `get_layers` to simplify extracting layers from a Keras/TF model.
- Added significant amount of docstrings throughout the codebase
- Add slight clarifications around formatting for CONTRIBUTING.md
- Misc. improvements to README.md

## 0.1.5 (2025-9-2)

Eighth release

- Custom Scaling Features: Added new functions and options for custom scaling of layer dimension sin visualizations.
- Bug fixes:
  - Fixed InputLayer detection in `layer_utils.py`, as detailed in Issue #82
  - Resolved duplicate output layers and major bugs in graph_view
  - Improved handling of input/output layers when `inout_as_tensor` is false
  - Better docstrings + cleanliness

We also added new documentation with usage examples and associated figures. We have created a `CONTRIBUTING.md` file for new contributors to reference.

## 0.1.4 (2024-11-24)

Seventh release: Compatibility with newer versions of packages

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

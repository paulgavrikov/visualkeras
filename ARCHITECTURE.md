# Architecture Guide

This document gives a short overview of how visualkeras is structured. It is
meant for contributors who want to understand where code is placed and how changes
should fit into the existing design.

## High Level Structure

The package is organized around renderers, shared helpers, and configuration
objects.

- `visualkeras/layered.py`
  Implements `layered_view`, which is the layered CNN style renderer.
- `visualkeras/graph.py`
  Implements `graph_view`, which focuses on topology and node connections.
- `visualkeras/functional.py`
  Implements `functional_view`, which combines graph structure with box based
  rendering.
- `visualkeras/lenet.py`
  Implements `lenet_view`, which renders feature map stack diagrams.
- `visualkeras/show.py`
  Implements `show(...)`, the unified high level entry point.
- `visualkeras/options.py`
  Defines options dataclasses, presets, and built in text callables.
- `visualkeras/layer_utils.py`
  Holds helpers for model traversal, graph extraction, and tensor dimension
  handling.
- `visualkeras/utils.py`
  Holds shared drawing, layout, image, and style helpers used by more than one
  renderer.
- `visualkeras/__init__.py`
  Re exports the public API from the package root.

## Public API Surface

The main user facing functions are

- `visualkeras.show(...)`
- `visualkeras.layered_view(...)`
- `visualkeras.graph_view(...)`
- `visualkeras.functional_view(...)`
- `visualkeras.lenet_view(...)`

The main user facing configuration objects are

- `LayeredOptions`
- `GraphOptions`
- `FunctionalOptions`
- `LenetOptions`

When you change a public function, option, or preset, you should also check the
API docs, examples, and tests.

## Renderer Flow

Each renderer follows the same broad pattern.

1. Accept a model plus renderer specific options.
2. Resolve defaults, presets, and `options=` values.
3. Apply explicit keyword arguments as the final override layer.
4. Inspect the model and derive the data needed for layout.
5. Build drawing primitives and render them into a `PIL.Image`.
6. Save the image if `to_file` is provided.
7. Return the final `PIL.Image`.

That shared flow is important. New features should fit into it rather than
create a separate control path unless there is a strong reason.

## Options and Presets

The structured configuration system lives in `visualkeras/options.py`.

Each renderer has a matching options dataclass. Those classes mirror the
keyword arguments accepted by the renderer. They do not implement a separate
configuration model. They are just a reusable container for renderer
arguments.

Each renderer also has a preset dictionary. Presets are named starting points.
They are not fixed modes.

The precedence order is

1. renderer defaults
2. preset values
3. `options=` values
4. explicit keyword arguments

When adding a new renderer argument, keep this system aligned.

That usually means

- add the new keyword argument to the renderer
- add the matching field to the options dataclass
- decide whether any presets should set it
- update `show(...)` support if needed
- add tests for the new behavior

## Shared Helpers

Two helper modules are used across the renderers.

### `layer_utils.py`

This module is responsible for model and layer introspection.

Typical responsibilities include

- getting a stable layer list from a model
- finding input and output layers
- building adjacency data
- grouping layers into hierarchy levels
- extracting shapes
- converting shapes into rendered dimensions

If a change is mainly about understanding model structure, it probably belongs
here.

### `utils.py`

This module is responsible for drawing and layout helpers that are not tied to
one renderer.

Typical responsibilities include

- style resolution
- color helpers
- geometric primitives
- image fitting and transforms
- shared layout helpers
- logo and legend helpers

If the same drawing logic would otherwise be copied into more than one
renderer, it probably belongs here.

## Renderer Specific Responsibilities

The renderers are not identical. Each one owns its own layout model.

### Layered

`layered_view` is best for mostly sequential models. It focuses on left to
right progression and tensor shape changes.

This module owns

- layered box placement
- funnels and connectors between adjacent layers
- legends for layer types
- grouped overlays, images, and logos in layered diagrams

### Graph

`graph_view` focuses on connectivity. It is a node based renderer and is more
abstract than layered mode.

This module owns

- graph node placement
- neuron and tensor style node rendering
- edge drawing
- graph specific images and grouped overlays

### Functional

`functional_view` handles richer graph structure while still keeping a box
based architectural look.

This module owns

- graph extraction for functional layouts
- rank assignment and ordering
- long edge handling
- component layout
- collapse rules and collapse annotations
- simple text visualization mode

### LeNet

`lenet_view` focuses on classic feature map stack figures.

This module owns

- stack based rendering of feature maps
- stack to stack connector primitives
- patch overlays
- top and bottom label logic
- per layer face images in LeNet style diagrams

## How `show(...)` Fits In

`show(...)` is a dispatch layer. It should stay thin.

Its main jobs are

- normalize the requested mode
- validate that the right options type is being used
- forward the call to the selected renderer

It should not duplicate renderer logic. If a behavior is renderer specific, the
implementation should live in the renderer module, not in `show.py`.

## Package Design Rules

These rules help keep the codebase easier to maintain.

- Keep renderer specific layout logic inside the renderer module.
- Move reusable drawing or model traversal logic into shared helpers.
- Keep options classes in sync with renderer signatures.
- Prefer extending the existing preset and options system over adding one off
  configuration paths.
- Keep `show(...)` thin.
- Keep public behavior covered by tests.

## Testing Strategy

Tests are split by feature area and by level.

- unit tests for helpers and validation
- integration tests for full renderer behavior
- end to end tests for file output and larger workflows

See `tests/README.md` for the detailed testing guide.

As a rule

- helper changes should add unit tests
- renderer changes should add integration tests
- public API changes should usually add `test_show.py` or `test_options.py`
  coverage when relevant

## Adding a New Feature

If you are adding a new feature, this order usually works well.

1. Decide whether the behavior is renderer specific or shared.
2. Add the code in the renderer or shared helper module.
3. Add or update options and presets if the feature is configurable.
4. Update `show(...)` only if the public high level API needs to expose the
   feature.
5. Add tests in the right place.
6. Update docs and docstrings for public changes.

## Adding a New Renderer

If the project ever adds another renderer, try to match the current pattern.

That means

- create a dedicated renderer module
- define a matching options dataclass and presets
- export the renderer from the package root
- add a mode in `show(...)`
- add renderer tests, options tests, and docs

## Where to Start

If you are new to the codebase, a good reading order is

1. `visualkeras/show.py`
2. `visualkeras/options.py`
3. one renderer module that matches the feature you want to change
4. `visualkeras/layer_utils.py` or `visualkeras/utils.py` if the change is more
   general
5. `tests/README.md` and the matching test files

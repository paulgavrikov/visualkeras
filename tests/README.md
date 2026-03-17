# Test Guide

This repository uses `pytest` for both unit tests and integration tests.

The test suite is split by feature area. Each file has a clear role. If you
are adding new functionality, the first step is to place the new test in the
right file or create a new file that follows the same structure.

## Goals of the Test Suite

The tests are meant to do four things

- protect the public API
- catch rendering regressions
- verify tricky helper logic in isolation
- make it easier to add features without breaking older behavior

Most changes should add tests at one of these levels

- unit level for helper logic and validation
- integration level for full renderer behavior
- end to end level for file output or combined workflows

## Test Layout

The suite is organized by module and by test type.

### Public API and configuration

- `tests/test_show.py`
  Tests the high level `visualkeras.show(...)` API. Use this file when a change
  affects mode dispatch, option validation, or preset handling at the unified
  entry point.
- `tests/test_options.py`
  Tests options dataclasses, preset dictionaries, and built in text callables.
  Use this file when you add new option fields, presets, or option related
  helper behavior.

### Shared helpers

- `tests/test_layer_utils_helpers.py`
  Tests graph and shape helpers in `visualkeras.layer_utils`. Use this file for
  layer discovery, adjacency logic, input and output detection, shape handling,
  and dimension calculation.
- `tests/test_utils_helpers.py`
  Tests shared helpers in `visualkeras.utils`. Use this file for style
  resolution, color handling, image layout, affine transforms, and low level
  drawing helpers.

### Layered renderer

- `tests/test_layered_helpers.py`
  Tests internal helper behavior in `visualkeras.layered`. Use this file for
  isolated logic that does not need a full model render.
- `tests/test_layered_renderer.py`
  Integration and regression tests for `layered_view`. Use this file when a
  change affects rendered output, options behavior, or image metrics.

### Graph renderer

- `tests/test_graph_helpers.py`
  Tests internal helper behavior in `visualkeras.graph`.
- `tests/test_graph_renderer.py`
  Integration and regression tests for `graph_view`.

### Functional renderer

- `tests/test_functional_validation.py`
  Tests validation of collapse rules and related user input handling.
- `tests/test_functional_helpers.py`
  Tests internal graph, layout, and annotation helpers in
  `visualkeras.functional`.
- `tests/test_functional_renderer.py`
  Integration tests for `functional_view`.
- `tests/test_functional_internal_integration.py`
  Integration tests for harder to reach internal branches in functional graph
  building and rendering. Use this file when the behavior depends on a full
  model but still targets internal paths rather than public smoke behavior.

### LeNet style renderer

- `tests/test_lenet_helpers.py`
  Tests internal helper logic in `visualkeras.lenet`.
- `tests/test_lenet_renderer.py`
  Integration tests for `lenet_view`.

### Cross renderer and end to end coverage

- `tests/test_renderer_advanced_paths.py`
  Integration tests for advanced branches that cut across renderers. Examples
  include images, groups, legends, and collapse features.
- `tests/test_end_to_end_file_output.py`
  End to end tests that verify renderers write output files correctly.

### Baseline data and scripts

- `tests/data/renderer_metrics.json`
  Stored metrics for renderer regression checks.
- `tests/renderer_baselines/generate_renderer_references.py`
  Script used to regenerate renderer reference data when output changes
  intentionally.

## Choosing Where a New Test Should Go

Use these rules when deciding where to place a new test.

### Add a unit test when

- the change is in a helper function
- the behavior can be checked without building a full image
- the logic has edge cases or validation branches
- a failure should point to one small code path

Examples

- shape normalization
- style resolution
- option conversion
- collapse rule validation

### Add an integration test when

- the change affects a public renderer
- the behavior depends on a built Keras model
- several helpers work together to produce the result
- you want to verify that a real render completes and has stable properties

Examples

- new renderer arguments
- image output size changes
- grouped overlays
- embedded images
- new label behavior

### Add an end to end test when

- the change affects file writing
- the change spans `show(...)` plus a renderer
- the behavior matters at the final user workflow level

## Writing New Tests

### General guidance

- Keep tests small and direct.
- Prefer one behavior per test.
- Use descriptive test names that explain what should happen.
- Build the smallest model that still exercises the feature.
- Assert stable properties. Avoid brittle checks unless strict regression
  coverage is the point of the test.

Good stable assertions often include

- image type
- image width and height
- file existence
- validated option values
- counts of nodes, edges, or layers
- label text
- normalized rule output

### When to use mocks or monkeypatch

Use mocks or `monkeypatch` when the code path is mostly about dispatch or
helper behavior and a full render would add noise.

Good examples in this repository include

- `tests/test_show.py`
- parts of `tests/test_graph_helpers.py`

### When to build a real model

Build a real Keras model when the feature depends on layer graph structure,
output shapes, renderer layout, or image generation.

Good examples in this repository include

- `tests/test_layered_renderer.py`
- `tests/test_graph_renderer.py`
- `tests/test_functional_renderer.py`
- `tests/test_lenet_renderer.py`

### Keep test models simple

Use the smallest model that still reaches the branch you need. Most tests in
this repository use very small Sequential or Functional models for that reason.

## Running Tests

Install dependencies with

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Some integration tests need TensorFlow and `aggdraw`.

Run the full suite with

```bash
python -m pytest -v
```

Run only unit tests with

```bash
python -m pytest -m "not integration" -v
```

Run only integration tests with

```bash
python -m pytest -m integration -v
```

Run one file with

```bash
python -m pytest tests/test_functional_renderer.py -v
```

Run one test with

```bash
python -m pytest tests/test_functional_renderer.py::test_functional_view_smoke -v
```

## Integration Test Notes

Many renderer tests use this pattern

```python
tf = pytest.importorskip("tensorflow")
pytest.importorskip("aggdraw")
pytestmark = pytest.mark.integration
```

This means

- the test is skipped if TensorFlow is not installed
- the test is skipped if `aggdraw` is not installed
- the test is marked as integration so it can be excluded from fast local runs

If you add a new renderer level test, follow the same pattern unless there is a
good reason not to.

## Renderer Regression Data

Some renderer tests compare generated output against stored metrics in
`tests/data/renderer_metrics.json`.

Use that approach when you need stronger regression coverage but do not want
the brittleness of pixel perfect image snapshots.

If renderer behavior changes intentionally, regenerate the reference data with

```bash
python tests/renderer_baselines/generate_renderer_references.py
```

Make sure the new reference data is intentional before committing it.

## Coverage

Generate terminal, XML, and HTML coverage reports with

```bash
python -m pytest --cov=visualkeras --cov-report=term-missing --cov-report=xml --cov-report=html
```

Open `htmlcov/index.html` to inspect file level coverage locally.

## Suggested Workflow for New Features

If you add a new library feature, this order usually works well

1. Add or update a focused unit test for the smallest helper or validation
   logic.
2. Add an integration test for the public renderer or API behavior.
3. Add an end to end or regression style test if the feature affects file
   output or a complex rendering path.
4. Update reference metrics only if the rendered output changed intentionally.

## Common Patterns in This Repository

### Options and presets

If you add a new option or preset

- update `tests/test_options.py`
- add renderer tests if the option changes visible behavior
- add `show(...)` tests if the option affects unified API behavior

### New renderer arguments

If you add a new argument to `layered_view`, `graph_view`, `functional_view`,
or `lenet_view`

- add an integration test for the renderer
- add an options test if the argument is mirrored in an options dataclass
- add a `show(...)` test if the argument should flow through the unified API

### Validation logic

If you add or change validation rules

- test the valid cases
- test the invalid cases
- assert the error type and a useful part of the message when practical

### Shared utilities

If you change low level utility functions, prefer direct unit tests with simple
inputs first. Only add renderer tests if the utility change affects visible
behavior at the image level.

## What Good Test Additions Look Like

A good test addition usually has these qualities

- it is placed in the right file
- it has a narrow purpose
- it uses the smallest useful model or input
- it checks stable behavior
- it helps a future contributor understand how a feature is expected to work

If you are unsure where a test belongs, match the closest existing file and
follow its style.

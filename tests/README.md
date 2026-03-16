# Test Suite

This repository uses `pytest` for both unit and integration tests.

## Layout

- `tests/test_options.py`: option dataclasses and presets.
- `tests/test_show.py`: high-level `visualkeras.show(...)` dispatch and validation.
- `tests/test_functional_validation.py`: collapse-rule validation for `functional_view`.
- `tests/test_functional_helpers.py`: unit tests for functional graph/collapse/layout helper functions.
- `tests/test_lenet_helpers.py`: deterministic helper behavior for LeNet rendering internals.
- `tests/test_layer_utils_helpers.py`: unit tests for layer graph and dimension helper functions.
- `tests/test_layered_helpers.py`: unit tests for layered renderer helper functions.
- `tests/test_graph_helpers.py`: unit tests for graph renderer helper functions.
- `tests/test_utils_helpers.py`: unit tests for shared geometry/layout/image helper functions.
- `tests/test_functional_renderer.py`: functional renderer integration smoke tests.
- `tests/test_functional_internal_integration.py`: integration tests for internal functional graph-building branches.
- `tests/test_lenet_renderer.py`: LeNet renderer integration smoke tests.
- `tests/test_end_to_end_file_output.py`: end-to-end tests verifying renderer output files are written.
- `tests/test_renderer_advanced_paths.py`: integration tests for advanced renderer branches (images, groups, collapse, legends).
- `tests/test_layered_renderer.py`: layered renderer integration/image-regression checks.
- `tests/test_graph_renderer.py`: graph renderer integration/image-regression checks.

## Running Tests

Install dependencies:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Run all tests:

```bash
python -m pytest
```

Run unit tests only (skip integration):

```bash
python -m pytest -m "not integration"
```

Run integration renderer tests only:

```bash
python -m pytest -m integration
```

## Coverage

Generate terminal + XML + HTML coverage reports:

```bash
python -m pytest --cov=visualkeras --cov-report=term-missing --cov-report=xml --cov-report=html
```

Open `htmlcov/index.html` to inspect per-file coverage.

## Renderer Baselines

If renderer behavior changes intentionally, regenerate reference images/metrics:

```bash
python tests/renderer_baselines/generate_renderer_references.py
```

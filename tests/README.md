# visualkeras tests

This folder contains the test suite for visualkeras. Tests cover utilities and key rendering paths to ensure functions run without errors across both `tf.keras` and standalone `keras` backends.

## Requirements

- Python 3
- TensorFlow and Keras are required for the full suite:

```bash
pip install tensorflow keras
```

Note: Some tests construct small models, no GPU is required. Tests avoid file I/O and network access.

## How to run

Run the whole suite:

```bash
python -m pytest -v
```

Run a subset by keyword:

```bash
python -m pytest -k utils -v
```

## Layout and fixtures

- `conftest.py` defines fixtures that generate both Functional and Sequential models using two libraries:
  - `tf.keras` (TensorFlow bundled Keras)
  - `keras` (standalone Keras)

  The `pytest_generate_tests` hook parametrizes tests so each one runs against both backends and model types. This ensures broad API compatibility.

- Individual test files focus on different areas:
  - `test_utils.py`: Assertions for color parsing, shape math, and helpers
  - `test_layer_utils.py`: Graph and layer traversal utilities and identification helpers
  - `test_graph.py` / `test_layered.py`: Smoke tests that rendering functions execute without exceptions
  - `test_graph_inout_flag.py`: Behavioral check that a rendering flag affects image dimensions as expected

## What we assert

- Utilities: Exact functional behavior via equality checks
- Graph/layer utils: Correct layer discovery and adjacency relationships
- Rendering paths: "Smoke" (no exceptions) plus coarse properties (e.g., image width/height) rather than pixel-perfect comparisons

Why not pixel-perfect? Rendering depends on fonts, platform differences, and upstream library versions. Coarse invariants make tests stable across environments.

## Writing new tests

When adding or changing features:

1) Cover new public functions with at least a smoke test ensuring they run without exceptions on small models.
2) Prefer assertions on stable, semantic properties (sizes, counts, names) instead of exact pixels.
3) Keep models tiny and deterministic; avoid training, randomness, disk, and network.
4) If a feature has toggles/flags, add at least one assertion that the flag changes a measurable property (like in `test_graph_inout_flag.py`).
5) Place general-purpose fixtures in `conftest.py`; keep per-test setup local to the test file.

Example pattern for a rendering option:

```python
img_a = graph_view(model, option=True)
img_b = graph_view(model, option=False)
assert img_a is not None and img_b is not None
assert img_a.width != img_b.width  # or other stable property
```

If you introduce heavy or optional dependencies, consider marking tests and documenting how to skip them. Currently, the suite assumes TensorFlow/Keras are installed.


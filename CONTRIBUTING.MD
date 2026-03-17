# Contributing to visualkeras

Thank you for contributing to visualkeras.

This guide explains how to report issues, set up a local environment, and open
pull requests. Keep changes focused. Clear reports and small PRs are much
easier to review.

## Before You Start

Please read the main project docs first.

- [README](https://github.com/paulgavrikov/visualkeras/blob/master/README.md)
- [Documentation site](https://visualkeras.readthedocs.io/)
- [Open issues](https://github.com/paulgavrikov/visualkeras/issues)

For larger changes, open an issue first. That helps avoid duplicate work and
makes sure the change fits the project.

## Questions

If you have a question, search the existing issues first. A similar question
may already have an answer.

If you still need help, open an issue and include enough context for someone
else to follow the problem. Version details are often useful. Examples include
Python, TensorFlow, Keras, and visualkeras.

## Reporting Bugs

Use the GitHub bug report template when possible.

Please include these details

- What happened
- What you expected to happen
- A small example that reproduces the problem
- Steps to reproduce it
- Version details and platform details
- Screenshots or output when they help explain the issue

Try to reduce the example as much as you can. Small reproductions are easier to
debug.

## Suggesting Features

Use the GitHub feature request template when possible.

Please include these details

- The problem you want to solve
- The change you want to see
- Any alternatives you considered
- Extra context that would help with review

## Development Setup

Clone the repository and create a virtual environment.

```bash
git clone https://github.com/paulgavrikov/visualkeras.git
cd visualkeras
python -m venv .venv
source .venv/bin/activate
```

On Windows use

```bash
.venv\Scripts\activate
```

Install the package in editable mode and install the dev dependencies.

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Some tests need TensorFlow.

```bash
pip install tensorflow==2.17.0
```

There is also a standalone Keras smoke check.

```bash
pip install "keras>=3,<4" tensorflow==2.17.0
```

Current CI coverage is based on these environments

- Python 3.9 through 3.12 for unit tests
- Python 3.11 with TensorFlow 2.17 for the full test suite
- Python 3.11 with standalone Keras and TensorFlow backend for the smoke check

## Running Tests

Run the fast test suite with

```bash
python -m pytest -m "not integration" -v
```

Run the full test suite with

```bash
python -m pytest -v
```

Generate coverage locally with

```bash
python -m pytest --cov=visualkeras --cov-report=term-missing --cov-report=xml --cov-report=html
```

If TensorFlow is not available, some tests will not run. In that case, run the
tests you can and explain the gap in your PR.

For more test details, see [tests/README.md](https://github.com/paulgavrikov/visualkeras/blob/master/tests/README.md).

## Linting and Formatting

Run these checks before opening a PR.

```bash
ruff check .
black .
isort .
```

If you use pre commit hooks, install them with

```bash
pre-commit install
```

## Building Documentation

If your change affects docs, docstrings, or public APIs, build the docs locally.

```bash
python -m sphinx -b html -W --keep-going -d docs/build/doctrees docs/source docs/build/html
```

The `-W` flag turns warnings into errors. That is useful before opening a PR.

## Code Style

Follow PEP 8. Keep functions small when practical. Prefer clear names over
clever code.

Please add or update tests for behavior changes when it makes sense. For visual
changes, stable checks on image size, object counts, labels, or related output
are often better than brittle full image assertions.

## Docstrings

Docstrings in this project should be clear and concise. For public APIs, prefer
NumPy style docstrings that work well with Sphinx and Napoleon.

A good docstring usually includes

- A short summary
- Parameters when needed
- Returns when needed
- Raises when needed
- Notes when behavior or precedence needs extra explanation

Keep docstrings practical. Explain what a setting does and when someone would
use it.

## Pull Requests

Open focused PRs. Small changes move faster.

Before you open a PR, please make sure that you have done the following

- Run the relevant tests
- Built the docs if your change affects docs or public APIs
- Updated docs or docstrings when needed
- Checked whether the change could break existing behavior

GitHub will apply the PR template automatically. Fill it out with enough detail
for review.

A good PR should include

- A short summary of the change
- The related issue, if there is one
- Notes on testing
- Notes on documentation updates
- Screenshots or example output when helpful

If you could not run a relevant check, say so clearly.

## Documentation Contributions

Documentation fixes are welcome. That includes small wording changes, example
fixes, API reference updates, and larger tutorial improvements.

Please make sure that

- Code examples are correct
- Example output still matches the code
- Links still work
- Formatting is consistent with nearby pages

If you change public APIs, update both the docs and the docstrings when needed.

## Reference Links

- [PEP 8](https://peps.python.org/pep-0008/)
- [Sphinx Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Project issues](https://github.com/paulgavrikov/visualkeras/issues)

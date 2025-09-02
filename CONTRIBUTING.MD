# Contributing to visualkeras
First off, thank you for your interest in contributing to visualkeras! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for the maintainer and smooth out the experience for all involved. The community looks forward to your contributions!

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would greatly appreciate:
> - Star the project on GitHub
> - [Cite the repository in your work](https://github.com/paulgavrikov/visualkeras/blob/master/CITATION.cff)
> - Tweet about it
> - Refer this project in your project's readme
> - Share it with your friends

## Table of Contents

- [I Have a Question](#i-have-a-question)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features and Improvements](#suggesting-features-and-improvements)
- [I Want To Contribute](#i-want-to-contribute)
- [Your First Code Contribution](#your-first-code-contribution)
- [Improving The Documentation](#improving-the-documentation)

## I Have a Question

> If you want to ask a question, we assume that you have already read the [README](https://github.com/paulgavrikov/visualkeras/blob/master/README.md) and its FAQ section.

Before you ask a question, it is best to search for [existing issues](https://github.com/paulgavrikov/visualkeras/issues?q=is%3Aissue) that might help you. In case you have found a suitable issue and still need clarification, you can write your question and @Soontosh in the comments of that issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:
- Open an [Issue](https://github.com/paulgavrikov/visualkeras/issues).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (Python, Keras, visualkeras, etc.), depending on what seems relevant.

## Reporting Bugs

If you come across a bug while using visualkeras, please open a new issue. You can use our *bug report template* to help structure your issue. In your issue, include the following details:
- A clear description of the bug.
- Minimal code and steps to reproduce the bug
- Expected behavior
- If applicable, a screenshot or example of the bug.
- Environment (OS, Python version, visualkeras version or commit hash, Keras version, etc.)
- Any additional context that might help us understand the issue.


## Suggesting Features and Improvements

To suggest new features or suggest improvements to existing features, please open a new issue. You can use our *feature request template* to help structure your suggestion. In your issue, include the following details:
- Whether or not your feature reqeust is related to a problem
- A description of the solution you'd like to see and what you want to happen
- A description of alternatives you've considered
- Any additional context that might help us understand the feature or improvement

## I Want To Contribute

If you want to contribute, please take a look at [existing issues](https://github.com/paulgavrikov/visualkeras/issues) and see if you can help with any of them, in which case you can comment on the issue to express your interest. If you have a new idea, please open an issue to discuss it first. This helps us avoid duplicate work and ensures that your contribution aligns with the project's goals.

We also have a list of improvements that we would like to see in the project, which you can find in the project [README](https://github.com/paulgavrikov/visualkeras/blob/master/README.md).


## Your First Code Contribution

### Development setup

Clone the repository and set up a virtual environment (recommended):

```bash
git clone https://github.com/paulgavrikov/visualkeras.git
cd visualkeras
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install the package in editable mode and recommended dev tools:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Some tests and examples require TensorFlow/Keras. To run all the tests:

```bash
pip install tensorflow keras  # or tensorflow-cpu for CPU-only
```

Newer Python versions almost always work with visualkeras, particularly because our underlying packages are constantly being updated. However, we only guarantee functionality for 3.6-3.8.

### Running tests

We use `pytest` for tests located under `tests/`.

```bash
python -m pytest -v
```

Notes:
- Several tests construct small Keras models; TensorFlow/Keras must be installed for these to run.
- If you cannot install TensorFlow locally, run only non-TF tests or skip them selectively.

For an overview of the test layout, fixtures, and expected assertions, see `tests/README.md`. We prefer that major new changes include at least a smoke test and, where feasible, assertions on stable properties (e.g., sizes, counts, or names).

### Linting and formatting

Please follow PEP 8 and keep imports organized. We recommend the following tools (installed via `requirements-dev.txt`):

```bash
# Code style & static checks
ruff check .

# Auto-format code
black .

# Sort imports
isort .
```

Optionally, you can install our recommended git hooks to run checks before commits:

```bash
pre-commit install
```

Before opening a PR, we recommend you:
- Run `black .` (or ensure your editor auto-formats on save)
- Run `ruff check .` and address warnings
- Run `python -m pytest -v`

Don't get too stressed over linting issues, you should focus on the functionality of your code first. We can help with fixing linting issues throughout the PR review process.

### Docstrings and documentation style

Docstrings in this project use Google-style conventions (compatible with Sphinx via the Napoleon extension). Please use concise, descriptive docstrings with `Args:`, `Returns:`, and `Raises:` where appropriate.

References for contributors:

- PEP 8: https://peps.python.org/pep-0008/
- Google Python Style Guide (Docstrings): https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- Sphinx Napoleon (Google/NumPy docstrings): https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

### Submitting a pull request

After making your changes and running the tests, you can submit a pull request. Please ensure that your code has clear commits and adds/updates tests for new features or bug fixes.

When submitting a pull request, please include:

- A clear description of the changes you made
- The issue number your pull request addresses, if applicable
- Any relevant documentation updates
- Screenshots or examples of the changes, if applicable
- Any additional context that might help the maintainers review your changes

@Soontosh or @paulgavrikov will review your pull request, rerun the tests on their end, and provide feedback. Please be patient, as it may take some time to get to your contribution.

## Improving The Documentation

If you find any issues in the documentation or have suggestions for improvements, please open an issue or submit a pull request with your changes. Documentation contributions are always welcome!

We accept small fixes, such as typos, grammar corrections, and formatting improvements, as well as larger changes that improve the overall clarity and usability of the documentation.

When submitting documentation changes, please ensure that:
- The changes are clear and concise.
- The formatting is consistent with the rest of the documentation.
- Any code examples are correct and tested.
- Any new sections or changes are well-organized and easy to follow.

If you have any questions about the documentation or need help with a specific section, feel free to open an issue or reach out to @Soontosh.
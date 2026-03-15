"""High-level convenience API for rendering model visualizations.

This module exposes :func:`visualkeras.show`, a single entry point that selects
between renderers, applies presets or option bundles, and still allows callers
to override individual parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from PIL import Image

from .layered import layered_view
from .graph import graph_view
from .functional import functional_view
from .lenet import lenet_view
from .options import LayeredOptions, GraphOptions, FunctionalOptions, LenetOptions


LayeredOptionsType = Union[LayeredOptions, Mapping[str, Any], None]
GraphOptionsType = Union[GraphOptions, Mapping[str, Any], None]
FunctionalOptionsType = Union[FunctionalOptions, Mapping[str, Any], None]
LenetOptionsType = Union[LenetOptions, Mapping[str, Any], None]
ShowOptionsType = Union[LayeredOptionsType, GraphOptionsType, FunctionalOptionsType, LenetOptionsType]


def _canonical_mode(mode_norm: str) -> Optional[str]:
    if mode_norm in {"layered", "layers"}:
        return "layered"
    if mode_norm in {"graph", "graph_view"}:
        return "graph"
    if mode_norm in {"functional", "func"}:
        return "functional"
    if mode_norm in {"lenet", "lenet_style"}:
        return "lenet"
    return None


def _validate_options_for_mode(mode: str, options: ShowOptionsType) -> None:
    if options is None or isinstance(options, Mapping):
        return

    expected_by_mode = {
        "layered": LayeredOptions,
        "graph": GraphOptions,
        "functional": FunctionalOptions,
        "lenet": LenetOptions,
    }
    expected_type = expected_by_mode.get(mode)
    if expected_type is None:
        return
    if not isinstance(options, expected_type):
        raise TypeError(
            "Invalid options type for mode '{mode}': expected {expected}, got {got}. "
            "Pass a matching options dataclass, a mapping, or None.".format(
                mode=mode,
                expected=expected_type.__name__,
                got=type(options).__name__,
            )
        )


def show(
    model: Any,
    *,
    mode: str = "layered",
    preset: Optional[str] = None,
    options: ShowOptionsType = None,
    **overrides: Any,
) -> Image.Image:
    """Render a model visualization using a selected renderer.

    Parameters
    ----------
    model :
        Keras/TensorFlow model instance.
    mode :
        Which renderer to use. One of: ``"layered"``, ``"graph"``, ``"functional"``, ``"lenet"``.
    preset :
        Optional preset name for the selected renderer.
    options :
        Options bundle (e.g. :class:`LayeredOptions`, :class:`GraphOptions`, :class:`FunctionalOptions`,
        :class:`LenetOptions`) or a plain mapping of keyword arguments. These values apply after presets
        and before explicit overrides.
    **overrides :
        Keyword arguments forwarded directly to the selected renderer.

    Returns
    -------
    PIL.Image.Image
        Rendered visualization.
    """
    mode_norm = (mode or "").lower().strip()
    canonical_mode = _canonical_mode(mode_norm)
    if canonical_mode is None:
        raise ValueError(
            "Unknown mode '{mode}'. Expected one of: layered, graph, functional, lenet.".format(mode=mode)
        )

    _validate_options_for_mode(canonical_mode, options)

    if canonical_mode == "layered":
        return layered_view(model, preset=preset, options=options, **overrides)
    if canonical_mode == "graph":
        return graph_view(model, preset=preset, options=options, **overrides)
    if canonical_mode == "functional":
        return functional_view(model, preset=preset, options=options, **overrides)
    if canonical_mode == "lenet":
        return lenet_view(model, preset=preset, options=options, **overrides)
    raise AssertionError("unreachable")

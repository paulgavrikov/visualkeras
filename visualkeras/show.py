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
    if mode_norm in {"layered", "layers"}:
        return layered_view(model, preset=preset, options=options, **overrides)
    if mode_norm in {"graph", "graph_view"}:
        return graph_view(model, preset=preset, options=options, **overrides)
    if mode_norm in {"functional", "func"}:
        return functional_view(model, preset=preset, options=options, **overrides)
    if mode_norm in {"lenet", "lenet_style"}:
        return lenet_view(model, preset=preset, options=options, **overrides)

    raise ValueError(
        "Unknown mode '{mode}'. Expected one of: layered, graph, functional, lenet.".format(mode=mode)
    )

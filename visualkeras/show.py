"""High-level convenience API for rendering model visualizations.

This module exposes :func:`visualkeras.show`, a single entry point that selects
between layered and graph renderers, applies presets or option bundles, and
still allows callers to override individual parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Union

from PIL import Image

from .layered import layered_view
from .graph import graph_view
from .options import (
    GraphOptions,
    LayeredOptions,
    GRAPH_PRESETS,
    LAYERED_PRESETS,
    LAYERED_TEXT_CALLABLES,
)

_LayeredOptionsType = Union[LayeredOptions, Mapping[str, Any], None]
_GraphOptionsType = Union[GraphOptions, Mapping[str, Any], None]


def show(
    model,
    mode: str = "layered",
    *,
    preset: Union[str, None] = None,
    options: Union[LayeredOptions, GraphOptions, Mapping[str, Any], None] = None,
    **overrides: Any,
) -> Image.Image:
    """Render a model visualization with optional presets or option bundles.

    Parameters
    ----------
    model :
        A Keras or TensorFlow model instance to visualize.
    mode : {"layered", "graph"}, default "layered"
        Selects which renderer to use. ``"layered"`` produces the classic CNN
        block diagram, while ``"graph"`` builds a node/edge plot.
    preset : str, optional
        Name of a preset from :data:`visualkeras.LAYERED_PRESETS` or
        :data:`visualkeras.GRAPH_PRESETS`. Presets provide curated defaults for
        common styles. When supplied, their values are merged before applying
        any explicit overrides.
    options : LayeredOptions or GraphOptions or Mapping, optional
        A configuration bundle generated via :class:`LayeredOptions`,
        :class:`GraphOptions`, or a plain mapping that follows the same field
        names. These values apply after presets and before ``**overrides``.
    **overrides :
        Individual keyword arguments forwarded to the underlying renderer.
        These take precedence over both presets and option bundles.

    Returns
    -------
    PIL.Image.Image
        The rendered visualization image.

    Examples
    --------
    Quick layered view with defaults::

        img = visualkeras.show(model)

    Graph view using a built-in preset and a custom background::

        img = visualkeras.show(
            model,
            mode=\"graph\",
            preset=\"detailed\",
            background_fill=\"#f7f7f7\"
        )

    Layered view with an options dataclass and a caption helper::

        opts = visualkeras.LayeredOptions(legend=True)
        img = visualkeras.show(
            model,
            options=opts,
            text_callable=\"name_shape\"
        )
    """

    mode = mode.lower()
    if mode not in {"layered", "graph"}:
        raise ValueError("mode must be 'layered' or 'graph'")

    if mode == "layered":
        params: MutableMapping[str, Any] = {}
        if preset is not None:
            try:
                params.update(LAYERED_PRESETS[preset].to_kwargs())
            except KeyError as exc:  # pragma: no cover - defensive
                available = ", ".join(sorted(LAYERED_PRESETS))
                raise ValueError(
                    f"Unknown layered preset '{preset}'. "
                    f"Available presets: {available}"
                ) from exc

        if options is not None:
            params.update(_coerce_layered_options(options))

        params.update(overrides)

        text_callable = params.get("text_callable")
        if isinstance(text_callable, str):
            try:
                params["text_callable"] = LAYERED_TEXT_CALLABLES[text_callable]
            except KeyError as exc:
                available = ", ".join(sorted(LAYERED_TEXT_CALLABLES))
                raise ValueError(
                    f"Unknown text callable preset '{text_callable}'. "
                    f"Available presets: {available}"
                ) from exc

        return layered_view(model, **params)

    # mode == "graph"
    params = {}
    if preset is not None:
        try:
            params.update(GRAPH_PRESETS[preset].to_kwargs())
        except KeyError as exc:  # pragma: no cover - defensive
            available = ", ".join(sorted(GRAPH_PRESETS))
            raise ValueError(
                f"Unknown graph preset '{preset}'. Available presets: {available}"
            ) from exc

    if options is not None:
        params.update(_coerce_graph_options(options))

    params.update(overrides)

    return graph_view(model, **params)


def _coerce_layered_options(
    options: _LayeredOptionsType,
) -> MutableMapping[str, Any]:
    if options is None:
        return {}
    if isinstance(options, LayeredOptions):
        return options.to_kwargs()
    if isinstance(options, Mapping):
        return dict(options)
    raise TypeError(
        "Layered visualizations require a LayeredOptions instance or a mapping."
    )


def _coerce_graph_options(options: _GraphOptionsType) -> MutableMapping[str, Any]:
    if options is None:
        return {}
    if isinstance(options, GraphOptions):
        return options.to_kwargs()
    if isinstance(options, Mapping):
        return dict(options)
    raise TypeError(
        "Graph visualizations require a GraphOptions instance or a mapping."
    )


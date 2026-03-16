import importlib

import pytest

import visualkeras
show_module = importlib.import_module("visualkeras.show")
show = visualkeras.show
from visualkeras.options import (
    FunctionalOptions,
    LayeredOptions,
    GraphOptions,
    LenetOptions,
    LAYERED_PRESETS,
    GRAPH_PRESETS,
)


@pytest.fixture(autouse=True)
def restore_presets():
    # Ensure each test starts with original preset dictionaries
    original_layered = dict(LAYERED_PRESETS)
    original_graph = dict(GRAPH_PRESETS)
    try:
        yield
    finally:
        LAYERED_PRESETS.clear()
        LAYERED_PRESETS.update(original_layered)
        GRAPH_PRESETS.clear()
        GRAPH_PRESETS.update(original_graph)


def test_show_defaults_layered(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)
    monkeypatch.setattr(show_module, "graph_view", lambda *a, **k: pytest.fail("graph_view should not be called"))

    result = show("dummy-model")

    assert result == "layered-image"
    assert captured["model"] == "dummy-model"
    assert captured["kwargs"] == {"preset": None, "options": None}


def test_show_graph_mode(monkeypatch):
    captured = {}

    def fake_graph(model, **kwargs):
        captured["kwargs"] = kwargs
        return "graph-image"

    monkeypatch.setattr(show_module, "layered_view", lambda *a, **k: pytest.fail("layered_view should not be called"))
    monkeypatch.setattr(show_module, "graph_view", fake_graph)

    result = show("model", mode="graph", node_size=64)
    assert result == "graph-image"
    assert captured["kwargs"]["node_size"] == 64


def test_show_with_layered_options(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)
    opts = LayeredOptions(legend=True, draw_volume=False)
    result = show("model", options=opts, spacing=42)

    assert result == "layered-image"
    assert captured["kwargs"]["options"] == opts
    assert captured["kwargs"]["preset"] is None
    assert captured["kwargs"]["spacing"] == 42


def test_show_with_graph_options_mapping(monkeypatch):
    captured = {}

    def fake_graph(model, **kwargs):
        captured["kwargs"] = kwargs
        return "graph-image"

    monkeypatch.setattr(show_module, "layered_view", lambda *a, **k: pytest.fail("layered_view should not be called"))
    monkeypatch.setattr(show_module, "graph_view", fake_graph)

    mapping = GraphOptions(layer_spacing=120).to_kwargs()
    mapping["node_spacing"] = 11
    result = show("model", mode="graph", options=mapping, connector_width=3)

    assert result == "graph-image"
    assert captured["kwargs"]["options"] == mapping
    assert captured["kwargs"]["preset"] is None
    assert captured["kwargs"]["connector_width"] == 3


def test_show_preset_is_forwarded(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)

    LAYERED_PRESETS["test"] = LayeredOptions(draw_reversed=True, spacing=25)

    result = show("model", preset="test", spacing=10)
    assert result == "layered-image"
    assert captured["kwargs"]["preset"] == "test"
    assert captured["kwargs"]["options"] is None
    assert captured["kwargs"]["spacing"] == 10


def test_show_text_callable_name(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)

    show("model", text_callable="name")

    assert captured["kwargs"]["text_callable"] == "name"


def test_show_functional_dispatch(monkeypatch):
    captured = {}

    def fake_functional(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "functional-image"

    monkeypatch.setattr(show_module, "functional_view", fake_functional)
    options = FunctionalOptions(column_spacing=123)

    result = show("model", mode="func", options=options, connector_width=5)
    assert result == "functional-image"
    assert captured["model"] == "model"
    assert captured["kwargs"]["options"] == options
    assert captured["kwargs"]["preset"] is None
    assert captured["kwargs"]["connector_width"] == 5


def test_show_lenet_dispatch(monkeypatch):
    captured = {}

    def fake_lenet(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "lenet-image"

    monkeypatch.setattr(show_module, "lenet_view", fake_lenet)
    options = LenetOptions(layer_spacing=55)

    result = show("model", mode="lenet_style", options=options, draw_connections=False)
    assert result == "lenet-image"
    assert captured["model"] == "model"
    assert captured["kwargs"]["options"] == options
    assert captured["kwargs"]["preset"] is None
    assert captured["kwargs"]["draw_connections"] is False


def test_show_invalid_mode():
    with pytest.raises(ValueError):
        show("model", mode="unknown")


def test_show_invalid_layered_preset():
    with pytest.raises(ValueError):
        show("model", preset="does-not-exist", options=LayeredOptions())


def test_show_invalid_graph_preset():
    with pytest.raises(ValueError):
        show("model", mode="graph", preset="missing", options=GraphOptions())


def test_show_invalid_functional_preset():
    with pytest.raises(ValueError):
        show("model", mode="functional", preset="missing", options=FunctionalOptions())


def test_show_invalid_lenet_preset():
    with pytest.raises(ValueError):
        show("model", mode="lenet", preset="missing", options=LenetOptions())


def test_show_invalid_options_type():
    with pytest.raises(TypeError):
        show("model", options=object())

    with pytest.raises(TypeError):
        show("model", mode="graph", options=object())

    with pytest.raises(TypeError):
        show("model", mode="functional", options=object())

    with pytest.raises(TypeError):
        show("model", mode="lenet", options=object())


def test_validate_options_for_unknown_mode_is_noop():
    # Direct helper call: unknown mode should early-return without raising.
    show_module._validate_options_for_mode("unknown-mode", object())

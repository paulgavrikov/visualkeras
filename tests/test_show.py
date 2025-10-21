import importlib
import types

import pytest

import visualkeras
show_module = importlib.import_module("visualkeras.show")
show = visualkeras.show
from visualkeras.options import (
    LayeredOptions,
    GraphOptions,
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
    assert captured["kwargs"] == {}


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
    assert captured["kwargs"]["legend"] is True
    assert captured["kwargs"]["draw_volume"] is False
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
    assert captured["kwargs"]["layer_spacing"] == 120
    assert captured["kwargs"]["node_spacing"] == 11
    assert captured["kwargs"]["connector_width"] == 3


def test_show_preset_merge(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)

    LAYERED_PRESETS["test"] = LayeredOptions(draw_reversed=True, spacing=25)

    result = show("model", preset="test", spacing=10)
    assert result == "layered-image"
    assert captured["kwargs"]["draw_reversed"] is True
    assert captured["kwargs"]["spacing"] == 10


def test_show_text_callable_name(monkeypatch):
    captured = {}

    def fake_layered(model, **kwargs):
        captured["kwargs"] = kwargs
        return "layered-image"

    monkeypatch.setattr(show_module, "layered_view", fake_layered)

    show("model", text_callable="name")

    assert callable(captured["kwargs"]["text_callable"])
    text, above = captured["kwargs"]["text_callable"](0, types.SimpleNamespace(name="foo", output_shape=(None, 8)))
    assert "foo" in text
    assert isinstance(above, bool)


def test_show_invalid_mode():
    with pytest.raises(ValueError):
        show("model", mode="unknown")


def test_show_invalid_layered_preset(monkeypatch):
    monkeypatch.setattr(show_module, "layered_view", lambda *a, **k: None)

    with pytest.raises(ValueError):
        show("model", preset="does-not-exist")


def test_show_invalid_graph_preset(monkeypatch):
    monkeypatch.setattr(show_module, "graph_view", lambda *a, **k: None)

    with pytest.raises(ValueError):
        show("model", mode="graph", preset="missing")


def test_show_invalid_options_type():
    with pytest.raises(TypeError):
        show("model", options=object())

    with pytest.raises(TypeError):
        show("model", mode="graph", options=object())

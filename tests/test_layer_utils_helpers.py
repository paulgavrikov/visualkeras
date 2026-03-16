from types import SimpleNamespace

import numpy as np
import pytest
import sys
import types

import visualkeras.layer_utils as lu


class _OldInboundNode:
    def __init__(self, inbound_layers):
        self.inbound_layers = inbound_layers


class _NewParentNode:
    def __init__(self, operation):
        self.operation = operation


class _NewInboundNode:
    def __init__(self, parents):
        self.parent_nodes = [_NewParentNode(p) for p in parents]


class _Layer:
    def __init__(self, name):
        self.name = name
        self._inbound_nodes = []
        self._outbound_nodes = []


class _Model:
    def __init__(self, layers):
        self._layers = layers
        self.built = False
        self.output_names = [layers[-1].name]

    def build(self):
        self.built = True

    def get_layer(self, name):
        for layer in self._layers:
            if layer.name == name:
                return layer
        raise KeyError(name)


def _build_chain_model():
    a = _Layer("a")
    b = _Layer("b")
    c = _Layer("c")

    b._inbound_nodes = [_OldInboundNode([a])]
    c._inbound_nodes = [_NewInboundNode([b])]

    a._outbound_nodes = [SimpleNamespace(outbound_layer=b)]
    b._outbound_nodes = [SimpleNamespace(outbound_layer=c)]
    c._outbound_nodes = []
    return _Model([a, b, c]), a, b, c


def test_spacing_dummy_layer_smoke():
    layer = lu.SpacingDummyLayer(spacing=77)
    assert layer.spacing == 77


def test_get_layers_variants_and_error():
    m1 = SimpleNamespace(_layers=[1, 2])
    assert lu.get_layers(m1) == [1, 2]

    m2 = SimpleNamespace(_self_tracked_trackables=["x"])
    assert lu.get_layers(m2) == ["x"]

    with pytest.raises(RuntimeError):
        lu.get_layers(SimpleNamespace())


def test_incoming_and_outgoing_layer_helpers():
    a = _Layer("a")
    b = _Layer("b")
    c = _Layer("c")

    b._inbound_nodes = [_OldInboundNode(a)]
    c._inbound_nodes = [_NewInboundNode([a, b])]
    a._outbound_nodes = [SimpleNamespace(outbound_layer=b), SimpleNamespace(outbound_layer=c)]

    assert list(lu.get_incoming_layers(b)) == [a]
    assert list(lu.get_incoming_layers(c)) == [a, b]
    assert list(lu.get_outgoing_layers(a)) == [b, c]


def test_adj_matrix_lookup_and_hierarchy_helpers():
    model, a, b, c = _build_chain_model()

    id_map, adj = lu.model_to_adj_matrix(model)
    assert model.built is True
    assert adj.shape == (3, 3)
    assert adj[id_map[id(a)], id_map[id(b)]] == 1
    assert adj[id_map[id(b)], id_map[id(c)]] == 1

    assert lu.find_layer_by_id(model, id(b)) is b
    assert lu.find_layer_by_id(model, -1) is None
    assert lu.find_layer_by_name(model, "c") is c
    assert lu.find_layer_by_name(model, "missing") is None

    inputs = list(lu.find_input_layers(model, id_map, adj))
    assert inputs == [a]

    hierarchy = lu.model_to_hierarchy_lists(model, id_map, adj)
    assert [layer.name for layer in hierarchy[0]] == ["a"]
    assert [layer.name for layer in hierarchy[1]] == ["b"]
    assert [layer.name for layer in hierarchy[2]] == ["c"]


def test_find_output_layers_old_and_new_api_paths():
    model, _, _, c = _build_chain_model()
    assert list(lu.find_output_layers(model)) == [c]

    output_tensor = SimpleNamespace(_keras_history=(c, None, None))
    new_model = SimpleNamespace(outputs=[output_tensor], _layers=[c])
    assert list(lu.find_output_layers(new_model)) == [c]


def test_augment_output_layers_adds_dummy_connections():
    model, _, _, c = _build_chain_model()
    id_map, adj = lu.model_to_adj_matrix(model)
    dummy = SimpleNamespace(name="dummy")
    new_map, new_adj = lu.augment_output_layers(model, [dummy], id_map, adj)
    assert new_adj.shape == (4, 4)
    assert id(dummy) in new_map
    assert new_adj[new_map[id(c)], new_map[id(dummy)]] == 1


def test_is_internal_input_fast_paths():
    InputLike = type("InputLayer", (), {})
    assert lu.is_internal_input(InputLike()) is True

    class NotInput:
        pass

    assert lu.is_internal_input(NotInput()) is False


def test_extract_primary_shape_cases():
    with pytest.warns(UserWarning, match="output shape is None"):
        assert lu.extract_primary_shape(None, "layer_x") == (None, 1)

    assert lu.extract_primary_shape((None, 10, 20), "single") == (None, 10, 20)

    with pytest.warns(UserWarning, match="Multi-output layer"):
        assert lu.extract_primary_shape(((None, 5), (None, 2)), "multi_tuple") == (None, 5)

    assert lu.extract_primary_shape([(None, 6)], "single_list") == (None, 6)

    with pytest.warns(UserWarning, match="Multi-output layer"):
        assert lu.extract_primary_shape([(None, 7), (None, 8)], "multi_list") == (None, 7)

    with pytest.warns(UserWarning, match="empty list"):
        assert lu.extract_primary_shape([], "empty") == (None, 1)

    with pytest.raises(RuntimeError, match="Unsupported tensor shape format"):
        lu.extract_primary_shape(123, "bad")


def test_calculate_layer_dimensions_modes_and_fallbacks():
    common = dict(scale_z=1, scale_xy=1, max_z=100, max_xy=200, min_z=5, min_xy=7)

    assert lu.calculate_layer_dimensions((None,), **common) == (7, 7, 5)

    assert lu.calculate_layer_dimensions((None, 12), one_dim_orientation="y", sizing_mode="accurate", **common) == (7, 12, 5)
    assert lu.calculate_layer_dimensions((None, 12), one_dim_orientation="z", sizing_mode="accurate", **common) == (7, 7, 12)
    assert lu.calculate_layer_dimensions((None, 10, 20), sizing_mode="accurate", **common) == (10, 20, 20)
    assert lu.calculate_layer_dimensions((None, 10, 20, 3), sizing_mode="accurate", **common) == (10, 20, 5)

    capped = lu.calculate_layer_dimensions((None, 500), one_dim_orientation="y", sizing_mode="capped", dimension_caps={"sequence": 30}, **common)
    assert capped == (7, 30, 5)

    capped2 = lu.calculate_layer_dimensions((None, 10, 40), sizing_mode="capped", dimension_caps={"sequence": 25, "channels": 15}, **common)
    assert capped2 == (10, 25, 5)

    balanced = lu.calculate_layer_dimensions((None, 128, 256, 3), sizing_mode="balanced", **common)
    assert all(v >= 5 for v in balanced)

    logarithmic = lu.calculate_layer_dimensions((None, 1000, 2000, 64), sizing_mode="logarithmic", **common)
    assert all(v >= 5 for v in logarithmic)

    relative1 = lu.calculate_layer_dimensions((None, 8), one_dim_orientation="z", sizing_mode="relative", relative_base_size=3, **common)
    assert relative1 == (7, 7, 24)

    relative2 = lu.calculate_layer_dimensions((None, 4, 5), sizing_mode="relative", relative_base_size=2, **common)
    assert relative2 == (8, 10, 10)

    relative3 = lu.calculate_layer_dimensions((None, 2, 3, 4), sizing_mode="relative", relative_base_size=2, **common)
    assert relative3 == (7, 7, 8)

    with pytest.warns(UserWarning, match="Unknown sizing mode"):
        fallback = lu.calculate_layer_dimensions((None, 9), one_dim_orientation="y", sizing_mode="unknown", **common)
    assert fallback == (7, 9, 5)


def test_find_input_and_hierarchy_without_precomputed_graph():
    model, a, b, c = _build_chain_model()
    assert [layer.name for layer in lu.find_input_layers(model)] == ["a"]
    hierarchy = lu.model_to_hierarchy_lists(model)
    assert [[layer.name for layer in level] for level in hierarchy] == [["a"], ["b"], ["c"]]


def test_find_output_layers_without_keras_history_yields_empty():
    model = SimpleNamespace(outputs=[SimpleNamespace(name="raw_tensor")], _layers=[])
    assert list(lu.find_output_layers(model)) == []


def test_is_internal_input_backend_paths_via_fake_keras(monkeypatch):
    class KerasLayersInput:
        pass

    class KerasEngineInput:
        pass

    class KerasSrcEngineInput:
        pass

    keras_mod = types.ModuleType("keras")
    keras_layers_mod = types.ModuleType("keras.layers")
    keras_layers_mod.InputLayer = KerasLayersInput
    keras_mod.layers = keras_layers_mod
    keras_mod.engine = SimpleNamespace(input_layer=SimpleNamespace(InputLayer=KerasEngineInput))
    keras_mod.src = SimpleNamespace(engine=SimpleNamespace(input_layer=SimpleNamespace(InputLayer=KerasSrcEngineInput)))

    monkeypatch.setitem(sys.modules, "keras", keras_mod)
    monkeypatch.setitem(sys.modules, "keras.layers", keras_layers_mod)

    assert lu.is_internal_input(KerasLayersInput()) is True
    assert lu.is_internal_input(KerasEngineInput()) is True
    assert lu.is_internal_input(KerasSrcEngineInput()) is True


def test_calculate_layer_dimensions_additional_branch_cases():
    common = dict(scale_z=1, scale_xy=1, max_z=200, max_xy=300, min_z=5, min_xy=7)

    # balanced -> smart_scale large+very-large branches
    b = lu.calculate_layer_dimensions((None, 4096, 8192, 32), sizing_mode="balanced", **common)
    assert all(v >= 5 for v in b)

    # capped one-dimensional z branch
    cap_z = lu.calculate_layer_dimensions((None, 999), one_dim_orientation="z", sizing_mode="capped", dimension_caps={"channels": 40}, **common)
    assert cap_z == (7, 7, 40)

    # balanced one-dimensional y branch
    bal_y = lu.calculate_layer_dimensions((None, 50), one_dim_orientation="y", sizing_mode="balanced", **common)
    assert bal_y[0] == 7 and bal_y[2] == 5

    # logarithmic one-dimensional y/z branches
    log_y = lu.calculate_layer_dimensions((None, 500), one_dim_orientation="y", sizing_mode="logarithmic", **common)
    log_z = lu.calculate_layer_dimensions((None, 1), one_dim_orientation="z", sizing_mode="logarithmic", **common)
    assert log_y[0] == 7 and log_z[1] == 7

    # relative scaling with zero dimension exercises proportional floor branch
    rel_zero = lu.calculate_layer_dimensions((None, 0), one_dim_orientation="y", sizing_mode="relative", relative_base_size=3, **common)
    assert rel_zero == (7, 7, 5)

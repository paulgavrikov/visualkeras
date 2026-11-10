from types import SimpleNamespace

import pytest

import visualkeras.functional as functional
import visualkeras.layered as layered
import visualkeras.lenet as lenet
from visualkeras.layer_utils import extract_primary_shape


@pytest.mark.parametrize("module", [layered, functional, lenet])
@pytest.mark.parametrize("container", [list, tuple, dict])
def test_output_shape_discovery_accepts_tensor_structures(module, container):
    outputs = [SimpleNamespace(shape=(None, 6)), SimpleNamespace(shape=(None, 2))]
    structured = {"first": outputs[0], "second": outputs[1]} if container is dict else container(outputs)
    layer = SimpleNamespace(output=structured)
    with pytest.warns(UserWarning, match="Multi-output layer"):
        primary = extract_primary_shape(module._resolve_layer_output_shape(layer), "split")
    assert primary == (None, 6)


@pytest.mark.parametrize("module", [layered, functional, lenet])
def test_computed_output_shape_precedes_input_shape(module):
    layer = SimpleNamespace(input_shape=(None, 8), compute_output_shape=lambda shape: (shape[0], 2))
    assert module._resolve_layer_output_shape(layer) == (None, 2)


@pytest.mark.parametrize("module", [layered, functional, lenet])
def test_unknown_rank_shape_resolves_without_iteration(module):
    tf = pytest.importorskip("tensorflow")
    assert module._shape_to_tuple(tf.TensorShape(None)) is None


def test_unknown_rank_caption_is_an_unknown_shape():
    tf = pytest.importorskip("tensorflow")
    from visualkeras.options import LAYERED_TEXT_CALLABLES

    layer = SimpleNamespace(output_shape=tf.TensorShape(None))
    assert LAYERED_TEXT_CALLABLES["shape"](0, layer) == ("?", False)


@pytest.mark.parametrize("mode", ["layered", "functional", "lenet"])
def test_unknown_rank_layer_metadata_renders_fallback(mode):
    tf = pytest.importorskip("tensorflow")
    import visualkeras

    inputs = tf.keras.Input((3,), name="unknown_input")
    dense = tf.keras.layers.Dense(2, name="unknown_rank")
    model = tf.keras.Model(inputs, dense(inputs))
    dense.output_shape = tf.TensorShape(None)
    image = visualkeras.show(model, mode=mode)
    assert image.width > 0 and image.height > 0


@pytest.mark.parametrize("module", [layered, functional, lenet])
@pytest.mark.parametrize("proxy", [False, True])
def test_explicit_dictionary_shape_metadata_selects_primary(module, proxy):
    from types import MappingProxyType

    shapes = {"main": (None, 6), "aux": (None, 2)}
    layer = SimpleNamespace(output_shape=MappingProxyType(shapes) if proxy else shapes)
    with pytest.warns(UserWarning, match="Multi-output layer"):
        shape = extract_primary_shape(module._resolve_layer_output_shape(layer), "split")
    assert shape == (None, 6)


def test_empty_dictionary_shape_metadata_uses_fallback():
    with pytest.warns(UserWarning, match="empty"):
        assert extract_primary_shape({}, "empty") == (None, 1)

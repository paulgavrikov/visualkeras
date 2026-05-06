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

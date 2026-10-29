import pytest

import visualkeras
from visualkeras.layer_utils import find_output_layers


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


class OutputAdapter:
    def __init__(self, model, outputs):
        self.layers = model.layers
        self.outputs = outputs
        self.output_shape = model.output_shape
        self.built = True
        self._nodes_by_depth = model._nodes_by_depth
        self._model = model

    def get_layer(self, name):
        return self._model.get_layer(name=name)


def build_model():
    inputs = tf.keras.Input((8,), name="in")
    first = tf.keras.layers.Dense(6, name="first")(inputs)
    second = tf.keras.layers.Dense(2, name="second")(inputs)
    return tf.keras.Model(inputs, [first, second])


@pytest.mark.parametrize("mode", ["graph", "functional"])
@pytest.mark.parametrize("nested", [False, True])
def test_structured_outputs_match_flat_output_nodes(mode, nested):
    model = build_model()
    first, second = model.outputs
    outputs = {"first": [first], "other": {"second": second}} if nested else {"first": first, "second": second}
    adapter = OutputAdapter(model, outputs)
    kwargs = {"add_output_nodes": True} if mode == "functional" else {}
    image = visualkeras.show(adapter, mode=mode, **kwargs)
    reference = visualkeras.show(model, mode=mode, **kwargs)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
    assert list(find_output_layers(adapter)) == [model.get_layer("first"), model.get_layer("second")]

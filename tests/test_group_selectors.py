import pytest

import visualkeras


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


class CustomDense(tf.keras.layers.Dense):
    pass


@pytest.mark.parametrize("mode", ["layered", "functional", "graph"])
@pytest.mark.parametrize("overlap", [False, True])
def test_group_classes_include_subclasses_without_duplicate_selection(mode, overlap):
    inputs = tf.keras.Input((8,), name="in")
    first = tf.keras.layers.Dense(6, name="first")
    second = CustomDense(4, name="second")
    outputs = tf.keras.layers.Activation("relu", name="out")(second(first(inputs)))
    model = tf.keras.Model(inputs, outputs)
    selectors = [tf.keras.layers.Dense, "first", first] if overlap else [tf.keras.layers.Dense]
    group = {"name": "Dense block", "layers": selectors, "fill": (20, 80, 140, 100), "padding": 14}
    image = visualkeras.show(model, mode=mode, layered_groups=[group])
    reference = visualkeras.show(model, mode=mode, layered_groups=[{**group, "layers": [first, second]}])
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
    assert group["layers"] == selectors

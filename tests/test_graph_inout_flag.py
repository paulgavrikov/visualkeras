import pytest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from visualkeras.graph import graph_view


def test_graph_view_inout_as_tensor_flag_changes_width():
    # Build a simple linear model
    inp = Input(shape=(6,), name='input_1')
    out = Dense(2, activation='tanh', name='dense')(inp)
    model = Model(inputs=inp, outputs=out)

    # Render with default inout_as_tensor (True)
    img_default = graph_view(model, inout_as_tensor=True)
    # Render with inout_as_tensor=False (flatten input/output tensors)
    img_flatten = graph_view(model, inout_as_tensor=False)

    # Ensure both images were generated
    assert img_default is not None
    assert img_flatten is not None

    # Flattening should increase horizontal spread (more nodes for input layer)
    assert img_flatten.width > img_default.width, (
        "Expected flattened view to be wider when inout_as_tensor=False"
    )
    # Height should remain similar
    assert abs(img_flatten.height - img_default.height) <= 2, (
        "Expected similar image height for both inout_as_tensor settings"
    )

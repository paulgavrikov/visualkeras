from visualkeras.layer_utils import get_incoming_layers, \
    get_outgoing_layers, find_layer_by_id, find_input_layers, find_output_layers, find_layer_by_name, is_internal_input


def test_get_incoming_layers(functional_model):

    assert list(get_incoming_layers(functional_model.get_layer('input_1'))) == []

    assert list(get_incoming_layers(functional_model.get_layer('layer_1_1'))) == [functional_model.get_layer('input_1')]

    assert list(get_incoming_layers(functional_model.get_layer('concat'))) == \
           [functional_model.get_layer('layer_1_2'), functional_model.get_layer('layer_2_2'), functional_model.get_layer('layer_3_2'),
                      functional_model.get_layer('input_2')]


def test_get_outgoing_layers(functional_model):
    assert len(list(get_outgoing_layers(functional_model.get_layer('dense_4')))) == 0

    assert list(get_outgoing_layers(functional_model.get_layer('input_1'))) == \
                     [functional_model.get_layer('layer_1_1'), functional_model.get_layer('layer_2_1'), functional_model.get_layer('layer_3_1')]

    assert list(get_outgoing_layers(functional_model.get_layer('concat'))) == \
                     [functional_model.get_layer('flatten')]


def test_find_layer_by_id(functional_model):
    assert find_layer_by_id(functional_model, 0) == None

    layer = functional_model.get_layer('dense_1')
    assert find_layer_by_id(functional_model, id(layer)) == layer


def test_find_layer_by_name(functional_model):
    assert find_layer_by_name(functional_model, 'input_1') == functional_model.get_layer('input_1')


def test_find_input_layers(functional_model):
    assert list(find_input_layers(functional_model)) == [functional_model.get_layer('input_1'), functional_model.get_layer('input_2')]


def test_find_output_layers(functional_model):
    assert list(find_output_layers(functional_model)) == [functional_model.get_layer('dense_4'), functional_model.get_layer('concat')]


def test_is_internal_input(model):
    assert is_internal_input(model.get_layer('dense_1')) is False
    assert is_internal_input(model._layers[0]) is True

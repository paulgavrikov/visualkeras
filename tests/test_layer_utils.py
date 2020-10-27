import unittest
import visualkeras
from visualkeras.layer_utils import get_incoming_layers, \
    get_outgoing_layers, find_layer_by_id, find_input_layers, find_output_layers, find_layer_by_name
from keras import layers, models, Model


def get_functional_test_model():
    shape_x = 48
    shape_y = 48

    input_img = layers.Input(shape=(shape_x, shape_y, 1), name='input_1')  # input

    layer_1 = layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_1_1')(input_img)
    layer_1 = layers.Conv2D(1, (3, 3), padding='same', activation='relu', name='layer_1_2')(layer_1)

    layer_2 = layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_2_1')(input_img)
    layer_2 = layers.Conv2D(1, (5, 5), padding='same', activation='relu', name='layer_2_2')(layer_2)

    layer_3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='layer_3_1')(input_img)
    layer_3 = layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_3_2')(layer_3)

    input_img2 = layers.Input(shape=(shape_x, shape_y, 1), name='input_2')  # input

    mid_1 = layers.concatenate([layer_1, layer_2, layer_3, input_img2], axis=3, name='concat')

    flat_1 = layers.Flatten(name='flatten')(mid_1)
    dense_1 = layers.Dense(1, activation='relu', name='dense_1')(flat_1)
    dense_2 = layers.Dense(1, activation='relu', name='dense_2')(dense_1)
    dense_3 = layers.Dense(1, activation='relu', name='dense_3')(dense_2)
    output = layers.Dense(1, activation='softmax', name='dense_4')(dense_3)

    model = Model([input_img, input_img2], [output, mid_1])
    return model


class LayerMethods(unittest.TestCase):

    def test_get_incoming_layers(self):
        model = get_functional_test_model()

        self.assertEqual(len(list(get_incoming_layers(model.get_layer('input_1')))), 0)

        self.assertEqual(list(get_incoming_layers(model.get_layer('layer_1_1'))), [model.get_layer('input_1')])

        self.assertEqual(list(get_incoming_layers(model.get_layer('concat'))),
                         [model.get_layer('layer_1_2'), model.get_layer('layer_2_2'), model.get_layer('layer_3_2'),
                          model.get_layer('input_2')])

    def test_get_outgoing_layers(self):
        model = get_functional_test_model()

        self.assertEqual(len(list(get_outgoing_layers(model.get_layer('dense_4')))), 0)

        self.assertEqual(list(get_outgoing_layers(model.get_layer('input_1'))),
                         [model.get_layer('layer_1_1'), model.get_layer('layer_2_1'), model.get_layer('layer_3_1')])

        self.assertEqual(list(get_outgoing_layers(model.get_layer('concat'))),
                         [model.get_layer('flatten')])

    def test_find_layer_by_id(self):
        model = get_functional_test_model()

        self.assertEqual(find_layer_by_id(model, 0), None)

        layer = model.get_layer('dense_1')
        self.assertEqual(find_layer_by_id(model, id(layer)), layer)

    def test_find_layer_by_name(self):
        model = get_functional_test_model()
        self.assertEqual(find_layer_by_name(model, 'input_1'), model.get_layer('input_1'))

        model = models.Sequential()
        model.add(layers.Dense(1, activation='relu', input_shape=(50,)))

        self.assertEqual(find_layer_by_name(model, 'dense_1_input'), model._layers[0])

    def test_find_input_layers(self):
        model = get_functional_test_model()
        self.assertEqual(list(find_input_layers(model)), [model.get_layer('input_1'), model.get_layer('input_2')])

    def test_find_output_layers(self):
        model = get_functional_test_model()
        self.assertEqual(list(find_output_layers(model)), [model.get_layer('dense_4'), model.get_layer('concat')])


if __name__ == '__main__':
    unittest.main()

import pytest

try:
    import tensorflow as tf
    HAS_TF = True
except ModuleNotFoundError:
    HAS_TF = False

try:
    import keras
    HAS_KERAS = True
except ModuleNotFoundError:
    HAS_KERAS = False


def get_functional_model(lib):
    shape_x = 48
    shape_y = 48

    input_img = lib.layers.Input(shape=(shape_x, shape_y, 1), name='input_1')  # input

    layer_1 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_1_1')(input_img)
    layer_1 = lib.layers.Conv2D(1, (3, 3), padding='same', activation='relu', name='layer_1_2')(layer_1)

    layer_2 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_2_1')(input_img)
    layer_2 = lib.layers.Conv2D(1, (5, 5), padding='same', activation='relu', name='layer_2_2')(layer_2)

    layer_3 = lib.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='layer_3_1')(input_img)
    layer_3 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_3_2')(layer_3)

    input_img2 = lib.layers.Input(shape=(shape_x, shape_y, 1), name='input_2')  # input

    mid_1 = lib.layers.concatenate([layer_1, layer_2, layer_3, input_img2], axis=3, name='concat')

    flat_1 = lib.layers.Flatten(name='flatten')(mid_1)
    dense_1 = lib.layers.Dense(1, activation='relu', name='dense_1')(flat_1)
    dense_2 = lib.layers.Dense(1, activation='relu', name='dense_2')(dense_1)
    dense_3 = lib.layers.Dense(1, activation='relu', name='dense_3')(dense_2)
    output = lib.layers.Dense(1, activation='softmax', name='dense_4')(dense_3)

    model = lib.Model([input_img, input_img2], [output, mid_1])
    return model


def get_functional_model_with_nested(lib):
    shape_x = 48
    shape_y = 48

    input_img = lib.layers.Input(shape=(shape_x, shape_y, 1), name='input_1')  # input

    layer_1 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_1_1')(input_img)
    layer_1 = lib.layers.Conv2D(1, (3, 3), padding='same', activation='relu', name='layer_1_2')(layer_1)

    layer_2 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_2_1')(input_img)
    layer_2 = lib.layers.Conv2D(1, (5, 5), padding='same', activation='relu', name='layer_2_2')(layer_2)

    layer_3 = lib.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='layer_3_1')(input_img)
    layer_3 = lib.layers.Conv2D(1, (1, 1), padding='same', activation='relu', name='layer_3_2')(layer_3)

    input_img2 = lib.layers.Input(shape=(shape_x, shape_y, 1), name='input_2')  # input

    mid_1 = lib.layers.concatenate([layer_1, layer_2, layer_3, input_img2], axis=3, name='concat')

    flat_1 = lib.layers.Flatten(name='flatten')(mid_1)
    dense_1 = lib.layers.Dense(1, activation='relu', name='dense_1')(flat_1)
    dense_2 = lib.layers.Dense(1, activation='relu', name='dense_2')(dense_1)
    dense_3 = lib.layers.Dense(1, activation='relu', name='dense_3')(dense_2)

    subsubnet_in = lib.layers.Input(shape=(1,), name='sub_input')
    subsubnet_l1 = lib.layers.Dense(10, activation='relu', name='sub_dense_1')(subsubnet_in)
    subsubnet_l2 = lib.layers.Dense(10, activation='relu', name='sub_dense_2')(subsubnet_in)
    subsubnet_m1 = lib.layers.concatenate([subsubnet_l1, subsubnet_l2], axis=1, name='sub_concatenate')

    subsubnet_model = lib.Model([subsubnet_in], [subsubnet_m1], name='sub_model')
    sub_out = subsubnet_model(dense_3)

    output = lib.layers.Dense(1, activation='softmax', name='dense_4')(sub_out)

    model = lib.Model([input_img, input_img2], [output, mid_1])
    return model


def get_sequential_model(lib):
    image_size = 8
    model = lib.models.Sequential()
    model.add(lib.layers.InputLayer(input_shape=(image_size, image_size, 3), name='input'))
    model.add(lib.layers.ZeroPadding2D((1, 1), name='zero_padding'))
    model.add(lib.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), name='conv'))
    model.add(lib.layers.MaxPooling2D((2, 2), strides=(2, 2), name='max_pooling'))
    model.add(lib.layers.Flatten(name='flatten'))
    model.add(lib.layers.Dense(1, activation='relu', name='dense_1'))
    model.add(lib.layers.Dropout(0.5, name='dropout'))
    model.add(lib.layers.Dense(1, activation='softmax', name='dense_2'))

    return model


def get_sequential_model_with_nested(lib):
    submodel = lib.models.Sequential()
    submodel.add(lib.layers.Dense(1, activation='relu', name='sub_dense_1'))
    submodel.add(lib.layers.Dropout(0.5, name='sub_dropout'))
    submodel.add(lib.layers.Dense(1, activation='relu', name='sub_dense_2'))

    image_size = 8
    model = lib.models.Sequential()
    model.add(lib.layers.InputLayer(input_shape=(image_size, image_size, 3), name='input'))
    model.add(lib.layers.ZeroPadding2D((1, 1), name='zero_padding'))
    model.add(lib.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), name='conv'))
    model.add(lib.layers.MaxPooling2D((2, 2), strides=(2, 2), name='max_pooling'))
    model.add(lib.layers.Flatten(name='flatten'))
    model.add(lib.layers.Dense(1, activation='relu', name='dense_1'))
    model.add(lib.layers.Dropout(0.5, name='dropout'))
    model.add(submodel)
    model.add(lib.layers.Dense(1, activation='softmax', name='dense_2'))

    return model


def pytest_generate_tests(metafunc):
    if "functional_model" in metafunc.fixturenames:
        metafunc.parametrize("functional_model", ["functional_model_tf", "functional_model_keras"], indirect=True)
    if "sequential_model" in metafunc.fixturenames:
        metafunc.parametrize("sequential_model", ["sequential_model_tf", "sequential_model_keras"], indirect=True)
    if "model" in metafunc.fixturenames:
        metafunc.parametrize("model", ["sequential_model_tf", "sequential_model_keras",
                                       "functional_model_tf", "functional_model_keras",
                                       "sequential_model_tf_with_nested", "sequential_model_keras_with_nested",
                                       "functional_model_tf_with_nested", "functional_model_keras_with_nested"
                                       ],
                             indirect=True)


@pytest.fixture
def model(request):
    return _get_models(request)


@pytest.fixture
def sequential_model(request):
    return _get_models(request)


@pytest.fixture
def functional_model(request):
    return _get_models(request)


def _get_models(request):
    if request.param == "functional_model_tf":
        return get_functional_model(tf.keras)
    elif request.param == "functional_model_keras":
        return get_functional_model(keras)
    elif request.param == "sequential_model_tf":
        return get_sequential_model(tf.keras)
    elif request.param == "sequential_model_keras":
        return get_sequential_model(keras)

    elif request.param == "functional_model_tf_with_nested":
        return get_functional_model_with_nested(tf.keras)
    elif request.param == "functional_model_keras_with_nested":
        return get_functional_model_with_nested(keras)
    elif request.param == "sequential_model_tf_with_nested":
        return get_sequential_model_with_nested(tf.keras)
    elif request.param == "sequential_model_keras_with_nested":
        return get_sequential_model_with_nested(keras)

    else:
        raise ValueError("invalid internal test config")

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D
import visualkeras
from collections import defaultdict


# create VGG16
image_size = 224
model = Sequential()
model.add(InputLayer(input_shape=(image_size, image_size, 3)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.FakeLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.FakeLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.FakeLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.FakeLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# Now visualize the model!

color_map = defaultdict(dict)
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'

visualkeras.cnn_arch(model,
              type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.FakeLayer],
              color_map=color_map,
              dense_orientation='x', draw_volume=True, distance=10).show()

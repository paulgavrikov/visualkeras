from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D
from collections import defaultdict
import visualkeras
from PIL import ImageFont

# create VGG16
image_size = 224
model = Sequential()
model.add(InputLayer(input_shape=(image_size, image_size, 3)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D())
model.add(visualkeras.SpacingDummyLayer())

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# Now visualize the model!

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[ZeroPadding2D]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'

font = ImageFont.truetype("arial.ttf", 32)

visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_show_dimension.png', type_ignore=[visualkeras.SpacingDummyLayer])
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_legend_show_dimension.png', type_ignore=[visualkeras.SpacingDummyLayer]
                         , font=font)
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_spacing_layers_show_dimension.png', spacing=0)
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_type_ignore_show_dimension.png',
                         type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.SpacingDummyLayer])
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_color_map_show_dimension.png',
                         color_map=color_map, type_ignore=[visualkeras.SpacingDummyLayer])
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_flat_show_dimension.png',
                         draw_volume=False, type_ignore=[visualkeras.SpacingDummyLayer])
visualkeras.layered_view(model, legend=True, show_dimension=True, to_file='../figures/vgg16_scaling_show_dimension.png',
                         scale_xy=1, scale_z=1, max_z=1000, type_ignore=[visualkeras.SpacingDummyLayer])
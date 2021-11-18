import keras
from tensorflow.keras import layers
from collections import defaultdict
import visualkeras
from PIL import ImageFont

inputs = layers.Input(shape=(200,), name='Noise vector')
model = visualkeras.SpacingDummyLayer(spacing=100)(inputs)

model = layers.Dense(64*8*8*8, use_bias=False, name='Dense1')(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=25)(model)
# state size. 64*8 * 8 * 8

model = layers.Reshape((8, 8, 64*8))(model)
model = visualkeras.SpacingDummyLayer(spacing=50)(model)
# state size. (64*8) x 8 x 8

model = layers.Conv2DTranspose(64*4, (5, 5), strides=(1, 1), padding='same', use_bias=False)(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=50)(model)
# state size. (64*4) x 8 x 8

model = layers.Conv2DTranspose(64*4, 5, strides=(2, 2), padding='same', use_bias=False)(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=50)(model)
# state size. (64*4) x 16 x 16

model = layers.Conv2DTranspose(64*4, 5, strides=(2, 2), padding='same', use_bias=False)(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=50)(model)
# state size. (64*4) x 32 x 32

model = layers.Conv2DTranspose(64*2, 5, strides=(2, 2), padding='same', use_bias=False)(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=75)(model)
# state size. (64*2) x 64 x 64

model = layers.Conv2DTranspose(64*2, 5, strides=(2, 2), padding='same', use_bias=False)(model)
model = layers.BatchNormalization()(model)
model = layers.ReLU()(model)
model = visualkeras.SpacingDummyLayer(spacing=100)(model)
# state size. (64*2) x 128 x 128

model = layers.Conv2DTranspose(2, 5, strides=(2, 2), padding='same', use_bias=False, activation='tanh')(model)
# state size. (2) x 256 x 256

model = keras.Model(inputs=inputs, outputs=model)

# Colors are colors from aggdraw package which uses CSS-style color names
cm = defaultdict(dict)
alpha = 175
cm[layers.Conv2DTranspose]['fill'] = (239, 71, 111, alpha)
cm[layers.BatchNormalization]['fill'] = (6, 214, 160, alpha)
cm[layers.Dense]['fill'] = (255, 209, 102, alpha)
cm[layers.Dense]['fill'] = (255, 209, 102, alpha)
cm[layers.Reshape]['fill'] = 'orange'
cm[layers.Reshape]['fill'] = (138, 43, 226, alpha)
cm[layers.Flatten]['fill'] = (138, 43, 226, alpha)
cm[layers.ReLU]['fill'] = (17, 138, 178, alpha)
cm[layers.LeakyReLU]['fill'] = (17, 138, 178, alpha)
cm[layers.InputLayer]['fill'] = 'gray'
cm[layers.InputLayer]['fill'] = (100, 100, 100, alpha)
cm[layers.UpSampling2D]['fill'] = (255, 127, 80, alpha)
cm[layers.Conv2D]['fill'] = (239, 71, 111, alpha)

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
font_shapes = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
visualkeras.layered_view(model, scale_xy=2, scale_z=0.5, max_xy=400, max_z=50,
                         one_dim_orientation='x', legend=True, color_map=cm, spacing=10,
                         padding=0, padding_left=0, padding_vertical=75,
                         draw_shapes=3,
                         font=font, font_shapes=font_shapes,
                         background_fill=(255,255,255,255),
                         # type_ignore=[layers.ReLU],
                         to_file='../figures/decoder.png').show()


visualkeras.layered_view(model, scale_xy=2, scale_z=0.5, max_xy=400, max_z=50,
                         one_dim_orientation='x', legend=True, color_map=cm, spacing=10,
                         padding=0, padding_left=200, padding_vertical=75,
                         draw_reversed=True, draw_shapes=3,
                         font=font, font_shapes=font_shapes,
                         background_fill=(255,255,255,255),
                         # type_ignore=[layers.ReLU],
                         to_file='../figures/decoder_reversed.png').show()

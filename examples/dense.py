from tensorflow.python.keras.models import Sequential
from tensorflow.keras import layers
import visualkeras


model = Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(4000,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
visualkeras.layered_view(model, to_file='../figures/spam.png', min_xy=10, min_z=10, scale_xy=100, scale_z=100, one_dim_orientation='x')
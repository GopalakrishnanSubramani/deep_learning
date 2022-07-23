from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150,150,3)
)

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(12,activation='relu'))
model.add(layers.Dense(1,activation='softmax'))

model.compile(loss='categorical_crossentropy',
optimizer=optimizers.Adam,
metrics=['acc'])
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

def build_model():
    inputs = keras.Input(shape=(150, 150, 3))
    conv = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    pool = keras.layers.MaxPooling2D(pool_size=(2,2))(conv)
    flattened = keras.layers.Flatten()(pool)
    dense = keras.layers.Dense(64, 'relu')(flattened)
    output = keras.layers.Dense(1, 'sigmoid')(dense)

    return keras.Model(inputs, output)

def create_image_flow(data_generator, directory, target_size, batch_size, mode, shuffle):
    return data_generator.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=mode,
            shuffle=shuffle)
np.random.seed(1)

train_gen = ImageDataGenerator(rescale=1./255)

train_ds = create_image_flow(train_gen, './train', (150, 150), 20, 'binary', True)

val_gen = ImageDataGenerator(rescale=1./255)

val_ds = create_image_flow(val_gen, './validation', (150, 150), 32, 'binary', True)

model = build_model()
model.summary()
optimizer = keras.optimizers.SGD(lr=0.002, momentum=0.8)
loss = keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(train_ds, steps_per_epoch=100, epochs=10, validation_data=val_ds, validation_steps=50)
print("median train acc:", np.median(history.history["accuracy"]))
print("stdev train loss:", np.std(history.history["loss"]))

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_ds = create_image_flow(train_gen, './train', (150, 150), 20, 'binary', True)

#val_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
#val_ds = create_image_flow(val_gen, './validation', (150, 150), 32, 'binary', True)

history = model.fit(train_ds, steps_per_epoch=100, epochs=10, validation_data=val_ds, validation_steps=50)

val_loss = history.history["val_loss"]
print("mean val loss:", np.mean(val_loss))
val_accuracy = history.history["val_accuracy"]
print("mean val acc last 5:", np.mean(val_accuracy[-5:]))


import tensorflow as tf
from tensorflow import keras

model_filepath = "/Users/syntactic/Downloads/dogs_cats_10_0.687.h5"
model = keras.models.load_model(model_filepath)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('/Users/syntactic/Downloads/dogs_cats_10_0.687.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

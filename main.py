import cv2
import numpy as np 
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras

# Preparing the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Network
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'), #max(0,z)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=12)

model.save('practical.model')

model = tf.keras.models.load_model('practical.model')


def predict(img_str):
    nparr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY) # cv2.IMREAD_COLOR in OpenCV 3.1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    img = cv2.resize(img, (28, 28)) # resize to (28, 28)
    img = np.invert(img) # invert the image
    img = img.reshape(1, 28, 28) # reshape to (1, 28, 28)
    img = tf.keras.utils.normalize(img, axis=1) # normalize the image
    prediction = model.predict(img)
    return np.argmax(prediction)


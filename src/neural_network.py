import time

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

class NeuralNetwork:
    def __init__(self):
        # Prepare Dataset
        cifar10 = keras.datasets.cifar10
        ((self.x_train, self.y_train), (self.x_test, self.y_test)) = cifar10.load_data()
        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 3)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 3)
        self.x_test = self.x_test / 255.0
        self.y_train = tf.one_hot(self.y_train.astype(np.int32), depth=10)
        self.y_test = tf.one_hot(self.y_test.astype(np.int32), depth=10)
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, 2, input_shape=self.x_train.shape[1:], activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(64, 2, input_shape=self.x_train.shape[1:], activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(128, 2, input_shape=self.x_train.shape[1:], activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(256, 2, input_shape=self.x_train.shape[1:], activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    def execute(self, batch_size=32, epochs=10, learning_rate=0.001):
        start_time = time.time()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc'])
        history = self.model.fit(self.x_train, self.y_train, batch_size, epochs)
        print("--- %s seconds ---" % (time.time() - start_time))
        return history

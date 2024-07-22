import time
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras


cifar10 = keras.datasets.cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test = x_test / 255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)


def execute(chromosome):
    start_time = time.time()
    model = keras.Sequential([
        keras.layers.Conv2D(32, 2, input_shape=x_train.shape[1:], activation='relu'),
        keras.layers.MaxPooling2D(strides=(2, 2)),

        keras.layers.Conv2D(64, 2, input_shape=x_train.shape[1:], activation='relu'),
        keras.layers.MaxPooling2D(strides=(2, 2)),

        keras.layers.Conv2D(128, 2, input_shape=x_train.shape[1:], activation='relu'),
        keras.layers.MaxPooling2D(strides=(2, 2)),

        keras.layers.Conv2D(256, 2, input_shape=x_train.shape[1:], activation='relu'),
        keras.layers.MaxPooling2D(strides=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(chromosome.learning_rate)),
        loss='categorical_crossentropy',
        metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        int(chromosome.batch_size),
                        epochs=chromosome.epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))
    return history, (time.time() - start_time)

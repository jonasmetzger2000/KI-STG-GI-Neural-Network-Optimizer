import time
import keras
import numpy as np
from tensorflow import keras
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def execute(chromosome):
    keras.backend.clear_session()
    start_time = time.time()
    model = keras.Sequential([
        keras.layers.Dense(chromosome.neurons, activation='softmax'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax'),
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

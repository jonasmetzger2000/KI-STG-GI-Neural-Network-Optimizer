import tensorflow as tf
from tensorflow.keras import layers, models
import keras
import numpy as np
import sns
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D, UpSampling2D, BatchNormalization, GlobalAveragePooling2D

cifar10 = tf.keras.datasets.cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

input_shape = (32, 32, 3)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test=x_test / 255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

batch_size = 32
num_classes = 10
epochs = 10

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
)

model = keras.Sequential([
    Conv2D(32, 2, input_shape=x_train.shape[1:], activation='relu'),
    MaxPooling2D(strides=(2, 2)),
    BatchNormalization(),

    Conv2D(64, 2, input_shape=x_train.shape[1:], activation='relu'),
    MaxPooling2D(strides=(2, 2)),
    BatchNormalization(),

    Conv2D(128, 2, input_shape=x_train.shape[1:], activation='relu'),
    MaxPooling2D(strides=(2, 2)),
    BatchNormalization(),

    Conv2D(256, 2, input_shape=x_train.shape[1:], activation='relu'),
    MaxPooling2D(strides=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


train_generator = img_gen.flow(x_train, y_train, batch_size=batch_size)

history = model.fit(train_generator, epochs=epochs, steps_per_epoch=len(x_train))
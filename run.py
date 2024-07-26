import json
import sys
from tensorflow import keras
import numpy as np

learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
neurons = int(sys.argv[4])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    keras.layers.Dense(neurons, activation='softmax'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
    loss='categorical_crossentropy',
    metrics=['acc'])
result = model.fit(x_train,
                    y_train,
                    int(batch_size),
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test))

f = open("temp", "w+")
f.write(json.dumps(result.history))

import json
import sys
from tensorflow import keras

learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
neurons = int(sys.argv[4])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Die Eingabebilder in Vektoren umwandeln (Flatten)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28 * 28,)),
    keras.layers.Dense(neurons, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])
result = model.fit(x_train,
                    y_train,
                    int(batch_size),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

f = open("temp", "w+")
f.write(json.dumps(result.history))

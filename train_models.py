import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

x_train_dense = x_train.reshape((x_train.shape[0], -1))
x_test_dense = x_test.reshape((x_test.shape[0], -1))

# Dense model
dense_model = models.Sequential([
    layers.Input((784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
dense_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

dense_model.fit(x_train_dense, y_train, epochs=5, batch_size=128, verbose=2)
dense_model.save("dense_model.h5")

# CNN model
cnn_model = models.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(x_train_cnn, y_train, epochs=5, batch_size=128, verbose=2)
cnn_model.save("cnn_model.h5")

print("Models saved: cnn_model.h5 and dense_model.h5")

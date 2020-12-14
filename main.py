import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from utils.load_training_set import load_training_dataset

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

# load dataset
X, y = load_training_dataset(image_size=(255, 255))

# Normalize pixel value [0, 255] ==> [-1, 1]
X = X / 127.5
X -= 1

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# parameter
num_classes = 3
input_shape = (255, 255, 3)

# model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="relu"),
        layers.Dense(3, activation="softmax")
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=10)

print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=40)
print("test loss, test acc:", results)

model.save('./my_model_dense.keras')

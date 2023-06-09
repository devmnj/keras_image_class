
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)
model = Sequential()
model.add(Dense(units=128, activation="relu", input_shape=(784,)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights("mnistmodel.h5")

img = test_x[601]
test_img = img.reshape((1, 784))

# Predicting the image
img_class = model.predict(test_img)
prediction = img_class[0]
className = np.argmax(prediction, axis=-1)
print("Class: ", className)
img = img.reshape((28, 28))

# Plotting the image
plt.imshow(img)
plt.title(className)
plt.show()

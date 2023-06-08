

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from io import BytesIO
from PIL import Image
import requests


def loadImage(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    img = img.convert('L')
    # img = img.convert('1')
    # img = img.convert('RGB')
    img = img.resize((28, 28), Image.NEAREST)
    img = keras.utils.img_to_array(img)

    return img


url = 'https://edwin-de-jong.github.io/blog/mnist-sequence-data/fig/5.png'

# img = keras.utils.load_img(path="OIP.jpg",color_mode = "grayscale",target_size=(28,28,1))
# img = keras.utils.img_to_array(img)

img = loadImage(url)
model = Sequential()

model.add(Dense(units=128, activation="relu", input_shape=(784,)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights("mnistmodel.h5")

test_img = img.reshape((1, 784))
img_class = model.predict(test_img)
prediction = img_class[0]
className = np.argmax(prediction, axis=-1)
print("Class : ",className)
img = img.reshape((28, 28))

plt.imshow(img)
plt.title(className)
plt.show()

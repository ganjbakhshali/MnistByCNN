import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# cv.waitKey()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.summary()

# if network not trained before...
# model.compile(optimizer=tf.keras.optimizers.Adadelta(
#     learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'
# ),
# tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False),
model.compile(optimizer=tf.keras.optimizers.Adadelta(
    learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'
),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

output = model.fit(x_train, y_train,
                   epochs=5,
                   validation_data=(x_test, y_test))

train_loss = output.history['loss']
train_acc = output.history['accuracy']

plt.plot(train_loss)
plt.plot(train_acc)
plt.show()

model.evaluate(x_test, y_test)

# import cv2
# img = cv.imread('4.jpg')
# img = cv.resize(img, (28, 28))
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# img = img / 255.0
# img = 1 - img
#
# img = img.reshape(1, 28, 28, 1)
#
# y_pred = model.predict(img)
# print(y_pred)
# print(np.argmax(y_pred))

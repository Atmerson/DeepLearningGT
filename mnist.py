from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.datasets import mnist
import numpy as np


###CNN

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# One hot encodes, turns values like 5
# into a vector of 0s and 1s
# 5 -> [0,0,0,0,0,1,0,0,0,0]
def one_hot_encode(values, num_classes):
    return np.eye(num_classes)[values]

# Turns y vectors into one hot encoded vectors
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)

# Keras likes the last dimension being
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# Sequential model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
# 64 different filters (or convolutions) of shape
# 3x3 each, input shape is 28x28, the size of the
# image

model.add(LeakyReLU(0.1))
# Activation function

model.add(Conv2D(16, (3,3))) # Shrink # of filters
model.add(LeakyReLU(0.1))

model.add(Conv2D(8, (3,3)))
model.add(LeakyReLU(0.1))

model.add(MaxPooling2D(pool_size=(2,2)))
# Max pooling just takes the largest of each 2x2 cell

# More convolutions
model.add(Conv2D(4, (3,3)))
model.add(LeakyReLU(0.1))

model.add(Flatten())
# Flattens 8 9x9 images into an 648-dimensional vector

model.add(Dense(10))
model.add(Softmax())

# print summary of model for our benefit
model.summary()

model.compile(optimizer=Adam(lr=0.001), \
              loss=categorical_crossentropy, \
              metrics=["accuracy"]
              )
# compile with Adam optimizer and categorical
# cross entropy loss (which is what we use for
# multiclass classification)

model.fit(x=x_train[:10000], y=y_train[:10000], \
          batch_size=64, \
          epochs=1, validation_data=(x_test, y_test))
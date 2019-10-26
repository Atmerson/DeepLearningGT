import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *

model = Sequential()

model.add(Dense(3, input_shape=(2,), use_bias=True))

model.add(Activation("sigmoid"))
model.add(Dense(1, use_bias = True))

model.add(Activation("sigmoid"))

model.summary()

model.compile(SGD(lr = .001), loss = "mean_squared_error")



X = np.zeros((4,2))
X[0] = (0,0)
X[1] = (1, 0)
X[2] = (0, 1)
X[3] = (1, 1)

y = np.zeros((4,1))

y[0] = 0
y[1] = 1
y[2] = 0
y[3] = 0

model.fit(X,y,epochs= 100)











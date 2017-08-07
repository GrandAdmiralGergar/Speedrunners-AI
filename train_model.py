import numpy as np
import os
from inception import inception_v3 as googleNet
from random import shuffle
from collections import deque

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'Inceptionv3_test.model'

#model = alexnet(WIDTH, HEIGHT, LR, output=5)
model = googleNet(WIDTH, HEIGHT, LR, output=5)
#model = inception_v3(WIDTH, HEIGHT, 3, 0.001, output=5)

train_data = np.load('SpeedrunnersCOLORS.npy')

X = []
Y = []
temp_list = deque()
for obj in train_data:
    features = obj[0]
    actual = obj[1]

    if len(temp_list) == 10 - 1:
        temp_list.append(features)
        X.append(np.array(list(temp_list)))
        Y.append(actual)
        temp_list.popleft()
    else:
        temp_list.append(features)
        continue

shuffle(train_data)
train = train_data[:-200]
test = train_data[-200:]

# Numpy.
X = np.array(X)
Y = np.array(Y)

# Reshape.
X = X.reshape(-1, 10, 3)


X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
test_y = [i[1] for i in test]

model.fit(X, Y, validation_set=(X, Y),
          show_metric=True, snapshot_step=100,
          n_epoch=4)

model.save(MODEL_NAME)
# tensorboard --logdir=C:/Users/synetic707/IdeaProjects/SpeedrunnersML/log

import numpy as np
import h5py
import os
from scipy import misc
import glob
import random
from keras.utils import np_utils

directories = glob.glob('imagenet/training/*')

files = []

for i, d in enumerate(directories):
    images = glob.glob(d + '/*.JPEG')
    for image in images:
        files.append((i, image))

random.shuffle(files)
N_TRAINING = 2500
N_VAL = len(files) - N_TRAINING
files_train = files[:N_TRAINING]
files_val = files[N_TRAINING:]

X_train = np.ndarray((N_TRAINING, 1, 256, 256), dtype=np.uint8)
Y_train = np.ndarray((N_TRAINING))
X_val = np.ndarray((N_VAL, 1, 256, 256), dtype=np.uint8)
Y_val = np.ndarray((N_VAL))

for i, f in enumerate(files_train):
    Y_train[i] = f[0]
    X_train[i, 0] = misc.imread(f[1])

for i, f in enumerate(files_val):
    Y_val[i] = f[0]
    X_val[i, 0] = misc.imread(f[1])

n_classes = 4
Y_train_cat = np_utils.to_categorical(Y_train, n_classes)
Y_val_cat = np_utils.to_categorical(Y_val, n_classes)

print X_train.shape
os.remove('data.h5')
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('1', data=X_train)
h5f.close()


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10

#a = cifar10.load_data()
# nb_filter, stack_size, nb_row, nb_col

#1
print 'creating model'
model = Sequential()
model.add(Convolution2D(48, 1, 11, 11))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# -> 246x246 -> 123x123

#2
print 'creating model'
model.add(Convolution2D(128, 48, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
# -> 119x119 -> 59x59
#3
print 'creating model'
model.add(Convolution2D(192, 128, 3, 3))
model.add(Activation('relu'))

# -> 57x57

#4
print 'creating model'
model.add(Convolution2D(192, 192, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# -> 55x55

#5
print 'creating model'
model.add(Convolution2D(128, 192, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# -> 53x53 -> 26x27

#fc1
print 'creating model'
model.add(Flatten())
model.add(Dense(26*26*128, 1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))

#fc2
print 'creating model'
model.add(Dense(1024, 1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))

#fc3
print 'creating model'
model.add(Dense(1024, 4))
#model.add(Dropout(0.5))
model.add(Activation('softmax'))

print 'creating sgd'
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
print X_train.shape
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print X_train.shape
if os.path.exists('weights'):
    model.load_weights('weights')
model.fit(X_train, Y_train_cat, batch_size=80, nb_epoch=50)
model.save_weights('weights', overwrite='True')

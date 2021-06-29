
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

import glob
import cv2

import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

import random
import matplotlib.pyplot as plt
filelist_A =  'saved_images/A/'
filelist_B = 'saved_images/B/'
IMG_HEIGHT = 500
IMG_WIDTH = 500

x_A_images = [x for x in sorted(os.listdir(filelist_A)) if x[-4:] == '.jpg']
x_A = np.empty((len(x_A_images), IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32') #Leeres Array
y_A = np.ones((x_A.shape[0],1)) # y mit Einsen

for i, name in enumerate(x_A_images):
    im = cv2.imread(filelist_A + name, cv2.IMREAD_UNCHANGED)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float32')
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_A[i] = im # Array wird befüllt

x_B_images = [x for x in sorted(os.listdir(filelist_B)) if x[-4:] == '.jpg']
x_B = np.empty((len(x_B_images), IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32')  # Leeres Array
y_B = np.zeros((x_B.shape[0], 1))  # y mit Nullen

for i, name in enumerate(x_B_images):
    im = cv2.imread(filelist_B + name, cv2.IMREAD_UNCHANGED)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float32')
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_B[i] = im # Array wird befüllt

fig, ax = plt.subplots(1,2, figsize = (8,4))
ax[0].imshow(x_A[3], cmap='gray')
ax[1].imshow(x_B[3], cmap='gray')
ax[0].imshow(cv2.cvtColor(x_A[40], cv2.COLOR_BGR2GRAY), cmap='gray')
ax[1].imshow(cv2.cvtColor(x_B[40], cv2.COLOR_BGR2GRAY), cmap='gray')
plt.show()

#concatenate the two classes for training and validation, x contains the image, y contains the labels (0 or 1)
# Zusammenfügen der Daten aus A und B
x = np.concatenate((x_B, x_A))
y = np.concatenate((y_B, y_A))
x.shape

#Versuch, durch Data Augmentation mehr Daten zu erstellen
imageDataGenerator = ImageDataGenerator(width_shift_range=0.1, rotation_range=40, shear_range= 0.2, zoom_range=0.2, fill_mode='nearest', horizontal_flip=True, vertical_flip=True)


#divide the data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
#x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.1, random_state=40)
print(x_train.shape)
#print(x_train1.shape)
#print(x_train2.shape)

# Create a sequential model, Alexnet
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(500, 500,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(500*500*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))#davor: softmax

model.summary()

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(
    imageDataGenerator.flow(x_train, y_train, batch_size=32),
    validation_data=imageDataGenerator.flow(x_test, y_test, batch_size=32),
    steps_per_epoch=len(x_train) // 32, epochs=10, shuffle=True)

# Test
y_pred = np.argmax(model.predict(x_test), axis=-1)
cmat = confusion_matrix(y_test, y_pred)
print(cmat)
print(y_test.ravel())
print(y_pred.ravel())
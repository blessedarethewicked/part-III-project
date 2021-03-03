import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

print('starting mymodel')

img_x = 80
img_y = 80

# we want to first read in the data form the .csv
mydb = pd.read_csv('mydb\\mydb.csv')

# get an array of just the labels
labels = mydb['0'].values

#  hard to explain but it binearoses the images
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

# drop the labels column from the database
mydb.drop('0',axis=1,inplace=True)

# we split the data into a traning and test set
# the x shows the datast and the y shows the labels
x_train, x_test, y_train, y_test = train_test_split(mydb, labels, test_size = 0.3, random_state = 101)

# we need to save this for the confusion matries
pd.DataFrame(x_train).to_csv("x_train.csv", header=None, index=None)
pd.DataFrame(x_test).to_csv("x_test.csv", header=None, index=None)
pd.DataFrame(y_train).to_csv("y_train.csv", header=None, index=None)
pd.DataFrame(y_test).to_csv("y_test.csv", header=None, index=None)

# reshape so we have images
myx_train = x_train.values
myx_train = np.array([np.reshape(i, (img_x, img_y)) for i in myx_train])
x_train = np.array([i.flatten() for i in myx_train])

print(x_test.shape)
myx_test = x_test.values
myx_test = np.array([np.reshape(i, (img_x, img_y)) for i in myx_test])
x_test = np.array([i.flatten() for i in myx_test])

batch_size = 128 # this ca help with train performance
num_classes = 24 # this is the number of classes that i have 
epochs = 20 # the number of passes though the whole training tes that we do


# normilise the data
# normilise the data
x_train = x_train/255
x_test = x_test/255

# reshape to match the input shape of the model
x_train = x_train.reshape(x_train.shape[0],img_x,img_y,1)
x_test = x_test.reshape(x_test.shape[0],img_x,img_y,1)

print(x_train.shape)
print(y_train.shape)
# CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu', input_shape=(img_x, img_y ,1) ))
# max pooling is done to overcome over fetting by having an abtract view of of the image
# this also helps with the computational cost     
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs,
                     batch_size=batch_size)

# this saves the model
model.save('mymodel21.h5')




print('ending mymodel')
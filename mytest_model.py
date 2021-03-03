import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow


# import the data which is the form of comma separated values
# train = pd.read_csv('sign_mnist_train\\sign_mnist_train.csv')
test = pd.read_csv('x_test.csv') # this is the test dataset
y_test = pd.read_csv('y_test.csv') # this is the labels for the test dataset

letter =['A', 'B', 'C', 'D', 'E', 'F',
         'G', 'H', 'I', 'K', 'L', 'M',
         'N', 'N', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y']

# this converts from  a pandas object to an array
labels = y_test.values
images = test.values

# normilise the data
x_test = images/255
# reshape the data into the correct form 
x_test = x_test.reshape(x_test.shape[0],40,40,1)

# load an existing model  
nmodel = tensorflow.keras.models.load_model('mymodel20.h5')

# my test data point 
datapoint_location = 11
datapoint_value = x_test[datapoint_location:datapoint_location+1]
label = labels[datapoint_location]

# make a prediction
results= nmodel(datapoint_value)
prediction = np.argmax(results)

# get an image
src_dir = 'mydb\\x\\x1385.png'
img = cv2.imread(src_dir,0)


# scale the image 
width = 40
height = 40
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# new_img =  np.array(np.reshape(img, (img.shape[1]*img.shape[0],)), dtype = np.uint8)
img = img/255
print(img.shape)
img = np.array(np.reshape(img,(1,40,40,1)))

# make a prediction
results= nmodel(img)
prediction = np.argmax(results)

print('prediction is')
print(letter[prediction])


import cv2
import numpy as np
import pandas as pd
import tensorflow
import os

# set up the webcam
#we are going to be using the usb cam
cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.1)
# cap.set(cv2.CAP_PROP_EXPOSURE, -1) 

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
font = cv2.FONT_HERSHEY_SIMPLEX
i=''
# theses are the dimension of the tracking window
x = 60
y = 70
h = 200
w = 200


#  scaling valriables
width = 80
height = 80

# load an existing model  
nmodel = tensorflow.keras.models.load_model('mymodel21.h5')

letter =['A', 'B', 'C', 'D', 'E', 'F',
         'G', 'H', 'I', 'K', 'L', 'M',
         'N', 'N', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y']

# more variable dataset
# make sure that the inputs are correct
# c

# this is the main loop
while(cap.isOpened()):
    # read in the frame into the frame 
    # ret is true is we get on image and false if we do not
    ret, frame = cap.read()
    
    if ret==True:
   
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break
        # elif cv2.waitKey(1) & 0xFF == ord('b'):
        
        #copy the part we want 
        img = frame[x:x+h,y:y+w]

        # change it to grey scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # scale the image 
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # have alook at the image resize

        # new_img =  np.array(np.reshape(img, (img.shape[1]*img.shape[0],)), dtype = np.uint8)
        # cv2.imwrite('image.png',img)
        img = img/255
        img = np.array(np.reshape(img,(1,width,height,1)))

        # make a prediction
        results= nmodel(img)
        prediction = np.argmax(results)
        confidance = str(results[0][prediction].numpy())
        i = letter[prediction] + ' ' + confidance 

        # draw a rectangle to help me place my hand
        frame2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        cv2.putText(frame,i,(0,460), font, 1,(255,255,255),2,cv2.LINE_AA)
        # show the frame 
        cv2.imshow('frame',frame)

    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# # import the data which is the form of comma separated values
# # train = pd.read_csv('sign_mnist_train\\sign_mnist_train.csv')
# test = pd.read_csv('x_test.csv') # this is the test dataset
# y_test = pd.read_csv('y_test.csv') # this is the labels for the test dataset

# letter =['A', 'B', 'C', 'D', 'E', 'F',
#          'G', 'H', 'I', 'K', 'L', 'M',
#          'N', 'N', 'P', 'Q', 'R', 'S',
#          'T', 'U', 'V', 'W', 'X', 'Y']

# # this converts from  a pandas object to an array
# labels = y_test.values
# images = test.values

# # normilise the data
# x_test = images/255
# # reshape the data into the correct form 
# x_test = x_test.reshape(x_test.shape[0],40,40,1)

# # load an existing model  
# nmodel = tensorflow.keras.models.load_model('mymodel20.h5')

# # my test data point 
# datapoint_location = 11
# datapoint_value = x_test[datapoint_location:datapoint_location+1]
# label = labels[datapoint_location]

# # make a prediction
# results= nmodel(datapoint_value)
# prediction = np.argmax(results)

# #compare the prediction to the label  
# print('label is ')
# print(letter[np.argmax(label)])
# print('prediction is')
# print(letter[prediction])

# # get an image
# src_dir = 'mydb\\A\\A20.png'
# img = cv2.imread(src_dir,0)


# # scale the image 
# width = 40
# height = 40
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# # new_img =  np.array(np.reshape(img, (img.shape[1]*img.shape[0],)), dtype = np.uint8)
# img = img/255
# print(img.shape)
# img = np.array(np.reshape(img,(1,40,40,1)))

# # make a prediction
# results= nmodel(img)
# prediction = np.argmax(results)

# print('prediction is')
# print(letter[prediction])


#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils as npu
from keras.backend import clear_session

# Load Model
(train_X , train_y), (test_X , test_y) = mnist.load_data("mymnist.data")
# Reshape data and change type
test_X = test_X.reshape(-1 , 28, 28, 1)
train_X = train_X.reshape(-1 ,  28, 28, 1)
test_X = test_X.astype("float32")
train_X = train_X.astype("float32")
# One hot encoding
test_y = npu.to_categorical(test_y)
train_y = npu.to_categorical(train_y)

accuracy= open("/mlops/accuracy.txt","r")
accuracy = float(accuracy.read())


kernel = 8
batch_size = 32


if int(accuracy) < 90:
    model = keras.backend.clear_session()
    model=Sequential()
    model.add(Conv2D(kernel,(3,3), input_shape = (28, 28, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile( optimizer= "Adam" , loss='categorical_crossentropy',  metrics=['accuracy'] )
    train_X.shape
    model_predict= model.fit(train_X, train_y,batch_size=batch_size,verbose=1,epochs=5,validation_data=(test_X, test_y),shuffle=True)
    accuracy=model_predict.history['accuracy']
print('Accuracy ', accuracy)





import smtplib

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()

s.login("naveensatwani99@gmail.com", "*******")


    # message
message_success = "Achieved your desired accuracy . Congrats "


    # sending the mail
s.sendmail("naveensatwani99@gmail.com", "navinsatwani99@gmail.com", message_success)


    # terminating the session
s.quit()


# In[ ]:

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.optimizers import Adam


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


len(dataset)


# In[4]:


train , test = dataset


# In[5]:


len(train)


# In[6]:


X_train , y_train = train


# In[7]:


X_train.shape


# In[8]:


X_test , y_test = test


# In[9]:


X_test.shape


# In[10]:


img1 = X_train[7]


# In[11]:


img1.shape


# In[12]:


import cv2


# In[13]:


img1_label = y_train[7]


# In[14]:


img1_label


# In[15]:


img1.shape


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


plt.imshow(img1 , cmap='gray')


# In[18]:


img1.shape


# In[19]:


img1_1d = img1.reshape(28*28)


# In[20]:


img1_1d.shape


# In[21]:


X_train.shape


# In[22]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[23]:


X_train_1d.shape


# In[24]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[25]:


X_train.shape


# In[26]:


y_train.shape


# In[27]:


from keras.utils.np_utils import to_categorical


# In[28]:


y_train_cat = to_categorical(y_train)


# In[29]:


y_train_cat


# In[30]:


y_train_cat[7]


# In[31]:


from keras.models import Sequential


# In[32]:


from keras.layers import Dense


# In[33]:


model = Sequential()


# In[34]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[35]:


model.summary()


# In[36]:


model.add(Dense(units=256, activation='relu'))


# In[37]:


model.add(Dense(units=128, activation='relu'))


# In[38]:


model.add(Dense(units=32, activation='relu'))


# In[39]:


model.summary()


# In[40]:


model.add(Dense(units=10, activation='softmax'))


# In[41]:


model.summary()


# In[42]:


from keras.optimizers import RMSprop


# In[43]:


model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[46]:


h = model.fit(X_train, y_train_cat, epochs=5)


# In[55]:


model.save('mnistt.h5')

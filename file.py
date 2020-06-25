#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import mnist
from keras.optimizers import Adam


# In[3]:


dataset = mnist.load_data('mymnist.db')


# In[4]:


len(dataset)


# In[5]:


train , test = dataset


# In[6]:


len(train)


# In[7]:


X_train , y_train = train


# In[8]:


X_train.shape


# In[9]:


X_test , y_test = test


# In[10]:


X_test.shape


# In[11]:


img1 = X_train[7]


# In[12]:


img1.shape


# In[13]:


import cv2


# In[14]:


img1_label = y_train[7]


# In[15]:


img1_label


# In[16]:


img1.shape


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.imshow(img1 , cmap='gray')


# In[19]:


img1.shape


# In[20]:


img1_1d = img1.reshape(28*28)


# In[21]:


img1_1d.shape


# In[22]:


X_train.shape


# In[23]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[24]:


X_train_1d.shape


# In[25]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[26]:


X_train.shape


# In[27]:


y_train.shape


# In[28]:


from keras.utils.np_utils import to_categorical


# In[29]:


y_train_cat = to_categorical(y_train)


# In[30]:


y_train_cat


# In[31]:


y_train_cat[7]


# In[32]:


from keras.models import Sequential


# In[33]:


from keras.layers import Dense


# In[34]:


model = Sequential()


# In[35]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[36]:


model.summary()


# In[37]:


model.add(Dense(units=256, activation='relu'))


# In[38]:


model.add(Dense(units=128, activation='relu'))


# In[39]:


model.add(Dense(units=32, activation='relu'))


# In[40]:


model.summary()


# In[41]:


model.add(Dense(units=10, activation='softmax'))


# In[42]:


model.summary()


# In[43]:


from keras.optimizers import RMSprop


# In[44]:


model.compile(optimizer='adam', loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[45]:


h = model.fit(X_train, y_train_cat, epochs=5)


# In[46]:


model.save('mnistt.h5')


# In[47]:


accuracy=h.history['accuracy'][-1]*100


# In[49]:


f=open("/mlops/accuracy.txt","w")


# In[50]:


f.write(str(accuracy))


# In[51]:


f.close()


# In[52]:


accuracy

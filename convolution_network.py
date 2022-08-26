#!/usr/bin/env python
# coding: utf-8

# In[40]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__
import cv2


# In[41]:


# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('F:/Project/augmented data',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 #color_mode="grayscale",
                                                 class_mode = 'binary')


# In[42]:


# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('F:/Project/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            #color_mode="grayscale",
                                            class_mode = 'binary')


# In[43]:


#Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[44]:


cnn.summary()


# In[45]:


# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[46]:


cnn.save('F:/Project/saved_models/cnn5_model.h5')


# In[57]:


#plotting accuracy
history.history.keys()

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('number of epoch')
plt.legend(['accuracy'],loc='upper left')
plt.show()


# In[50]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()


# In[ ]:





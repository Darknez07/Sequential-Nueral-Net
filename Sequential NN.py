
# coding: utf-8

# In[13]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis= 1)
x_test = tf.keras.utils.normalize(x_test, axis= 1)
#four layers of nn
#Sequantial model
#128 128 10 1 nodes
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
#adam optimizer and loss
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics =['accuracy'])
model.fit(x_train, y_train, epochs = 3)


# In[10]:


import matplotlib.pyplot as plt
print(x_train[0])


# In[14]:


#Model is tested
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[16]:


#showing the trained
plt.imshow(x_train[10], cmap = plt.cm.binary)
plt.show()


# In[22]:


#all the prediction done by the user
predictions = model.predict([x_test])
print(predictions)


# In[25]:


#Showing each prediction from array of pixels to image
#matching with predictions
import numpy as np
for i in range(len(predictions)):
    print(np.argmax(predictions[i]))
    plt.imshow(x_test[i])
    plt.show()


#!/usr/bin/env python
# coding: utf-8

# Using the Heart Disease Dataset for Training a Heart Disease Predictor Utilizing an ANN

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# In[4]:


diseases=pd.read_csv("Dataset Heart Disease.csv")
diseases.head()


# In[5]:


diseases.info()


# In[6]:


diseases.drop('Unnamed: 0',axis=1,inplace=True)
diseases


# In[7]:


y = diseases['target']
x = diseases.drop('target',axis=1)


# In[8]:


scaler = StandardScaler()
X = scaler.fit_transform(x)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


model = Sequential([
    Dense(256, input_shape=(11,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    Dropout(0.2),  
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    Dropout(0.2),
    Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    Dropout(0.2),
    Dense(1, activation='sigmoid')  
])


# In[11]:


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[12]:


import matplotlib.pyplot as plt

# Train the model and store history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
pickle.dump(history, open("model.pkl", 'wb'))

# Extract accuracy values
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

# Plot accuracy values
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[13]:


y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
y_pred_flat = np.ravel(y_pred)
y_test_flat = np.ravel(y_test)
conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)

print("Confusion Matrix:")
print(conf_matrix)


# In[15]:


import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# In[ ]:





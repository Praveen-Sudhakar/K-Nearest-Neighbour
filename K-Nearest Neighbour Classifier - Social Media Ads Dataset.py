#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary packages

import numpy as np
import pandas as pd


# In[2]:


#Reading the dataset

df = pd.read_csv("D:\AIML\Dataset\Social_Network_Ads.csv")

df


# In[3]:


df.info()


# In[20]:


#Declaring the IV

x = df.iloc[:,1:4].values
y = df.iloc[:,-1]


# In[21]:


print(x[0:5])
print(y[0:5])


# In[18]:


print(set(df['Gender']))
print(set(df['Purchased']))


# In[23]:


#Label encoding the 'Object' column

from sklearn.preprocessing import LabelEncoder

gend = LabelEncoder()

gend.fit(['Female', 'Male'])

x[:,0] = gend.transform(x[:,0])


# In[129]:


#Splitting the data

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.4,random_state=50)

print("Train shape",train_x.shape,train_y.shape)
print("Test shape",test_x.shape,test_y.shape)


# In[147]:


#Modeling

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors = 3)

neigh.fit(train_x,train_y)


# In[148]:


#Evaluating the model using the test data

pred_y = neigh.predict(test_x)


# In[149]:


#Checking the accuracy score

from sklearn.metrics import f1_score

print("Accuracy score = ",f1_score(test_y,pred_y)*100)


# In[150]:


#Plotting

import matplotlib.pyplot as plt

plt.scatter(test_x[:,1],test_x[:,2],c=test_y)


# In[151]:


plt.scatter(test_x[:,1],test_x[:,2],c=pred_y)


# In[ ]:





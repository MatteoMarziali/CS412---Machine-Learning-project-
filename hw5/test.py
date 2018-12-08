#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump, load


# In[11]:


print("loading preprocessed data..")
dataset = pd.read_csv('preprocessedDataset2.csv')
dataset.head()


# In[12]:


dataset = dataset.drop('Unnamed: 0',1)
dataset.head()


# In[13]:


y = dataset['Empathy']
del dataset['Empathy']
dataset.shape


# In[14]:


print("splitting into train and test set..")

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2,random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[23]:


print("loading models ")
DummyClf = load('dummyClf.joblib') 
RidgeClf = load('RidgeClf.joblib') 
RidgefClf = load('Ridge-fClf.joblib') 


# In[24]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
X_new2 = SelectKBest(f_classif, k=70).fit_transform(dataset, y)
X_new2.shape


# In[25]:


X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_new2, y, test_size=0.2,random_state = 0)


# In[28]:


print("max_frequent classifier (BASELINE) test accuracy = " + str(DummyClf.score(X_test,y_test)))


# In[27]:


print("Ridge classifier test (to compare) accuracy = " + str(RidgeClf.score(X_test,y_test)))


# In[26]:


print("Ridge classifier + selectKBest f-classif (best) test accuracy = " + str(RidgefClf.score(X_testf,y_testf)))


# In[ ]:





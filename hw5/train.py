#!/usr/bin/env python
# coding: utf-8

# In[33]:


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


# In[3]:


dataset = pd.read_csv('preprocessedDataset2.csv')
dataset.head()


# In[6]:


dataset = dataset.drop('Unnamed: 0',1)
dataset.head()


# In[7]:


dataset.shape


# In[35]:


y = dataset['Empathy']
del dataset['Empathy']
dataset.shape


# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2,random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## Dummy classifier 

# In[109]:


clfDummy = DummyClassifier(strategy='most_frequent',random_state=0)
clfDummy.fit(X_train, y_train)


# In[110]:


clfDummy.score(X_test,y_test)


# In[111]:


dump(clfDummy, 'dummyClf.joblib') 


# ## Ridge classifier

# In[122]:


#hyperparameter tuning
for i in [0.5,5,50,100,200,400,800,1000,2000]:
    clf = RidgeClassifier(alpha=i,fit_intercept=True)
    scores = cross_val_score(clf,X_train,y_train,cv=10)
    print("alpha = "+ str(i) +" , acc = " + str(np.mean(scores)))


# In[124]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000,fit_intercept=True)
clf.fit(X_train,y_train)
print("test acc = " + str(clf.score(X_test,y_test)))
print("train acc = " +str(clf.score(X_train,y_train)))


# In[125]:


dump(clf, 'RidgeClf.joblib') 


# In[127]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
X_new2 = SelectKBest(f_classif, k=70).fit_transform(dataset, y)
X_new2.shape


# In[128]:


X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_new2, y, test_size=0.2,random_state = 0)


# In[129]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000).fit(X_trainf, y_trainf)
print("test accuracy = " + str( clf.score(X_testf, y_testf) ))
print("train accuracy = " + str( clf.score(X_trainf, y_trainf) ))


# In[130]:


dump(clf, 'Ridge-fClf.joblib') 


# In[ ]:





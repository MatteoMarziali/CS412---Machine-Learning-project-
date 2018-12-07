#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


print("loading dataset..")
dataset = pd.read_csv('responses.csv')
dataset.head()


# In[3]:


print("values for Smoking:")
print(dataset.Smoking.unique())
print("values for Alcohol:")
print(dataset.Alcohol.unique())
print("values for Punctuality:")
print(dataset.Punctuality.unique())
print("values for Lying:")
print(dataset.Lying.unique())
print("values for Internet usage:")
print(dataset['Internet usage'].unique())
print("values for Gender:")
print(dataset.Gender.unique())
print("values for Education:")
print(dataset.Education.unique())
print("values for Only child:")
print(dataset['Only child'].unique())
print("values for left - right handed:")
print(dataset['Left - right handed'].unique())
print("values for Village - town:")
print(dataset['Village - town'].unique())
print("values for House - block of flats:")
print(dataset['House - block of flats'].unique())


# ## Feature exploration

# Let's see if, considering features with strings value, they have very rare values that we can merge together, or let's see how the age values are distributed and if it makes sense to encode them in a certain way.

# In[4]:


pd.value_counts(dataset.Smoking)


# In[5]:


pd.value_counts(dataset.Alcohol)


# In[6]:


pd.value_counts(dataset.Lying)


# In[7]:


pd.value_counts(dataset.Punctuality)


# In[8]:


pd.value_counts(dataset.Education)


# given this, we can merge currently a primary school pupil with primary school, also looking the Age Column cannot be persons that are currently doing the primary school, considering that the youngest are 15.
# We cannot delete 10 rows of the dataset, even if we have to take into account that these rows can possibly contain more mistakes. 

# In[9]:


pd.value_counts(dataset['Internet usage'])


# In[10]:


pd.value_counts(dataset.Age)


# As we can see from Ages values, is not possible to have someone that is currently a primary school pupil, can this is an error?

# In[11]:


pd.value_counts(dataset.Weight)


# 165.0 and 150.0 are clearly outliers, maybe they can be heights instead of weights,I want to remove them as may be wrong

# In[12]:


dataset = dataset[dataset.Weight != 150.0]
dataset = dataset[dataset.Weight != 165.0]


# In[13]:


pd.value_counts(dataset.Height)


# 62.0 is clearly an outlier, maybe is it a weight instead of a height, I want to remove it as may be wrong.

# In[14]:


dataset = dataset[dataset.Height != 62.0]


# In[ ]:


print("deleting noisy examples..")


# In[15]:


dataset.shape


# In[16]:


def convertSmoking(value):
    if(value=='never smoked'):
        return 1
    elif(value=='tried smoking'):
        return 2
    elif(value == 'former smoker'):
        return 3 
    elif(value=='current smoker'):
        return 4

def convertAlcohol(value):
    if(value=='never'):
        return 1
    elif(value=='social drinker'):
        return 2
    elif(value == 'drink a lot'):
        return 3 
    
def convertPunctuality(value):
    if(value=='i am often running late'):
        return 1
    elif(value=='i am often early'):
        return 2
    elif(value == 'i am always on time'):
        return 3 

def convertLying(value):
    if(value=='never'):
        return 1
    elif(value=='sometimes'):
        return 2
    elif(value == 'only to avoid hurting someone'):
        return 3 
    elif(value == 'everytime it suits me'):
        return 4

def convertInternetUsage(value):
    if(value=='few hours a day'):
        return 3
    elif(value=='less than an hour a day'):
        return 2
    elif(value == 'most of the day'):
        return 4 
    elif(value == 'no time at all'):
        return 1

def convertEducation(value):
    if(value=='currently a primary school pupil'):
        return 1
    elif(value=='primary school'):
        return 2
    elif(value == 'secondary school'):
        return 3
    elif(value == 'college/bachelor degree'):
        return 4
    elif(value == 'masters degree'):
        return 5
    elif(value == 'doctorate degree'):
        return 6


# In[17]:


dataset['Smoking'] = dataset.Smoking.apply(convertSmoking)
dataset['Alcohol'] = dataset.Alcohol.apply(convertAlcohol)
dataset['Lying'] = dataset.Lying.apply(convertLying)
dataset['Punctuality'] = dataset.Punctuality.apply(convertPunctuality)
dataset['Education'] = dataset.Education.apply(convertEducation)
dataset['Internet usage'] = dataset['Internet usage'].apply(convertInternetUsage)


# In[18]:


def GenderConversion(value):
    if value=='male':
        return 1
    elif value == 'female':
        return 0

def OnlyChildConversion(value):
    if value=='yes':
        return 1
    elif value == 'no':
        return 0
    
def LRHandedConversion(value):
    if value=='right handed':
        return 0
    elif value == 'left handed':
        return 1

def VillageTownConversion(value):
    if value=='city':
        return 0
    elif value == 'village':
        return 1

def HouseBoFlatsConversion(value):
    if value=='house/bungalow':
        return 1
    elif value == 'block of flats':
        return 0

def Range1_5ValuesConversion(value):
    if value==4 or value==5:
        return 1
    elif value == 1 or value == 2 or value == 3:
        return 0


# In[19]:


dataset['Gender'] = dataset['Gender'].apply(GenderConversion)
dataset['Only child'] = dataset['Only child'].apply(OnlyChildConversion)
dataset['Left - right handed'] = dataset['Left - right handed'].apply(LRHandedConversion)
dataset['Village - town'] = dataset['Village - town'].apply(VillageTownConversion)
dataset['House - block of flats'] = dataset['House - block of flats'].apply(HouseBoFlatsConversion)
dataset['Empathy'] = dataset['Empathy'].apply(Range1_5ValuesConversion)


# In[ ]:


print("encoding attributes...")


# # Missing values analysis

# In[20]:


df_na = (dataset.isnull().sum() / len(dataset)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data


# In[21]:


dataset = dataset.apply(lambda x: x.fillna(x.median()),axis=0)


# In[22]:


dataset.isnull().values.any()


# In[ ]:


print("imputing missing values...")


# In[ ]:


print("saving the new preprocessed dataset as 'preprocessedDataset2'")


# In[23]:


dataset.to_csv("preprocessedDataset2.csv")


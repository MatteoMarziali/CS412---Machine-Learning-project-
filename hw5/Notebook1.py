#!/usr/bin/env python
# coding: utf-8

# In[593]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})  

import warnings 
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Feature exploration

# In[594]:


dataset = pd.read_csv('./young-people-survey/responses.csv')
dataset.head()


# In[595]:


print("(rows, columns)")
dataset.shape


# In[596]:


categorical=['Smoking','Alcohol','Punctuality','Lying','Internet usage','Gender','Left - right handed','Education','Only child','Village - town','House - block of flats']
columns = pd.read_csv('./young-people-survey/columns.csv')
features = []
for v in columns.loc[:,'short']:
     features.append(v)
features #this contains both numerical and categorical features
numerical = []
for v in features:
    if v not in categorical:
        numerical.append(v)

print("numerical features: " + str(len(numerical)))
print("categorical features: " + str(len(categorical)))


# In[597]:


dfnum=dataset[numerical]
dfcat=dataset[categorical]


# In[598]:


dfcat.head()


# In[599]:


min_range_max = pd.DataFrame({
        'min': dfnum.min(),
        'range': dfnum.max() - dfnum.min(),
        'max': dfnum.max()
    })

min_range_max[['min', 'range', 'max']]


# # ONE HOT ENCODING and VALUES CONVERSION

# Since the classification would require to estimate whether a person is either 'very empathetic' (values 4 and 5) or 'not very empathetic' (values 1,2,3), it makes sense to convert the values 4 and 5 into 1 and the values 1,2,3 into 0 for every feature that has these kind of values. 
# Concerning the other features that have 'String' values (i.e. smoking) I'll perform ONE HOT ENCODING to transform them into binary variables.
# Before that I have to check which values can they assume and define apposite functions to perform the conversion. 

# In[600]:


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


# ### OHE for Smoking

# In[601]:


def OHE_Smoking_neverSmoked(value):
    return 1 if value=='never smoked' else 0

def OHE_Smoking_triedSmoking(value):
    return 1 if value=='tried smoking' else 0

def OHE_Smoking_formerSmoker(value):
    return 1 if value=='former smoker' else 0

def OHE_Smoking_currentSmoker(value):
    return 1 if value=='current smoker' else 0


# In[602]:


dataset['Smoking_neverSmoked'] = dataset.Smoking.apply(OHE_Smoking_neverSmoked)
dataset['Smoking_triedSmoking'] = dataset.Smoking.apply(OHE_Smoking_triedSmoking)
dataset['Smoking_formerSmoker'] = dataset.Smoking.apply(OHE_Smoking_formerSmoker)
dataset['Smoking_currentSmoker'] = dataset.Smoking.apply(OHE_Smoking_currentSmoker)


# ### OHE for Alcohol

# In[603]:


def OHE_Alcohol_drinkALot(value):
    return 1 if value=='drink a lot' else 0

def OHE_Alcohol_socialDrinker(value):
    return 1 if value=='social drinker' else 0

def OHE_Alcohol_never(value):
    return 1 if value=='never' else 0


# In[604]:


dataset['Alcohol_drinkALot'] = dataset.Alcohol.apply(OHE_Alcohol_drinkALot)
dataset['Alcohol_socialDrinker'] = dataset.Alcohol.apply(OHE_Alcohol_socialDrinker)
dataset['Alcohol_never'] = dataset.Alcohol.apply(OHE_Alcohol_never)


# ### OHE for Punctuality

# In[605]:


def OHE_Punctuality_alwOnTime(value):
    return 1 if value=='i am always on time' else 0

def OHE_Punctuality_oftenEarly(value):
    return 1 if value=='i am often early' else 0

def OHE_Punctuality_oftenRunningLate(value):
    return 1 if value=='i am often running late' else 0


# In[606]:


dataset['Punctuality_alwOnTime'] = dataset.Punctuality.apply(OHE_Punctuality_alwOnTime)
dataset['Punctuality_oftenEarly'] = dataset.Punctuality.apply(OHE_Punctuality_oftenEarly)
dataset['Punctuality_oftenRunningLate'] = dataset.Punctuality.apply(OHE_Punctuality_oftenRunningLate)


# ### OHE for Lying

# In[607]:


def OHE_Lying_never(value):
    return 1 if value=='never' else 0

def OHE_Lying_sometimes(value):
    return 1 if value=='sometimes' else 0

def OHE_Lying_notToHurtSomeone(value):
    return 1 if value=='only to avoid hurting someone' else 0

def OHE_Lying_whenSuitsMe(value):
    return 1 if value=='everytime it suits me' else 0


# In[608]:


dataset['Lying_never'] = dataset.Lying.apply(OHE_Lying_never)
dataset['Lying_sometimes'] = dataset.Lying.apply(OHE_Lying_sometimes)
dataset['Lying_notToHurtSomeone'] = dataset.Lying.apply(OHE_Lying_notToHurtSomeone)
dataset['Lying_whenSuitsMe'] = dataset.Lying.apply(OHE_Lying_whenSuitsMe)


# ### OHE for Internet usage

# In[609]:


def OHE_InternetUsage_fewHoursDay(value):
    return 1 if value=='few hours a day' else 0

def OHE_InternetUsage_mostOfDay(value):
    return 1 if value=='most of the day' else 0

def OHE_InternetUsage_lessOneHourDay(value):
    return 1 if value=='less than an hour a day' else 0

def OHE_InternetUsage_never(value):
    return 1 if value=='no time at all' else 0


# In[610]:


dataset['InternetUsage_fewHoursDay'] = dataset['Internet usage'].apply(OHE_InternetUsage_fewHoursDay)
dataset['InternetUsage_mostOfDay'] = dataset['Internet usage'].apply(OHE_InternetUsage_mostOfDay)
dataset['InternetUsage_lessOneHourDay'] = dataset['Internet usage'].apply(OHE_InternetUsage_lessOneHourDay)
dataset['InternetUsage_never'] = dataset['Internet usage'].apply(OHE_InternetUsage_never)


# ### OHE for Education

# In[611]:


def OHE_Education_collegeBachelor(value):
    return 1 if value=='college/bachelor degree' else 0

def OHE_Education_secondary(value):
    return 1 if value=='secondary school' else 0

def OHE_Education_primary(value):
    return 1 if value=='primary school' else 0

def OHE_Education_master(value):
    return 1 if value=='masters degree' else 0

def OHE_Education_doctorate(value):
    return 1 if value=='doctorate degree' else 0

def OHE_Education_currPrimaryPupil(value):
    return 1 if value=='currently a primary school pupil' else 0


# In[612]:


dataset['Education_collegeBachelor'] = dataset['Education'].apply(OHE_Education_collegeBachelor)
dataset['Education_secondary'] = dataset['Education'].apply(OHE_Education_secondary)
dataset['Education_primary'] = dataset['Education'].apply(OHE_Education_primary)
dataset['Education_master'] = dataset['Education'].apply(OHE_Education_master)
dataset['Education_doctorate'] = dataset['Education'].apply(OHE_Education_doctorate)
dataset['Education_CurrPrimaryPupil'] = dataset['Education'].apply(OHE_Education_currPrimaryPupil)


# ## Binary conversion

# Done the OHE, what we still have to do is convert the other features that do not require one hot encoding into binary features, we will use the same approach used for one hot encoding, writing functions to apply to different features.

# ### Gender

# In[613]:


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


# In[614]:


dataset['Gender'] = dataset['Gender'].apply(GenderConversion)
dataset['Only child'] = dataset['Only child'].apply(OnlyChildConversion)
dataset['Left - right handed'] = dataset['Left - right handed'].apply(LRHandedConversion)
dataset['Village - town'] = dataset['Village - town'].apply(VillageTownConversion)
dataset['House - block of flats'] = dataset['House - block of flats'].apply(HouseBoFlatsConversion)
#for f in numerical:
#    if f!='Age' and f!='Number of siblings' and f!='Weight' and f!='Height':
#        dataset[f] = dataset[f].apply(Range1_5ValuesConversion)
dataset['Empathy'] = dataset['Empathy'].apply(Range1_5ValuesConversion)


# In[615]:


dataset.shape


# now we have to remove some columns from the dataset, such as the ones corresponding to features that have been 'One Hot encoded'

# In[616]:


dataset = dataset.drop(columns=['Smoking', 'Alcohol','Lying','Punctuality','Internet usage','Education'])


# Now that we have just numerical attributes we can easily replace the missing values. 
# I just want to point out that now the size of the dataset is increased, in fact One Hot Encoding added 18 new columns to the dataset. 

# # Missing values analysis

# In[617]:


df_na = (dataset.isnull().sum() / len(dataset)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data


# As we can see above, we have a few missing values, anyway they do not have a very high missing ratio so we can try to impute all the missing values.
# Since most of the values are in a 1 to 5 range, and features like smoking, height and weight are not very indicative of how much a person is empathetic and will probably be neglected, I decided to impute the missing values with their median value, to keep the overall distribution of data.

# In[618]:


dataset = dataset.apply(lambda x: x.fillna(x.median()),axis=0)


# Let's check that we don't have missing values anymore:

# In[619]:


dataset.isnull().values.any()


# ### Now that we have a dataset with only numerical values and without missing values we may want to save it for the next work to be done

# In[620]:


dataset.to_csv("preprocessedDataset.csv")


# ## We also need to split this new dataset into train and test set

# we need to define the target variable and store the relative column into y and remove it from the daset cause we don't want to keep it in the training set.

# In[621]:


y = dataset.pop('Empathy') # define the target variable (dependent variable) as y
X = dataset


# Now we use the function provided from sklearn library to split the dataset into train and test, keeping the test size the 20% of the dataset size

# In[622]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 69)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Let's try to perform a first classification by predicting always the most frequent using sklearn.DummyClassifier and let's see the accuracy to have a comparison baseline

# In[623]:


clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train, y_train)
DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
clf.score(X_test, y_test)  


# In[624]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=11,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# What can we do to improve the performances? For sure Hyperparameter tuning on a dev set will allow us to exploit the full power of our algorithms.
# Before doing that, it seems more meaningful to change the econding of 'string' feature values to keep the order relations that they have.

# Looking at the scores, we want to try to improve the performances, since we have a very limited-size dataset, in order to avoid to add too many features, something that we can do is replacing OHE with ordinal encoding for some features, for example for Smoking, since it represents a different levels of Smoking we can assign a score from 1 to 4 replacing the 'strings' values of the original feature, keeping the meaning of the attribute.

# This and further improvements to the analysis are done in notebook2

# In[ ]:





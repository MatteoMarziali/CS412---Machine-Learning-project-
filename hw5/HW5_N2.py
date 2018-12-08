#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


dataset = pd.read_csv('./young-people-survey/responses.csv')
dataset.head()


# In[4]:


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

# In[5]:


pd.value_counts(dataset.Smoking)


# In[6]:


pd.value_counts(dataset.Alcohol)


# In[7]:


pd.value_counts(dataset.Lying)


# In[8]:


pd.value_counts(dataset.Punctuality)


# In[9]:


pd.value_counts(dataset.Education)


# given this, we can merge doctorate degree with masters and currently a primary school pupil with primary school

# In[10]:


pd.value_counts(dataset['Internet usage'])


# we can merge no time at all with less than one hour

# In[11]:


pd.value_counts(dataset.Age)


# As we can see from Ages values, is not possible to have someone that is currently a primary school pupil, can this is an error?

# In[12]:


pd.value_counts(dataset.Weight)


# In[13]:


pd.value_counts(dataset.Height)


# In[14]:


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
        return 2
    elif(value=='less than an hour a day'):
        return 1
    elif(value == 'most of the day'):
        return 3 
    elif(value == 'no time at all'):
        return 1

def convertEducation(value):
    if(value=='currently a primary school pupil'):
        return 1
    elif(value=='primary school'):
        return 1
    elif(value == 'secondary school'):
        return 2
    elif(value == 'college/bachelor degree'):
        return 3
    elif(value == 'masters degree'):
        return 4
    elif(value == 'doctorate degree'):
        return 5


# In[15]:


dataset['Smoking'] = dataset.Smoking.apply(convertSmoking)
dataset['Alcohol'] = dataset.Alcohol.apply(convertAlcohol)
dataset['Lying'] = dataset.Lying.apply(convertLying)
dataset['Punctuality'] = dataset.Punctuality.apply(convertPunctuality)
dataset['Education'] = dataset.Education.apply(convertEducation)
dataset['Internet usage'] = dataset['Internet usage'].apply(convertInternetUsage)


# In[16]:


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


# In[17]:


dataset['Gender'] = dataset['Gender'].apply(GenderConversion)
dataset['Only child'] = dataset['Only child'].apply(OnlyChildConversion)
dataset['Left - right handed'] = dataset['Left - right handed'].apply(LRHandedConversion)
dataset['Village - town'] = dataset['Village - town'].apply(VillageTownConversion)
dataset['House - block of flats'] = dataset['House - block of flats'].apply(HouseBoFlatsConversion)
#for f in numerical:
#    if f!='Age' and f!='Number of siblings' and f!='Weight' and f!='Height':
#        dataset[f] = dataset[f].apply(Range1_5ValuesConversion)
dataset['Empathy'] = dataset['Empathy'].apply(Range1_5ValuesConversion)


# # Missing values analysis

# In[18]:


df_na = (dataset.isnull().sum() / len(dataset)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data


# In[19]:


dataset = dataset.apply(lambda x: x.fillna(x.median()),axis=0)


# In[20]:


dataset.isnull().values.any()


# In[21]:


dataset.to_csv("preprocessedDataset2.csv")


# Let's now explore the feature correlation in the dataset

# In[22]:


d=dataset.assign(C=dataset.Empathy.astype('category').cat.codes).corr()


# In[23]:


d['C']


# In[56]:


plt.rcParams['figure.figsize'] = [150.0,150.0]
plt.rcParams['figure.dpi']=70
covariance=dataset.corr(method='pearson')
sns.set(font_scale=4)
sns.heatmap(covariance,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# As we can see in the correlation plot, features are quite uncorrelated, with few exceptions such as 'shopping' and 'finances'that seem to be strongly correlated. 
# On the other hand, what we may notice is that there are groups of features that seems to be correlated, in particular, if you have a deeper look, it seems that these kind of groups are also semantically correlated in fact we can highlight some areas: Music, Movies, Interests,Fears, and Personal attitudes, Spending.
# what we can do right now is selecting this groups and performing PCA over them, to look for the main directions along with the most part of information in groups is, in order to try to reduce the number of features.

# In[57]:


music = dataset.iloc[:,0:19] 
movies = dataset.iloc[:,19:31] 
fears = dataset.iloc[:,63:73] 
interests = dataset.iloc[:,31:63] 
personal = dataset.iloc[:, 76:133] 
spending = dataset.iloc[:,133:140]


# In[58]:


other = pd.concat([dataset.iloc[:,73:76],dataset.iloc[:,139:150]],axis=1)


# In[59]:


other.head()


# In[60]:


other.shape


# In[61]:


covarianceMusic=music.corr(method='pearson')
covarianceMovies=movies.corr(method='pearson')
covarianceFears=fears.corr(method='pearson')
covarianceInterests=interests.corr(method='pearson')
covariancePersonal=personal.corr(method='pearson')
covarianceSpending=spending.corr(method='pearson')


# In[62]:


sns.set(font_scale=10)
plt.rcParams['figure.dpi']=20


# In[63]:


sns.heatmap(covarianceMusic,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[64]:


from sklearn.decomposition import PCA

n_components=12
p_list = ['pMusic'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(music)
musicPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[65]:


sns.heatmap(covarianceMovies,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[66]:


from sklearn.decomposition import PCA

n_components=8
p_list = ['pMovies'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(movies)
moviesPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[67]:


sns.heatmap(covarianceFears,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[68]:


from sklearn.decomposition import PCA

n_components=7
p_list = ['pFears'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(fears)
fearsPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[69]:


sns.heatmap(covarianceInterests,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[70]:


from sklearn.decomposition import PCA

n_components=20
p_list = ['pInterests'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(interests)
interestsPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[71]:


sns.heatmap(covariancePersonal,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[72]:


from sklearn.decomposition import PCA

n_components=37
p_list = ['pPersonal'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(personal)
personalPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[73]:


sns.heatmap(covarianceSpending,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[74]:


from sklearn.decomposition import PCA

n_components=4
p_list = ['pSpending'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(spending)
spendingPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# We can now merge the different PCA datasets and train the algorithms on this new dataset

# In[75]:


dfPCAs = pd.concat([musicPCADf,moviesPCADf,fearsPCADf,interestsPCADf,personalPCADf,spendingPCADf,other],axis=1)
dfPCAs.head()


# This way we have reduced the dimension of the dataset from 150 to 101 columns, let's see how this impact performances

# In[76]:


from sklearn.model_selection import train_test_split
X_trainPCAs, X_testPCAs, y_trainPCAs, y_testPCAs = train_test_split(dfPCAs, y, test_size=0.2,random_state = 67)
print(X_trainPCAs.shape, y_trainPCAs.shape)
print(X_testPCAs.shape, y_testPCAs.shape)


# ## KNN

# In[54]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=29)  
classifier.fit(X_trainPCAs, y_trainPCAs)  


# In[55]:


y_pred = classifier.predict(X_testPCAs) 
classifier.score(X_testPCAs,y_testPCAs)


# ## XGBoost

# In[578]:


model = xgb.XGBClassifier(n_estimators=200)
model.fit(X_trainPCAs, y_trainPCAs)


# In[579]:


model.score(X_testPCAs,y_testPCAs)


# ## Random Forests

# In[600]:


from sklearn.model_selection import cross_val_score
for n in [50,100,150,200]:
    for d in range(1,20):
        
        clf = RandomForestClassifier(n_estimators=n, max_depth=d,
                             random_state=0)
        scores = cross_val_score(clf, dfPCAs, y, cv=10)
        print(str(np.mean(scores))+", depth= "+str(d)+", n_estim= "+str(n))


# In[601]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=150, max_depth=17,
                             random_state=0)
clf.fit(X_trainPCAs, y_trainPCAs)
clf.score(X_testPCAs,y_testPCAs)


# ## Adaboost

# In[603]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=11),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_trainPCAs, y_trainPCAs)
print(bdt.score(X_testPCAs,y_testPCAs))
scores = cross_val_score(bdt,dfPCAs,y,cv=10)
scores


# In[605]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# ## RIDGE 

# In[607]:


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=10).fit(X_trainPCAs, y_trainPCAs)
clf.score(X_testPCAs, y_testPCAs) 


# In[ ]:





# In[77]:


dataset['Empathy']


# In[78]:


from sklearn.decomposition import PCA

n_components=40
p_list = ['p'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
z = dataset.drop('Empathy',1)
principalComponents = pca.fit_transform(z)
principalDf = pd.DataFrame(data = principalComponents, columns = p_list)


# In[79]:


pca.explained_variance_ratio_


# In[80]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[81]:


plt.plot(var1)


# In[82]:


y = dataset.pop('Empathy') # define the target variable (dependent variable) as y
X = principalDf


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 67)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[84]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=11,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# In[85]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=29)  
classifier.fit(X_train, y_train)  


# In[86]:


y_pred = classifier.predict(X_test) 
classifier.score(X_test,y_test)


# As we can see, the information is very spreaded along different directions in fact, doing PCA, we can see that considering the first 6 features we reach a variance of about 65%, that is a quite low value, if we had see that with two or less components we could have above the 90 percent of the variance of the whole dataset we could have added the principal component to the dataset.
# However, this result, confirms what we could have seen with the correlation analysis, the information in the dataset is very spreaded around all the features. 

# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.2,random_state = 32)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[88]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=11,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# Since we get almost the same accuracy, it seems better to keep the number of attributes the lowest possible. 
# Let's try to perform PCA analysis. 

# In[89]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=29)  
classifier.fit(X_train, y_train) 


# In[90]:


y_pred = classifier.predict(X_test)  
classifier.score(X_test,y_test)


# In[95]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2,random_state = 32)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)


# In[361]:


model.score(X_test,y_test)


# We can compare the performances of doing XGBoost with or without PCA, as we can see, with PCA performances are slightly better

# In[96]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
X_trainPCA, X_testPCA, y_trainPCA, y_testPCA = train_test_split(X, y, test_size=0.2,random_state = 32)
print(X_trainPCA.shape, y_trainPCA.shape)
print(X_testPCA.shape, y_testPCA.shape)
# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_trainPCA, y_trainPCA)


# In[363]:


model.score(X_testPCA,y_testPCA)


# We can split the training set into training and dev set, making the latter 0.1 the size of the dataset and tune some Hyperparameters

# In[97]:


from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
print(X_train.shape, y_train.shape)
print(X_dev.shape, y_dev.shape)


# For example, we can try to tune the K for knn algorithm

# In[98]:


from sklearn.neighbors import KNeighborsClassifier  
for k in range(1,50):
    classifier = KNeighborsClassifier(n_neighbors=k)  
    classifier.fit(X_train, y_train)  
    print("K = "+str(k)+" score= " + str(classifier.score(X_dev,y_dev)))  


# Best k is k=31, now we can fit the model using the training set with this k and see what happens

# In[99]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=31)  
classifier.fit(X_train, y_train) 
classifier.score(X_test,y_test)


# Now we can apply the same reasoning to the Random forests, to tune the max_depth and the n_estimators

# In[100]:


from sklearn.ensemble import RandomForestClassifier

for n in [50,100,200,300,400]:
    for m in range(1,20):
        clf = RandomForestClassifier(n_estimators=n, max_depth=m,
                             random_state=0)
        clf.fit(X_train, y_train)
        print("max_depth = "+ str(m)+", n_estimators= "+str(n) + " acc= "+ str(clf.score(X_dev, y_dev)))


# We can see that the best accuracy is obtained when we have max_depth=15 and n_estimators=50, let's see how we perform building the model on the training set

# In[101]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_depth=15,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# Still, we get an accuracy that is not the best available with our dataset, to prove that:

# In[102]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# So still we can see how using a dev set for this problem is not the bast choice, considering the size of the dataset, what we get from tuning 
# hyperparameters using the dev set are almost never the best parameters independently from the algorithm in question, so, instead of tuning the 
# hyperparameters what we can do is not to waste data on the dev set and perform cross validation to get an average of our performances, 
# making them independent from the random seed used for splitting the dataset. 
# In this scenario, the tuning of the hyperparameters can be done by looking for the best average possible.

# In[104]:


X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.2,random_state = 32)
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=11,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)


# In[105]:


clf.feature_importances_


# I want to train a RandomForest classifier with the most important features given by the Random Forests on the whole dataset

# In[106]:


mostImp=[]
for i in range(len(clf.feature_importances_)):
    if(clf.feature_importances_[i]>0.007):
        mostImp.append(i)
len(mostImp)


# In[107]:


columns=[]
for i in mostImp:
    columns.append(dataset.iloc[:,i])
len(columns)


# In[108]:


newDf = pd.DataFrame()
for i in range(len(columns)):
    newDf[i] = columns[i]
newDf.head()


# In[109]:


from sklearn.model_selection import train_test_split
X_trainRFbest, X_testRFbest, y_trainRFbest, y_testRFbest = train_test_split(newDf, y, test_size=0.2,random_state = 32)


# In[110]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                             random_state=0)
clf.fit(X_trainRFbest, y_trainRFbest)
scores = cross_val_score(clf, newDf, y, cv=10)
scores


# In[111]:


clf.score(X_trainRFbest,y_trainRFbest)


# In[112]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# Let's see what happens if we do crossvalidation for random forest using the PCA dataset with the 40 most relevant dimensions

# In[113]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=11,
                             random_state=0)
clf.fit(X_trainPCA, y_trainPCA)
scores = cross_val_score(clf, principalDf, y, cv=10)
scores


# In[114]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# Right now, the best result we get is an accuracy of 0.71 with Random Forests that should perform a little bit of feature selection by themselves, What we can do to try to improve this result is changing the algorithm.
# Another two algorithms that can be useful are Adaboost that iteratively try to improve the classification and a Ridge classifier that perform feature selection by itself.

# In[115]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, y_train)
print(bdt.score(X_test,y_test))
scores = cross_val_score(bdt,dataset,y,cv=10)
scores


# In[116]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# In[118]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=10).fit(X_train, y_train)
clf.score(X_test, y_test) 


# In[120]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=50).fit_transform(dataset, y)
X_new.shape


# In[121]:


X_trainChi2, X_testChi2, y_trainChi2, y_testChi2 = train_test_split(X_new, y, test_size=0.2,random_state = 32)


# In[123]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=10).fit(X_trainChi2, y_trainChi2)
clf.score(X_testChi2, y_testChi2) 


# In[124]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
X_new2 = SelectKBest(f_classif, k=70).fit_transform(dataset, y)
X_new2.shape


# In[125]:


X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_new2, y, test_size=0.2,random_state = 32)


# In[127]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=10).fit(X_trainf, y_trainf)
clf.score(X_testf, y_testf) 


# In[128]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                             random_state=0)
clf.fit(X_trainf, y_trainf)
scores = cross_val_score(clf, X_new2, y, cv=10)
scores


# In[129]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# In[130]:


clf.score(X_trainf,y_trainf)


# In[ ]:





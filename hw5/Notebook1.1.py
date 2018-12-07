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


# given this, we can merge currently a primary school pupil with primary school, also looking the Age Column cannot be persons that are currently doing the primary school, considering that the youngest are 15.
# We cannot delete 10 rows of the dataset, even if we have to take into account that these rows can possibly contain more mistakes. 

# In[10]:


pd.value_counts(dataset['Internet usage'])


# In[11]:


pd.value_counts(dataset.Age)


# As we can see from Ages values, is not possible to have someone that is currently a primary school pupil, can this is an error?

# In[12]:


pd.value_counts(dataset.Weight)


# 165.0 and 150.0 are clearly outliers, maybe they can be heights instead of weights,I want to remove them as may be wrong

# In[13]:


dataset = dataset[dataset.Weight != 150.0]
dataset = dataset[dataset.Weight != 165.0]


# In[14]:


pd.value_counts(dataset.Height)


# 62.0 is clearly an outlier, maybe is it a weight instead of a height, I want to remove it as may be wrong.

# In[15]:


dataset = dataset[dataset.Height != 62.0]


# In[16]:


dataset.shape


# In[17]:


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


# In[18]:


dataset['Smoking'] = dataset.Smoking.apply(convertSmoking)
dataset['Alcohol'] = dataset.Alcohol.apply(convertAlcohol)
dataset['Lying'] = dataset.Lying.apply(convertLying)
dataset['Punctuality'] = dataset.Punctuality.apply(convertPunctuality)
dataset['Education'] = dataset.Education.apply(convertEducation)
dataset['Internet usage'] = dataset['Internet usage'].apply(convertInternetUsage)


# In[19]:


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


# In[20]:


dataset['Gender'] = dataset['Gender'].apply(GenderConversion)
dataset['Only child'] = dataset['Only child'].apply(OnlyChildConversion)
dataset['Left - right handed'] = dataset['Left - right handed'].apply(LRHandedConversion)
dataset['Village - town'] = dataset['Village - town'].apply(VillageTownConversion)
dataset['House - block of flats'] = dataset['House - block of flats'].apply(HouseBoFlatsConversion)
dataset['Empathy'] = dataset['Empathy'].apply(Range1_5ValuesConversion)


# # Missing values analysis

# In[21]:


df_na = (dataset.isnull().sum() / len(dataset)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data


# In[22]:


dataset = dataset.apply(lambda x: x.fillna(x.median()),axis=0)


# In[23]:


dataset.isnull().values.any()


# In[24]:


dataset.to_csv("preprocessedDataset2.csv")


# In[25]:


y=dataset['Empathy']


# Let's now explore the feature correlation in the dataset with Pearson method

# In[26]:


plt.rcParams['figure.figsize'] = [150.0,150.0]
plt.rcParams['figure.dpi']=70
covariance=dataset.corr(method='pearson')
sns.set(font_scale=4)
sns.heatmap(covariance,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# As we can see in the correlation plot, features are quite uncorrelated, with few exceptions such as 'shopping' and 'finances'that seem to be strongly correlated. 
# On the other hand, what we may notice is that there are groups of features that seems to be correlated, in particular, if you have a deeper look, it seems that these kind of groups are also semantically correlated in fact we can highlight some areas: Music, Movies, Interests,Fears, and Personal attitudes, Spending.
# what we can do right now is selecting this groups and performing PCA over them, to look for the main directions along with the most part of information in groups is, in order to try to reduce the number of features.

# In[27]:


music = dataset.iloc[:,0:19] 
movies = dataset.iloc[:,19:31] 
fears = dataset.iloc[:,63:73] 
interests = dataset.iloc[:,31:63] 
personal = dataset.iloc[:, 76:133] 
spending = dataset.iloc[:,133:140]


# In[28]:


other = pd.concat([dataset.iloc[:,73:76],dataset.iloc[:,139:150]],axis=1)


# In[29]:


other.head()


# In[30]:


other.shape


# In[31]:


covarianceMusic=music.corr(method='pearson')
covarianceMovies=movies.corr(method='pearson')
covarianceFears=fears.corr(method='pearson')
covarianceInterests=interests.corr(method='pearson')
covariancePersonal=personal.corr(method='pearson')
covarianceSpending=spending.corr(method='pearson')
covarianceOther = other.corr(method='pearson')


# In[32]:


sns.set(font_scale=10)
plt.rcParams['figure.dpi']=20


# In[33]:


sns.heatmap(covarianceMusic,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[34]:


from sklearn.decomposition import PCA

n_components=9
p_list = ['pMusic'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(other)
otherPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[35]:


sns.heatmap(covarianceMusic,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[36]:


from sklearn.decomposition import PCA

n_components=12
p_list = ['pMusic'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(music)
musicPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[37]:


sns.heatmap(covarianceMovies,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[38]:


from sklearn.decomposition import PCA

n_components=8
p_list = ['pMovies'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(movies)
moviesPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[39]:


sns.heatmap(covarianceFears,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[40]:


from sklearn.decomposition import PCA

n_components=7
p_list = ['pFears'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(fears)
fearsPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[41]:


sns.heatmap(covarianceInterests,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[42]:


from sklearn.decomposition import PCA

n_components=20
p_list = ['pInterests'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(interests)
interestsPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[43]:


del personal['Empathy']
sns.heatmap(covariancePersonal,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[44]:


from sklearn.decomposition import PCA
n_components=37
p_list = ['pPersonal'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(personal)
personalPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[45]:


sns.heatmap(covarianceSpending,square=True, cmap="Oranges",linewidths=".5",cbar_kws={"shrink": .5})


# In[46]:


from sklearn.decomposition import PCA

n_components=4
p_list = ['pSpending'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(spending)
spendingPCADf = pd.DataFrame(data = principalComponents, columns = p_list)
print(pca.explained_variance_ratio_)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[47]:


other.shape


# We can now merge the different PCA datasets and train the algorithms on this new dataset

# In[48]:


dfPCAs = pd.concat([musicPCADf,moviesPCADf,fearsPCADf,interestsPCADf,personalPCADf,spendingPCADf,otherPCADf],axis=1,sort=False)
dfPCAs.shape


# In[49]:


dfPCAs.isnull().values.any()


# This way we have reduced the dimension of the dataset from 150 to 101 columns, let's see how this impact performances

# In[50]:


from sklearn.model_selection import train_test_split
X_trainPCAs, X_testPCAs, y_trainPCAs, y_testPCAs = train_test_split(dfPCAs, y, test_size=0.2,random_state = 67)
print(X_trainPCAs.shape, y_trainPCAs.shape)
print(X_testPCAs.shape, y_testPCAs.shape)


# ## KNN

# In[51]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=29)  
classifier.fit(X_trainPCAs, y_trainPCAs)  


# In[52]:


y_pred = classifier.predict(X_testPCAs) 
classifier.score(X_testPCAs,y_testPCAs)


# ## Random Forests

# In[71]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
for n in [50,100,150,200]:
    for d in range(1,20):
        
        clf = RandomForestClassifier(n_estimators=n, max_depth=d,
                             random_state=0)
        scores = cross_val_score(clf, dfPCAs, y, cv=10)
        print(str(np.mean(scores))+", depth= "+str(d)+", n_estim= "+str(n))


# In[70]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=150, max_depth=17,
                             random_state=0)
clf.fit(X_trainPCAs, y_trainPCAs)
clf.score(X_testPCAs,y_testPCAs)


# ## Adaboost

# In[55]:


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


# In[56]:


summ=0.0
for i in scores:
    summ+=i
average = summ /len(scores)
average


# ## RIDGE 

# In[57]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=10).fit(X_trainPCAs, y_trainPCAs)
clf.score(X_testPCAs, y_testPCAs) 


# What we got, out of this different processing using PCA on each feature subset, is that regardless to the algorithm, the accuracy is not that good, 
# in fact is almost always below 70%, and our baseline is 66%. 
# Honestly, I can say that the reason could be that since we don't know which feature is more related to our target variable, but we have just plotted 
# the correlation among all the features but Empathy, we are doing something conceptually wrong. 
# In fact, let's assume that the subset Music is strongly correlated with Empathy, we are reducing its variance. 
# Moreover, reading online, I understood that pearson correlation is not the bast correlation algorithm when you are dealing with categorical variables, 
# An alternative approach can be performing PCA on the whole dataset

# In[58]:


del dataset['Empathy']
dataset.shape


# In[59]:


from sklearn.decomposition import PCA

n_components=70
p_list = ['p'+str(x) for x in range(1, n_components+1)]
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(dataset)
principalDf = pd.DataFrame(data = principalComponents, columns = p_list)


# In[60]:


pca.explained_variance_ratio_


# In[61]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1


# In[62]:


plt.plot(var1)


# In[138]:


X = principalDf


# In[139]:


X.shape


# In[140]:


X.isnull().values.all()


# In[141]:


np.isfinite(X).values.all()


# ### generating train and test split out of PCA dataset

# In[142]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## Random Forest with PCA on the whole dataset

# In[143]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
for n in [50,100,200,300,400]:
    for m in range(1,20):
        clf = RandomForestClassifier(n_estimators=n, max_depth=m,
                             random_state=0)
        scores = cross_val_score(clf, X_train, y_train, cv=10)
        print("depth = " + str(m) +", n = "+ str(n)+ ", acc = " + str(np.mean(scores)))


# In[144]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)  


# In[145]:


clf.score(X_train,y_train)


# ## KNN with PCA on the whole dataset

# In[147]:


from sklearn.neighbors import KNeighborsClassifier  
for k in range(1,50):
    classifier = KNeighborsClassifier(n_neighbors=k)  
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print("k = "+ str(k)+ ", acc = " + str(np.mean(scores)))


# In[148]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=29)  
scores = cross_val_score(classifier, principalDf, y, cv=10)
print(" acc = " + str(np.mean(scores)))


# We didn't even reach an accuracy of 0.7 both using KNN and Random Forest, we can do better without using PCA.
# Let's try to perform the same approaches, using KNN and RF on the standard prerocessed dataset

# ## KNN 

# In[152]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2,random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[162]:


from sklearn.neighbors import KNeighborsClassifier  
for k in range(1,50):
    classifier = KNeighborsClassifier(n_neighbors=k)  
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print("k = "+ str(k)+ ", acc = " + str(np.mean(scores)))


# Best k is k=32, now we can fit the model using the training set with this k and see what happens

# In[154]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=32)  
classifier.fit(X_train,y_train)
print("acc = " + str(classifier.score(X_test,y_test)))


# In[155]:


print("acc = " + str(classifier.score(X_train,y_train)))


# We got a pretty high accuracy without overfitting, how we can see looking at the accuracy on the training set, 
# however we can see how changing the random seed of the train_test_split function the results change considerably

# In[157]:


from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(dataset, y, test_size=0.2,random_state = 47)
print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)


# In[161]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=32)  
classifier.fit(X_train2,y_train2)
print("acc = " + str(classifier.score(X_test2,y_test2)))


# We don't want a classifier that is so much prone to the randomness of the train and test split, let' see how Random Forest performs

# ### Random Forest

# Now we can apply the same reasoning to the Random forests, to tune the max_depth and the n_estimators

# In[115]:


from sklearn.ensemble import RandomForestClassifier

for n in [50,100,200,300,400]:
    for m in range(1,20):
        clf = RandomForestClassifier(n_estimators=n, max_depth=m,
                             random_state=0)
        scores = cross_val_score(clf, X_train, y_train, cv=10)
        print("depth = " + str(m) +", n = "+ str(n)+ ", acc = " + str(np.mean(scores)))


# We can see that the best accuracy is obtained when we have max_depth=15 and n_estimators=200, let's see how we perform building the model on the training set and testing with cross validation on the whole dataset

# In[163]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             random_state=0)
clf.fit(X_train,y_train)
print("test acc = " + str(clf.score(X_test,y_test)))


# The best result that we got is 0.7326732673267327 with n_estimators = 200 and max_depth = 15 using Random Forests. 
# What can we check now is how things change by varying the random seed used to split the training and test set.

# In[225]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             random_state=0)
clf.fit(X_train2,y_train2)
print("test acc = " + str(clf.score(X_test2,y_test2)))


# The result are slightly worse than in the previus case, however we still got an accuracy around 70%, that is much better 
# compared with KNN, for which the accuracy dropped of 10% by changing the seed. 
# We can conclude that Random Forests are a more robust classifier, reason why until now they represent the best obtained results. 

# Another approach than we can try is to train a random forest classifier, this, will select by itself the most relevant features. 
# Then, we can pick these most frequent features and use them to feed a new Random Forest.

# ## Random Forest with Most Relevant Features

# In[226]:


X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2,random_state = 0)
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)


# In[227]:


clf.feature_importances_


# I want to train a RandomForest classifier with the most important features given by the Random Forests on the whole dataset

# In[228]:


mostImp=[]
for i in range(len(clf.feature_importances_)):
    if(clf.feature_importances_[i]>0.007):
        mostImp.append(i)
len(mostImp)


# In[229]:


columns=[]
for i in mostImp:
    columns.append(dataset.iloc[:,i])
len(columns)


# In[230]:


newDf = pd.DataFrame()
for i in range(len(columns)):
    newDf[i] = columns[i]
newDf.head()


# In[231]:


from sklearn.model_selection import train_test_split
X_trainRFbest, X_testRFbest, y_trainRFbest, y_testRFbest = train_test_split(newDf, y, test_size=0.2,random_state = 0)


# In[236]:


from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             random_state=0)
clf.fit(X_trainRFbest, y_trainRFbest)
scores = cross_val_score(clf, newDf, y, cv=10)
scores


# In[238]:


print("test acc = " + str(np.mean(scores)))
print("train acc = " + str(np.mean(scores)))


# As we can see the results are sliglthly worse than doing Random Forest on the whole dataset.

# Right now, the best result we get is an accuracy of 0.732 with Random Forests that should perform a little bit of feature selection by themselves, What we can do to try to improve this result is changing the algorithm.
# Another approach that I want to try is Ridge Classification, this kind of classifier can be helpful because it aims at reducing the magnitude of the attribute values, in our dataset we have columns like Height, Weight and age itself that can assume a wide range of values, in contrast with the most part of features that go from 1 to 5. Let's see how this works. 

# ## RIDGE Classifier

# let's start by tuning the alpha hyperparameter

# In[200]:


for i in [0.5,5,50,100,200,400,800,1000,2000]:
    clf = RidgeClassifier(alpha=i,fit_intercept=True)
    scores = cross_val_score(clf,X_train,y_train,cv=10)
    print("alpha = "+ str(i) +" , acc = " + str(np.mean(scores)))


# In[213]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000,fit_intercept=True)
clf.fit(X_train,y_train)
print("test acc = " + str(clf.score(X_test,y_test)))
print("train accuracy = " + str( clf.score(X_train, y_train)))


# In[199]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000,fit_intercept=True)
clf.fit(X_train2,y_train2)
print("test acc = " + str(clf.score(X_test2,y_test2)))
print("train accuracy = " + str( clf.score(X_train2, y_train2)))


# As we can see from the results above, not only we are not overfitting, but also, by considering two different train/test split we noticed that the Ridge Classifier results are slightly better of Random Forest ones, in fact we are overfitting less and the accuracy is comparable.
# This is the best result for now, on the train/test split with random seed '0' we got an accuracy of 0.722

# Still focusing on the problem of having so many features and so few examples, I want to try to perform feature selection selecting 
# the k best features with respect to  chi squared and f classification measures and I'm gonna apply them to the best model we have so far, 
# Ridge.

# ## Select K best w.r.t chi2 & f_classif

# In[202]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=50).fit_transform(dataset, y)
X_new.shape


# ### Ridge

# In[239]:


#### Chi-squared


# In[203]:


X_trainChi2, X_testChi2, y_trainChi2, y_testChi2 = train_test_split(X_new, y, test_size=0.2,random_state = 0)


# In[208]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000).fit(X_trainChi2, y_trainChi2)
print("test accuracy = " + str( clf.score(X_testChi2, y_testChi2) ))
print("train accuracy = " + str( clf.score(X_trainChi2, y_trainChi2)))


# What about a different split?

# In[209]:


X_trainChi2, X_testChi2, y_trainChi2, y_testChi2 = train_test_split(X_new, y, test_size=0.2,random_state = 47)


# In[245]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000).fit(X_trainChi2, y_trainChi2)
print("test accuracy = " + str( clf.score(X_testChi2, y_testChi2)))
print("train accuracy = " + str( clf.score(X_trainChi2, y_trainChi2)))


# #### f-classification

# In[215]:


from sklearn.feature_selection import f_classif
X_new2 = SelectKBest(f_classif, k=70).fit_transform(dataset, y)
X_new2.shape


# In[243]:


X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_new2, y, test_size=0.2,random_state = 0)


# In[244]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000).fit(X_trainf, y_trainf)
print("test accuracy = " + str( clf.score(X_testf, y_testf) ))
print("train accuracy = " + str( clf.score(X_trainf, y_trainf) ))


# In[221]:


X_trainf, X_testf, y_trainf, y_testf = train_test_split(X_new2, y, test_size=0.2,random_state = 47)


# In[241]:


from sklearn.linear_model import RidgeClassifier
clf = RidgeClassifier(alpha=1000).fit(X_trainf, y_trainf)
print("test accuracy = " + str( clf.score(X_testf, y_testf) ))
print("train accuracy = " + str( clf.score(X_trainf, y_trainf) ))


# What clearly comes out here is that we outperformed the previous ridge classification, in fact we got 0.777 of accuracy using selectKbest with
# f-classification measure, that is our best result so far and 0.767 using chi2 measure, considering the split correspondent to random seed '0'.
# If we change the random seed the results are comparable with Random Forest ones, however in this case we are not overfitting at all, that is a 
# valid parameter to prove that we are generalizing well.

# Looking at these results, what we can understand is that
# probably the split with random seed = 0 is a lucky one, performances with every approach are better than with a different split. 
# In particular, we may also notice that with ridge we are underfitting a little bit, in fact to obtain meaningful performances we have to push the alpha parameter till 1000, and this is clear since once we get a test accuracy higher than a training accuracy. 
# By concluding, I still want to remark how every approach is strongly affected by the limited size of the dataset, however I think that a good balance among performances, overfitting and underfitting consist in the Ridge classifier applied to the whole preprocessed dataset, without performing feature selection with chi2 or f-classification.

# ## Conclusion

# By concluding, I still want to remark how every approach is strongly affected by the limited size of the dataset, however I think that a good balance among performances, overfitting and underfitting consists in the Ridge classifier applied to the whole preprocessed dataset, without performing feature selection with chi2 or f-classification.

# In[ ]:





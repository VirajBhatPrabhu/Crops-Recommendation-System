#!/usr/bin/env python
# coding: utf-8

# # CROP RECOMMENDATION SYSTEM

# #### Import the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df =pd.read_csv('Crop_recommendation.csv')
df.head()


# In[3]:


df.tail()


# #### Checking for Null values

# In[4]:


df.isna().sum()


# #### checking the unique values in label

# In[5]:


df['label'].unique()


# #### Checking for imbalance

# In[6]:


df['label'].value_counts()


# ###### There is no imbalance

# #### Check the statistics and correation

# In[7]:


df.describe()


# In[8]:


plt.figure(figsize=(12,6))
mask = np.triu(df.corr())
sns.heatmap(df.corr(),mask=mask,annot=True)
plt.title('Correlation Heatmap',fontsize=12)
plt.show()


# ###### No multi-collinearity exists

# In[9]:


X = df.drop('label',axis=1)
y= df.label
labels = df['label']


# #### Train-Test-Split

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# #### Trying different models to check the accuracy 

# In[11]:


model = []
accuracy = []
cvscore = []
from sklearn.metrics import classification_report,accuracy_score


# #### 1. Logistic Regrression

# In[12]:


from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression(solver='lbfgs', max_iter=1000,random_state=42)
Lr.fit(X_train,y_train)
y_pred = Lr.predict(X_test)
model.append('Logistic regression')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))


from sklearn.model_selection import cross_val_score

score = cross_val_score(LogisticRegression(solver='lbfgs', max_iter=1000),X,y,cv=5)
cvscore.append(np.mean(score))
print(score)


# .

# #### 2. Gaussian Naive Bayes

# In[13]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train,y_train)
y_pred = GNB.predict(X_test)
model.append('Gaussian Naive Bayes')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score

score = cross_val_score(GNB,X,y,cv=5)
cvscore.append(np.mean(score))
print(score)


# .

# #### 2. SVC

# In[14]:


from sklearn.svm import SVC
SVC =SVC(random_state=42)
SVC.fit(X_train,y_train)
y_pred = SVC.predict(X_test)
model.append('SVC')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))
from sklearn.model_selection import cross_val_score

score = cross_val_score(SVC,X,y,cv=5)
cvscore.append(np.mean(score))
print(np.mean(score))


# .

# #### 4. Decision Tree

# In[15]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)
DTC.fit(X_train,y_train)
y_pred = DTC.predict(X_test)
model.append('Decision Tree')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))


from sklearn.model_selection import cross_val_score

score = cross_val_score(DTC,X,y,cv=5)
cvscore.append(np.mean(score))
print('Crossval Score :', np.mean(score))


# .

# #### 5.Random Forest

# In[16]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=42)
RF.fit(X_train,y_train)
y_pred = RF.predict(X_test)

model.append('Random Forest')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))


from sklearn.model_selection import cross_val_score

score = cross_val_score(RF,X,y,cv=5)
cvscore.append(np.mean(score))
print('Crossval Score :', np.mean(score))


# .

# #### 6. Adaboost

# In[17]:


from sklearn.ensemble import AdaBoostClassifier
ADB = AdaBoostClassifier(DTC,n_estimators=40,random_state=42)
ADB.fit(X_train,y_train)
y_pred = ADB.predict(X_test)

model.append('ADABOOST')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))


from sklearn.model_selection import cross_val_score

score = cross_val_score(ADB,X,y,cv=5)
cvscore.append(np.mean(score))
print('Crossval Score :', np.mean(score))


# .

# #### 7. Gradient boost

# In[18]:


from sklearn.ensemble import GradientBoostingClassifier
GBC =GradientBoostingClassifier(random_state=42,n_estimators=40)
GBC.fit(X_train,y_train)
y_pred = GBC.predict(X_test)

model.append('Gradient Boosting')
x = accuracy_score(y_test,y_pred)
accuracy.append(x)
print('Accuracy Score is :',x)
print('')
print(classification_report(y_test,y_pred))


from sklearn.model_selection import cross_val_score

score = cross_val_score(GBC,X,y,cv=5)
cvscore.append(np.mean(score))
print('Crossval Score :', np.mean(score))


# .

# In[19]:


results_df = pd.DataFrame({"Accuracy Score": accuracy,
                               "CV score": cvscore,
                               "ML Models": model})
    
results = (results_df.sort_values(by=['CV score'], ascending=False).reset_index(drop=True))    


# In[20]:


results


# #### Lets Predict

# In[26]:


data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = GNB.predict(data)
print(prediction)


# In[29]:


data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)


# In[27]:


data = np.array([[90,42,43,20.879744,82.002744,6.502985,202.935536]])
prediction = GNB.predict(data)
print(prediction)


# In[28]:


data = np.array([[90,42,43,20.879744,82.002744,6.502985,202.935536]])
prediction = RF.predict(data)
print(prediction)


# In[30]:


import pickle 

pickle.dump(GNB,open('croprecGNB.pkl','wb'))
pickle.dump(RF,open('croprecRF.pkl','wb'))
pickle.dump(ADB,open('croprecADB.pkl','wb'))


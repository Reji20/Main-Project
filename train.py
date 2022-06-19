
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


data = pd.read_csv('framingham.csv')
data = data[~data.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)


# In[10]:


data.shape


# In[12]:


from sklearn import model_selection


# In[11]:


data=data.drop(['education'], 1)


# In[13]:


X = data.drop(['TenYearCHD'], 1)
y = data['TenYearCHD']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.svm import SVC


# In[ ]:

svm_clf = SVC()
svm_clf.fit(X_train, y_train)


# In[ ]:


y_pred = svm_clf.predict(X_test) 
print(y_pred)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


print("accuracy_score: \n",accuracy_score(y_test,y_pred))
print("confusion matrix: \n",confusion_matrix(y_test,y_pred))  
print("classification report: \n",classification_report(y_test,y_pred)) 


# In[ ]:


import pickle


# In[ ]:


pickle.dump(scaler, open('std_scaler.sav', 'wb'))
pickle.dump(svm_clf, open('model_svm.sav', 'wb'))


# In[ ]:


from scipy import stats
import numpy as np
data[(np.abs(stats.zscore(data)) > 0.1).all(axis=1)]


# In[ ]:


data.describe()


# In[15]:


from matplotlib import pyplot as plt
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
data.hist(ax = ax)
fig.savefig("Visualization.png")


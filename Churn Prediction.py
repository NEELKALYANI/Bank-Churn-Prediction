#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# # Loading Dataset
# 

# In[2]:


df = pd.read_csv('Churn_Modelling.csv')
df.head()


# In[3]:


df.drop(["RowNumber","CustomerId","Surname"],axis=1,inplace=True)


# In[4]:


df.info()


# # Data Visualization

# In[5]:


Exited_percent = df['Exited'].value_counts(normalize = True)* 100
plt.figure(figsize=(8,6))
plt.pie(Exited_percent, labels = ['not exited','exited'], autopct='%.2f',explode = [0,0.1], shadow=True)
plt.title('Percentage of Exited and Not exited')
plt.axis('equal')
plt.show()


# In[6]:


sns.countplot(x='Geography', hue='Exited', data = df)
plt.show()
pd.crosstab(df['Geography'],df['Exited'])


# In[7]:


sns.countplot(x='Gender', hue='Exited', data = df)
plt.show()
pd.crosstab(df['Gender'],df['Exited'])


# In[8]:


sns.countplot(x='IsActiveMember', hue = 'Exited', data = df)
plt.show()
pd.crosstab(df['IsActiveMember'],df['Exited'])


# This is showing that less active members have exited more than active members

# In[9]:


sns.boxplot(x = 'Exited', y = 'Age', data = df )


# This Boxplot shows that older people are most likely to exit than the younger ones
# 

# In[10]:


sns.boxplot(x = 'Exited', y = 'Tenure', data = df )


# This shows at what tenure people have exited and not exited
# 

# In[11]:


sns.boxplot(x = 'Exited', y = 'Balance', data = df )


# This shows that people with higher balance have exited more

# In[12]:


sns.boxplot(x = 'Exited', y = 'EstimatedSalary', data = df )


# This shows that the estimated salary of the people who have exited and not exited is almost similar

# In[15]:


sns.boxplot(x='Exited', y='CreditScore', data = df)


# This is showing that people with less credit score have exited
# 

# In[17]:


lab_encode = LabelEncoder()
df['Geography'] = lab_encode.fit_transform(df['Geography'])
df['Gender'] = lab_encode.fit_transform(df['Gender'])


# In[18]:


df.head()


# In[20]:


corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[21]:


df.hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle('Histograms of Numeric Columns')
plt.show()


# In[23]:


X = df.drop(['Exited'], axis = 1)
y = df['Exited']


# # Standardizing the Data

# In[30]:


scaler = StandardScaler()
X= scaler.fit_transform(X)


# # Spliting the Dataset

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # Creating Logistic Regression Model for Prediction

# In[32]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg_pred = logreg.predict(X_test)


# In[34]:


print("\nAccuracy: ",accuracy_score(y_test,logreg_pred))
print("\nClassification Report: ",classification_report(y_test,logreg_pred))
print("\nConfusion Matrix: \n",confusion_matrix(y_test,logreg_pred))
print("\nROC Score: \n", roc_auc_score(y_test, logreg_pred))


# In[35]:


conf_matrix = confusion_matrix(y_test, logreg_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[36]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # Creating Random Forest Model for Prediction

# In[37]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[39]:


print("\nAccuracy: ",accuracy_score(y_test,rf_pred))
print("\nClassification Report: ",classification_report(y_test,rf_pred))
print("\nConfusion Matrix: \n",confusion_matrix(y_test,rf_pred))
print("\nROC_AUC_Score: ",roc_auc_score(y_test,rf_pred))


# In[40]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# # Creating Gradient Boosting Model for Prediction

# In[41]:


gr_boost = GradientBoostingClassifier()
gr_boost.fit(X_train, y_train)
gr_pred = gr_boost.predict(X_test)


# In[42]:


print("\nAccuracy: ",accuracy_score(y_test,gr_pred))
print("\nClassification Report: ",classification_report(y_test,gr_pred))
print("\nConfusion Matrix: \n",confusion_matrix(y_test,gr_pred))
print("\nROC_AUC_Score: ",roc_auc_score(y_test,gr_pred))


# In[44]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, gr_boost.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





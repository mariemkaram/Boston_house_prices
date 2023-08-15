#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


# In[2]:


data=pd.read_csv("boston.csv")
# print(data.head())
print(data.isnull().sum())


# In[3]:


print(data.count)


# In[4]:


features=data.iloc[:,0:-1]
# print(features)
Y=data.iloc[:,-1]


# In[5]:


#show features with Regression
plt.figure(figsize=(18, 18))

for i, idx in enumerate(data.columns[0:13]):
    plt.subplot(4,4,i+1)
    x = data[idx]
    y = data['MEDV']
    plt.plot(x, y, 'o')

    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.xlabel(idx)
    plt.ylabel('MEDV')


# In[6]:


X_train,X_test,Y_train,Y_test=train_test_split(features,Y,test_size=0.3,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[7]:


scaler=StandardScaler()
scaler.fit(X_train)
scaler.transform(X_test)


# In[8]:


#check the outliers
num_cols = X_train.select_dtypes(include='number').columns
num_cols_count = len(num_cols)
num_cols_sqrt = int(num_cols_count**0.5) + 1

fig, axs = plt.subplots(nrows=num_cols_sqrt, ncols=num_cols_sqrt, figsize=(10,10))

for i, column in enumerate(num_cols):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    sns.boxplot(X_train[column], ax=axs[row][col])
    axs[row][col].set_title(column)

# Remove empty subplots
for i in range(num_cols_count, num_cols_sqrt * num_cols_sqrt):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    fig.delaxes(axs[row][col])

plt.tight_layout()
plt.show()


# In[9]:


X_train.info()


# In[10]:


#Feature Selection
plt.figure(figsize=(16,20))
corr_matrix = pd.concat([X_train, Y_train], axis=1).corr()
print(corr_matrix)
top_feature = corr_matrix.index[abs(corr_matrix['MEDV']) > .3]
top_corr = pd.concat([X_train, Y_train], axis=1)[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature=top_feature.delete(-1)
X_train=X_train[top_feature]
X_test=X_test[top_feature]


# In[11]:


#Test DecisionTree Overfitting
#Model 1
def get_mae(max_leaf_nodes,  X_train, X_test, Y_train, Y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=10)
    model.fit(X_train, Y_train)
    preds_val = model.predict(X_test)
    preds_train = model.predict(X_train)
    # maetest = mean_absolute_error(Y_test, preds_val)
    # maetrain = mean_absolute_error(Y_train, preds_train)
    print("Test Mean absloute error of Boosting: ", mean_absolute_error(np.asarray(Y_test), preds_val))
    print("Train Mean absloute error of Boosting: ", mean_absolute_error(np.asarray(Y_train), preds_train))
    print("Model Accuracy(%): \t" + str(r2_score(Y_test, preds_val) * 100) + "%")

get_mae(30,  X_train, X_test, Y_train, Y_test)


# In[12]:


#Model 2 RandomForest
model2=RandomForestRegressor()
model2.fit(X_train[top_feature],Y_train)
TestPrediction2=model2.predict(X_test[top_feature])
print("Test Mean absloute error of RandomForest: ",mean_absolute_error(Y_test,TestPrediction2))
Train_Prediction2=model2.predict(X_train[top_feature])
print("Train Mean absloute error of RandomForest: ",mean_absolute_error(Y_train,Train_Prediction2))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, TestPrediction2) * 100) + "%")


# In[13]:


#Model 3 Ridge Regression
model3=RidgeCV()
model3.fit(X_train[top_feature],Y_train)
Test_pred3=model3.predict(X_test[top_feature])
Train_pred3= model3.predict(X_train[top_feature])
print("Test Mean absloute error of Ridge: ",mean_absolute_error(np.asarray(Y_test),Test_pred3))
print("Train Mean absloute error of Ridge: ",mean_absolute_error(np.asarray(Y_train),Train_pred3))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, Test_pred3) * 100) + "%")


# In[14]:


model4=GradientBoostingRegressor()
model4.fit(X_train[top_feature],Y_train)
Test_pred4=model4.predict(X_test[top_feature])
Train_pred4=model4.predict(X_train[top_feature])
print("Test Mean absloute error of Boosting: ",mean_absolute_error(np.asarray(Y_test),Test_pred4))
print("Train Mean absloute error of Boosting: ",mean_absolute_error(np.asarray(Y_train),Train_pred4))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, Test_pred4) * 100) + "%")


# In[15]:


model5=svr_regressor = SVR(kernel = 'rbf')
model5= svr_regressor.fit(X_train[top_feature], Y_train)
prediction = svr_regressor.predict(X_test[top_feature])
prediction_train = svr_regressor.predict(X_train[top_feature])
print("Support Vector regression model")
print("Test Mean absloute error of SVM:", mean_squared_error(np.asarray(Y_train), prediction_train))
print("Train Mean absloute error of SVM :", mean_squared_error(np.asarray(Y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction) * 100) + "%")


# In[ ]:





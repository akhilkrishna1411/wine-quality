# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:02:16 2020

@author: akshit
"""


import numpy as np
import pandas as pd
import seaaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("winequality.csv")
df.columns
df.shape
df.corr
df.isnull().sum()
df.head()
df['fixed acidity'].fillna(df['fixed acidity'].mean(),inplace=True)
df['volatile acidity'].fillna(df['volatile acidity'].mean(),inplace=True)
df['citric acid'].fillna(df['citric acid'].mean(),inplace=True)
df['residual sugar'].fillna(df['residual sugar'].mean(),inplace=True)
df['chlorides'].fillna(df['chlorides'].mean(),inplace=True)
df['pH'].fillna(df['pH'].mean(),inplace=True)
df['sulphates'].fillna(df['sulphates'].mean(),inplace=True)
sns.pairplot(df)
reviews = []
for i in df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df['Reviews'] = reviews
from collections import Counter
Counter(df['Reviews'])
pd.get_dummies(df['type']).head()
pd.concat([df, pd.get_dummies(df['type'])], axis=1).head()
pd.get_dummies(df, drop_first=True).head()
df_copy=pd.concat([df[['fixed acidity', 'volatile acidity', 'citric acid','residual sugar', 'chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','Reviews']], pd.get_dummies(df.type, drop_first=True)],axis=1)
x = df_copy.drop('Reviews',axis=1)
y = df['Reviews']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.24)
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score*100)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)

from sklearn.svm import SVC
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
lin_svc=rf.predict(x_test)
lin_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
lin_svc_acc_score = accuracy_score(y_test, rf_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)


rbf_svc = SVC(kernel='linear')
rbf_svc.fit(x_train, y_train)
rbf_svc=rf.predict(x_test)
rbf_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
rbf_svc_acc_score = accuracy_score(y_test, rf_predict)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)
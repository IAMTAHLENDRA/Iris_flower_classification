#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:03:20 2019

@author: thalendra
"""

# data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

#import data

dataset = pd.read_csv('IRIS.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#lable encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
y = y.reshape(len(y), 1)
y = onehot_encoder.fit_transform(y)
y = y[ :, :-1]

#spliting data into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#fitting trainig data set into logistic regression model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0, min_samples_leaf=5)
classifier.fit(x_train, y_train)

#predict test set results
y_pred = classifier.predict(x_test)

#decoding one hot
y_test1 = []
for i in range(30):
    if(y_pred[ i, 0] == 1):
        y_test1.append('Iris-sestosa')
    elif (y_pred[i, 1] == 1):
        y_test1.append("iris-versicolor")
    else:
        y_test1.append("iris-verginica")

y_result = []
for i in range(30):
    if(y_pred[ i, 0] == 1):
        y_result.append("Iris-sestosa")
    elif (y_pred[i, 1] == 1):
        y_result.append("iris-versicolor")
    else :
        y_result.append("iris-verginica")

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_result)

# k-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=x_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())
print("hello")

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:38:45 2018

@author: nikhi
"""

#creating all classifier in a single loop

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualizations
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal length','sepal width','petal length','petal width','label']
dataset=pd.read_csv(url,names=names)


array=dataset.values
x=array[:,0:4]
y=array[:,4]

from sklearn.cross_validation import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=7)

#confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn import model_selection

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
predictions1=knn.predict(x_test)

print(accuracy_score(y_test,predictions1)) 


#######################
a=input()
a=int(a)
if a%2==0:
    print("EVEN")
else:
    print("ODD")
    
b=input()
for i in range(len(b)+1):
    print (b[:i])
    
    
c=input()
d=len(c)
for i in range(len(c)+1):
    print (c[:d]) 
    d=d-1


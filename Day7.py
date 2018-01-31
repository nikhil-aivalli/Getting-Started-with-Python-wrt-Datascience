import pandas as pd

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal length','sepal width','petal length','petal width','label']
dataset=pd.read_csv(url,names=names)

dataset.head()
dataset.label.astype("category")

dummies=pd.get_dummies(dataset[('label')],prefix_sep='_')

#lamda function

sq=lambda x:x*x
sq(4)

cat_label=lambda x:x.astype('category')

data_lab=dataset[['label']].apply(cat_label,axis=0)

dataset.dtypes.value_counts()

from sklearn.preprocessing import MinMaxScaler

data=[[-1,2],[-0.5,6],[0,10],[1,18]]
Scaler=MinMaxScaler()
print(Scaler.fit(data))
print(Scaler.data_max_)
print(Scaler.transform(data))


from sklearn.preprocessing import Binarizer
x=[20,13,80,90,100,45]
Bin_data=Binarizer(threshold=40).fit(x)

##################################################################
import matplotlib.pyplot as plt
#step 1:load data

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal length','sepal width','petal length','petal width','label']
dataset=pd.read_csv(url,names=names)

#dimension
dataset.shape

##few elements
dataset.head()

#statistical summary
dataset.describe()

#find class distribution
dataset.groupby('label').size()

#step2: univariate analysis

dataset.hist()
dataset.plot(kind='box',subplots=True,sharex=False,sharey=False)

#step3: multivariate analysis
from pandas.tools.plotting import  scatter_matrix

scatter_matrix(dataset)
plt.show()


array=dataset.values
x=array[:,0:4]
y=array[:,4]

from sklearn.cross_validation import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=7)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
predictions1=knn.predict(x_test)

#confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,predictions1))   #90%

print(confusion_matrix(y_test,predictions1))

print(classification_report(y_test,predictions1))

##########################################################

#logistic data on iris data

import pandas as pd

from sklearn.linear_model import LinearRegression,LogisticRegression

#fitting the classifier
lr=LogisticRegression()
lr.fit(x_train,y_train)
predictions2=lr.predict(x_test)

#confusion matrix


print(accuracy_score(y_test,predictions2))   #80%

print(confusion_matrix(y_test,predictions2))

print(classification_report(y_test,predictions2))


###########
#decision tree
from sklearn.tree import DecisionTreeClassifier
#fitting the classifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
predictions3=dt.predict(x_test)
#confusion matrix

print(accuracy_score(y_test,predictions3))    #86.66%

print(confusion_matrix(y_test,predictions3))

print(classification_report(y_test,predictions3))

#################
#naive bayes
from sklearn.naive_bayes import GaussianNB
#fitting the classifier
nb=GaussianNB()
nb.fit(x_train,y_train)
predictions4=nb.predict(x_test)

#confusion matrix

print(accuracy_score(y_test,predictions4))   #83.33%

print(confusion_matrix(y_test,predictions4))

print(classification_report(y_test,predictions4))


from sklearn.naive_bayes import MultinomialNB
#fitting the classifier
nb1=MultinomialNB()
nb1.fit(x_train,y_train)
predictions4a=nb1.predict(x_test)

#confusion matrix

print(accuracy_score(y_test,predictions4a))   #83.33%

print(confusion_matrix(y_test,predictions4a))

print(classification_report(y_test,predictions4a))


#############

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


x1=np.array([3,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8,1])
x2=np.array([5,6,6,5,8,6,7,6,7,1,2,1,2,3,2,3,4])

plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

#create new plot and data
plt.plot()
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
colors=['b','g','r']
markers=['o','v','s']

#KMeans algorithm

K=3
kmenas_model=KMeans(n_clusters=K).fit(X)
kmenas_model.labels_

plt.plot()
for i,l in enumerate(kmenas_model.labels_):
    plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
    
plt.show()
###################

#kmeans for iris data

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal length','sepal width','petal length','petal width','label']
dataset=pd.read_csv(url,names=names)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

dataset.head()
x1=dataset['sepal length']
x2=dataset['sepal width']
plt.plot()

plt.xlim([np.min(x1),np.max(x1)])
plt.ylim([np.min(x2),np.max(x2)])
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

plt.plot()
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
colors=['b','g','r']
markers=['o','v','s']

#KMeans algorithm
K=3
kmeans_model=KMeans(n_clusters=K).fit(X)
plt.plot()
for i ,l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i],x2[i],color=colors[l], marker=markers[l], ls='None')
    plt.xlim([np.min(x1),np.max(x1)])
    plt.ylim([np.min(x2),np.max(x2)])
    
    
plt.show()

#####################
#random forest clasifier
from sklearn.ensemble import RandomForestClassifier
#fitting the classifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
predictions5=rf.predict(x_test)

#confusion matrix

print(accuracy_score(y_test,predictions5))   #83.33%

print(confusion_matrix(y_test,predictions5))

print(classification_report(y_test,predictions5)) 

#################################

#random forest clasifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
#fitting the classifier
rf=RandomForestClassifier()
irisRandomForest=rf.fit(x_train,y_train)

#dumping the model
joblib.dump(irisRandomForest,"MY_RF",compress=0)
#predicting the testset results
predictions5=rf.predict(x_test)

#confusion matrix

print(accuracy_score(y_test,predictions5))   #83.33%

print(confusion_matrix(y_test,predictions5))

print(classification_report(y_test,predictions5)) 

#dumping
model=joblib.load("MY_RF")
predictions6=model.predict(x_test)
print(accuracy_score(y_test,predictions6))
joblib.load.__closure__

#########################
import datetime

now = datetime.datetime.now()
print(str(now))
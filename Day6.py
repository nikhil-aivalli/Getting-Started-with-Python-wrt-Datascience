#machine learning

import numpy as np
import statsmodels.api as sm
import pandas as pd

df=pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv',index_col=0)

df.head()
df.describe()
df.info()

y=df.Employed
x=df.GNP
x=sm.add_constant(x)
est=sm.OLS(y,x)
est=est.fit()
est.summary()

est.fittedvalues

#import get_ipython().magic('matplotlib inline')

import pylab
import scipy.stats as stats
stats.probplot(est.resid)
pylab.show()

#fitting two variables
y=df.Employed
x=df[['GNP','Population']]
x=sm.add_constant(x)
est=sm.OLS(y,x)
est=est.fit()
est.summary()

#x=[df.GNP,df.Population]

#################################################################
import pandas as pd

sales=pd.read_csv('D:/IMARTICUS/R/dataset/home_data.csv',index_col=0)

sales.head()

import  seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x="sqft_living",y="price",data=sales)

len(sales)

#split
train_data=sales.sample(frac=0.8)

len(train_data)

test_data=sales.sample(frac=0.2)

#another method to split
from sklearn.cross_validation import train_test_split 

train_data,test_data=train_test_split(sales,train_size=0.8,random_state=42)

len(train_data)
len(test_data)


#create liner regression model

lr=linear_model.LinearRegression()
regr_one_feature=linear_model.LinearRegression()

train_sq=train_data['sqft_living']
train_pr=train_data['price']
train_data.sqft_living.shape

import numpy as np
train_sq=np.array([train_sq]).T
train_pr=np.array([train_pr]).T

training_data_features.shape

lr.fit(train_sq,train_pr)

test_sq=test_data['sqft_living']
test_pr=test_data['price']

test_sq=np.array([test_sq]).T
test_pr=np.array([test_pr]).T

lr.predict(test_sq)

regr_one_feature.predict(test_data)

import math
math.sqrt(np.mean((lr.predict(test_sq)-test_pr)**2))

lr.score(test_sq,test_pr)



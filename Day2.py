import numpy as np

np.absolute(-56.6)

import math

np.sqrt(169)
math.sqrt(169)

a=np.array([1,2,3,4,5],float)
b=np.array([1,2,3,4,5])

type(a)  #numpy.ndarray
type(b)  #numpy.ndarray
##############
a[:2]
a[-1]
###############
a[2:4]
a[-3:-1]
############

c=np.array([[1,2,3,4,5],[6,7,8,9,10]])
c[0,2]
c[:,4]
c[0,1:4]

c.shape   #(2, 5)  2 row 5 col
a.shape    #(5,)

len(a)
len(c)

d=c.copy()
e=np.array(range(10),float)
e.reshape(2,5)

d[-1,-1]

9 in d  #True

print(d>3)   # [[False False False  True  True]
           #   [ True  True  True  True  True]]

d[d>3]

f=d.tolist()
type(f)

g=d.tostring()
g

a[::2]  # out---array([ 1.,  3.,  5.])
 
h=np.array(range(2,10),)


a+b
a-b
a*b
a/b

i=np.array([21.121,12.21215,15.222,45.236,12.236])
i=np.floor(i)
i=np.ceil(i)

np.sum(a)
np.prod(a)
np.mean(a)
np.max(a)
np.min(a)


for val in a:
    print(val)
    
for val in d:
    print(val)
    
np.sum(d[0,:])    
np.sum(d[1,:]) 
i=0
for val in d:
    print(np.sum(d[i,:]) )
    i=i+1
    
#standard deviation
np.std(a)
#varience
np.var(a)
d
d.sum(axis=0)  #array([ 7,  9, 11, 13, 15])
d.sum(axis=1)  #array([15, 40])
d.sum(axis=2)  #'axis' entry is out of bounds

d1=np.array([[1,2,3,4,5],[6,7,8,9,10],[2,3,5,6,4]])

# 3 dimentional
a = np.arange(15).reshape(3, 5,1)
a.sum(axis=0)  
a.sum(axis=1) 
a.sum(axis=2) 

###############################
d2=np.array([[1,2],[6,7]])
d3=np.array([[1,2],[6,7]])
np.dot(d2,d3)

np.cov(d2)
np.corrcoef(d2)


np.random.random()
np.random.seed(7896)
np.random.rand(3)
np.random.rand(2,5)

#############################################################################
a3=np.array([1,2,3,4])
a4=np.array([0,0,2,1,3])
a3.take(a4)

a3.put([0,2,3,2],a4)
##############################################3
import pandas as pd

data=pd.read_csv('D:/IMARTICUS/R/dataset/Data_Case_Study_Loss_Given_Default.csv')
data
#######
da2=np.array([1,2,3,4,5])
i1=['a','b','c','d','e']

pd.Series(da,i)
#######

da1={'name':'nikhil','mobile':'7795933438','place':'hubballi'}
pd.Series(da1)

da=np.array([10,90,200,500,505,260])
i=['a','b','c','d','e','f']

se=pd.Series(da,i)
se['d']
se[3]
se['c':'e']
se[::2]
se['c','e','f']
se[-3]

max(se)
####################################################################3
da=np.array([10,90,200,500,505,260])
i=['a','b','c','d','e','f']

da2=np.array([1,2,3,4,5])
i1=['a','b','c','d','e']


df={'one':pd.Series(da,i),
    'two':pd.Series(da2,i1)
}

df

data2=pd.DataFrame(df)
data2

######

list=[{'a':1,'b':2},
      {'a':3,'b':1,'c':5},
        {'a':1,'b':2,'c':10}]
i=['first','second','third']
c=['a','b','c']
dataset=pd.DataFrame(list,index=i,columns=c)


list=[{'a':1,'b':2},
      {'a':3,'b':1,'c':5},
        {'a':1,'b':2,'c':10}]
i=['first','second','third']
c=['a','b','c1']
dataset2=pd.DataFrame(list,index=i,columns=c)

dataset.to_csv('dataset.csv')

pd.read_csv('dataset.csv')

dataset['b']
dataset[0:2]
############
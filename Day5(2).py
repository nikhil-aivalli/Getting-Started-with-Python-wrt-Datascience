# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:16:52 2018

@author: nikhi
"""

import matplotlib as plt

import matplotlib.pyplot as plt
baseball=[180,215,210,210,188,176,209,200]

plt.hist(baseball, bins=3)

plt.show()

help('matplotlib')


import numpy as np
pop=np.random.rand(6)
plt.hist(pop,bins=6)

plt.hist(pop,bins=6,color='green')
plt.hist(pop,bins=6,orientation='horizontal')
plt.xscale('log')
plt.xticks([.15,2.75])
plt.yticks([.25,3.75])

h=[10,20,30,40,50]
plt.hist(h)
plt.yticks([0.0,0.4,0.6,0.8,0.7,1.2])
plt.xticks([0,5,10,15,20,25,30,35,40,45])
plt.xlabel('VALUE OF H')
plt.ylabel('VALUE OF Y')
plt.title("H VS Y")
plt.legend(['blue'])

#for pie chart
labels='python','c++','ruby','java'
sizes=[310,30,14,110]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0,0.1,0.2,0)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,
        autopct='%21.8f%%',shadow=True,startangle=90)
        
#scatter
x=np.random.rand(5)
y=np.random.rand(5)
plt.scatter(x,y)

x=np.random.rand(5)
y=np.random.rand(5)
s1=[10,20,30,40,50]  #size
c1=x=np.random.rand(5) #color
plt.scatter(x,y,s=s1,c=c1)


#BAR

x=np.arange(4) #gives array of specified no. of elements
money=[1.5e5,2.5e6,5.5e6,2.0e7]
a,b,c,d=plt.bar(x,money)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('black')
plt.xticks(x,('A','B','C','D'))
plt.show()


x=np.arange(4) #gives array of specified no. of elements
money=[1.5e5,2.5e6,5.5e6,2.0e7]
a,b,c,d=plt.bar(x,money).scatter(True)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('black')
plt.xticks(x,('A','B','C','D'))
plt.show()


#parllel coordinates
import pandas as pd

df2=pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])

df2.plot.bar()
df2.plot.bar(stacked=True)


from pandas.tools.plotting import parallel_coordinates

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal length','sepal width','petal length','petal width','class']
dataset=pd.read_csv(url,names=names)
plt.figure()
parallel_coordinates(dataset,'class')
print(dataset.groupby('class').size())


#boxplot

df=pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
color=dict(boxes='DarkGreen',whiskers='DarkOrange',medians='DarkBlue',caps='Gray')
df.plot.box(color=color,sym='r+')

df.plot.area()



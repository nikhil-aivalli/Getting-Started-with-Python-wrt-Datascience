
###########################################################################
import pandas as pd
import numpy as np

data=pd.read_csv('D:/IMARTICUS/R/dataset/titanic/train.csv')
data

data.head()  #show top 5 rows by default
data.tail()  #show bottom 5 rows by default

#To get summary statistics
data.describe()
data.shape

v=data['Age']
v.shape

v.dtype
data.size

#to transpose
data.T
data.head()
data.T

#to get indexes
data.axes

#to find data tpes of whole dataset
data.dtypes

data.empty

data.values
data.keys

data.sum() #coloum sum
data.sum(1)  #row wise sum

data.mode()
data['Age'].mode()

data1=data.rename(columns={'PassengerId':'PID'})
data1.head()

data.rename(columns={'PassengerId':'PID'},inplace=True)
#################

df=pd.DataFrame(np.random.rand(5,3),index=[4,5,9,8,3],columns=['B','D','A'])
df

df.sort_index() #sorted according to index number
df.sort_index(axis=1)  #sorted according to column name

df.sort_index(axis=1,ascending=False)

df.sort_values(by ='D')

df.sort_values(by =['D','A'])

df.sort_values(by ='D',kind='quicksort')

df.loc[:,'A']
df.loc[4,:]
df.loc[:,['A','D']]
df.loc[[4,9,3],['A','D']]
df.loc[:,'A']>0.25

df
#Indexes location
df.iloc[0,2]
df.iloc[:,2]

df.iloc[0,0:2]
df.iloc[0:2,0:2]

# ( for both indexes and columns)(general)
df.ix [0:2,0:2]
df.ix[:,['A','D']]


#To find the missing values in titanic dataset
data['Age'].isnull()

data['Age'].notnull()

#To find the sum of the missing values
sum(data['Age'].isnull())


#to replace the missing values
dataset=data.fillna(0)
dataset.head(10)

dataset.dropna()
dataset.shape

dataset2=data
d=dataset2.dropna() #drop whole row if NA is there
d.shape

dataset3=data
d1=dataset3.dropna(axis=1)  #drop whole column if NA is there
d1.shape
##########################
#df=pd.DataFrame(np.random.rand(5,3),index=[4,5,9,8,3],columns=['B','D','A'])
df.assign(np.random.rand(),columns=['X''Z'])
df['total']=df['A']+df['D']

df['E']=np.random.rand(5)
df

#############################

data['Age'][90:200]

data[data['Age']>60]
data[data['Age']>60][['PassengerId','Age' ]]

data[data['Age'].isnull()][['PassengerId','Age']]
######################################
dataset['Age'].drop

dataset.duplicated()
dataset.drop_duplicates()
##########################

data.info()
data.Age[0:5]
data[['PassengerId','Age']][0:5]

data['Survived'].value_counts() # 0    549
                                #1    342

data['Survived'].value_counts(normalize=True)*100  #0    61.616162
                                                   #1    38.383838
pd.crosstab(data.Sex,data.Survived)   #Survived    0    1
                                      #Sex               
                                      #female     81  233
                                      #male      468  109

#pd.crosstab(data.Sex,data.Survived,normalize='index')

len(data[data.Age<=5][data.Survived==1])

data[data.Name.str.contains('Allen')]

data.Embarked.unique()
data.Embarked.value_counts(dropna=False)
data.Embarked.value_counts()

data[(data.Age<=5) & (data.Survived==1)][['Pclass','Age','Survived']][0:5]

data.groupby('Pclass')['Age'].mean()

data.groupby(['Pclass','Sex'])['Age'].mean()

data.groupby(['Pclass','Sex'])['Survived'].sum()

data.groupby(['Pclass','Sex'])['Age'].mean()

data.groupby(['Pclass','Sex'])['Age'].mean().reset_index()




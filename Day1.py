"""
Guido Vasn Rossum
1991
www.poojaangurala.com
starting with python
data types
sequence types
"""
x=5
print(x)
print(type(x))
y=5.6
print(type(y))
s='hello'
print(type(s))
b=True
print(type(b))

#sequence types
#list
l=[1,3,5,'abc']
print(type(l))

#tuple
t=('nikhil',22,'hubballi')

print(type(t))

l[1]
l[0:3]
l[2:]
l1=[1,2,3,4,5,6,7,8,9,0,12,12,223,133,33,11,33,22,'ds',(100,101,102)]
l1[2:]
l1[4]

#in operator
3 in l           #output-true
3 in t         #output-false
3 not in t      #output-true
'a' in s      #output-false
'll' in s     #output-true
'lol' in s    #output-false

#airthmatic operators
x+y
x-y
x*y
x/y
x%y
x^y
2^2  #exponent 2e^2
2**2  #power

"""
create two lists,
two tuples,
two strings
then apply '+' on them
and print
"""

list1=[1,2,3]
list2=['a','b','c']
tuple1=('x','y',3)
tuple2=(4,5,6)

print(list1+list2)
print(tuple1+tuple2)

list1+list2
tuple1+tuple2

#'*' operator
t1=t*2
l2=l*2

a= input()
print(a)

b= input(' ')
print(b)

c='HI AKXDM SAKCM'
print (c.lower())
print (len(c))

#count gives number of occurence of the specified character
print (c.count('A'))

#find gives index of first occurence of the specified character
print (c.find('A'))

print(c.isdigit())  #output-false
print(c.isalpha())  #output-false

c1=c.split()
c2=" ".join(c1)

print(type(c1)) #list
print(type(c2)) #string

str1="hi kcks kskc kmm"
str2=str1.split()

z="asdfd"
print(z+"nikhil")
z[2:5]
##########################################
str=input()

if str.isalpha():
    print(str[2:].upper()+str[:2].upper()+ "J1")
    
else :
    print("enter a valid string")
    
###################################################   
    
#function of list

l=[1,3,5,'abc']

l.append('nikhi')
l.append(3)
l.count(3)

l.remove(3)
del l[2]
l[2]='csdcs'
l[4]='jkij'
l.insert(4,56)
t=('ds',85)
l.extend(t)
l1=list(t)
l.reverse()

l4=[1,2,3,4,5,6,7,8,9,0,12,12,223,133,33,11,33,22,'ds',(100,101,102)]
l4.sort()

l4.pop(1)

l5=[2,3,4,3,43,43,432,43423,4,32423,42342,11111111111]
max(l5)
min(l5)
sum(l5)

for i in l5:
    print(i)

l5.index(4)

t1=(2,34,33,1,'abc','a')
max(t1)  #max min will not work for tuple with character value

#Dictionary

d={'name':'NIKHIL','occupation':'STUDENT','email':'nikhilaivalli@gmail.com'}
d['contact']=7795933438

del d['occupation']

d.items()
d.keys()
d.values()

d1=d
d.clear()
# it will delete contents of both d1 and d as d1 is the copy of d.....d1 has only address of d

d.get('email')
d['sacsac'] #keyerror will get
d.get('cszfsd')  #no error  return null

d['age']=22
d['age']=23

d.popitem()  # poped ('occupation', 'STUDENT')

for i in d.values():
    print(i)
###########################################    
l1=['INDIA','CHINA','USA','AUSTRELIA','SRILANKA']
l2=['10M','15M','5M','6M','1M']

d2=zip(l1,l2)
print(d2)

d2=dict(d2)


###################################################


-



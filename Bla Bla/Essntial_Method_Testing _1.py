# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:07:16 2019

@author: GS65 8RF
"""
"""
Note:
    if len(tensor) the output is dim0
    for eg. t = torch.randn(3,4)
            len(t) -->3
"""


import torch

 #How torch.max work
a = torch.randn(5,4)
b = torch.randn(5,4)
torch.max(a)
torch.max(a, b)
 
 
#python continue
#skip the iteration    
for i in range(9):
  if i == 3:
    continue
  print(i)
 
    
#append
#how data entered with append
a = torch.tensor([2,2,1,3])
b = torch.tensor([3,5,2,1])
c = torch.tensor([4,5,3,2])

t = list()
t.append(a)
t.append(b)
t.append(c)


#numpy 
#how numpy.arrange work . Note (start,stop,step)_
import numpy as np

np.arange(3)
np.arange(3,7)
np.arange(3,7,2)


 #Tuple
 #A Tuple consists of a number of values separated by commas, for instance:
t = 12345, 54321, 'hello!'
t[0]
t
# Tuples may be nested (This is the main feature.A tuple can contain many nested arrays and others)
u = t, (1, 2, 3, 4, 5)
u
u[0][1]
# Tuples are immutable( but they can contain mutable objects)
t[0] = 88888


#dictionary
#zip and dic
str1 = {1,2,3}
str2 = {"ko","ma","oo"}
# Creating a Dictionary  
Dict = {1: 'Geeks', 'name': 'For', 3: 'Geeks'} 
# accessing a element using key 
print("Acessing a element using key:") 
print(Dict['name']) 
# accessing a element using key 
print("Acessing a element using key:") 
print(Dict[1])  
# accessing a element using get() 
# method 
print("Acessing a element using get:") 
print(Dict.get(3)) 
#creating dictionay with key and value
dictionary = dict(zip(str1,str2))
print(dictionary)
 
 
#itertools.product
#it work like 
#for(x)
 #for(y)
from itertools import product as product 
for x,y in product(range(10),range(5)):
    print('%d * %d = %d '%(x,y,x*y))
    
    
#torch.nn.constant_
import torch.nn as nn

w = torch.empty(3, 5)
nn.init.constant_(w, 0.3)


#torch.expand and expand_as
x = torch.zeros(1)
y = torch.zeros(3,4)
x.expand_as(y)

t = torch.randn(5,10)
c = torch.tensor(range(10)).unsqueeze(0).expand_as(t)
c

z = torch.zeros(1)
z.expand(2,3)


#torch.cat
x = torch.randn(2, 3)
torch.cat((x, x, x), 0)
torch.cat((x, x, x), 1)

x = [torch.randn(3,4),torch.randn(2,4),torch.randn(5,4)]
x = torch.cat(x,0) #changed to tensor 
 
#torch.tensor.index_select
x = torch.randn(4,8)
index = torch.tensor([0,2,3])
x.index_select(1,index)   #index_select(dim,index)
 
 
#torch.stack
#tensors need to be in same shape
x = torch.randn(2,3,2)
y = torch.randn(2,3,2)
z = torch.stack([x,y],dim = 1)


#range
x = torch.tensor([1,22,43,45,75,76,23])
for y in range(1,len(x)):
    print (x[y])

for y in range(10):
    print(y)

#time.time
#like stopwatch
import time

t = time.time()
#take some time
t1 = time.time()-t
t1

#torch.cumsum
#cumulative sum of input tensor
t = torch.FloatTensor([3,2,1,4,5,3])
a = torch.cumsum(t,0)

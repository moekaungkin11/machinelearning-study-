# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:39:07 2019

@author: GS65 8RF
"""
import torch

#Note that for tensors more than 2 dims the final dim is the column for eg for (3,4,5,6) tensor there is 6 column
#in this you can found 4 usage that i want to remind (squeeze,torch.max when squeeze,3dim slice and a[b])
t = torch.randn(7,4)
t1 =  torch.randn(5,4)
z = t[:,:2].unsqueeze(1)
x = t1[:,:2]
u = torch.max(z,x)
z = t[:,2:].unsqueeze(1)
x = t1[:,2:]
l = torch.max(z,x)
diff = torch.clamp(u-l ,min = 0)
r = diff[:,:,0]*diff[:,:,1]
_,a = r.max(0)
_,b = r.max(1)
a[b]
 
 #if you don't set comma ':' doesn't mean all and it may work as start,stop,step rather than row and column
 # split train and test
from numpy import array
data = array([[11, 22, 33],
		[44, 55, 66],
		[77, 88, 99],
        [12, 12, 34],
        [45, 34, 86]])
# separate data
split = 2
train,test = data[:split,:],data[split:,:] #:2 mean first two row and 2: mean remaining after first 2 row
x =data[:-split,:]  #:-2 mean exclude last 2 row

#dual number in :
t = torch.randn(3,5,4)
c = t[:,1:3,2:3] #when two number in single dim,eg in dim 1-> step-1->do 1: step-2-> |1-3| step-3->retain only got from step 2
c = t[:,2:1,2:2] # we got empty tensor coz in step-2 of dim 2 the difference is 0

#testing slicing with start stop and step
#note that in slicing first argument take upper bound and 2nd arg take lower bound mean if you slip start from
#2 and end in 14 it will start from 2 and end in 13
import numpy as np
sss = np.arange(20)
print(sss[-18:16:1]) 
print(sss[2:12:1])

# separate data
X, y = data[:, :-1], data[:, -1] #:-1 all except last column,-1 mean anly last
print(X)
print(y)

# looping for total number of objects in calculate_mAP
import torch
x = [torch.tensor([2,3,1,3,4]),torch.tensor([4,5,6,5]),torch.tensor([1,2]),torch.tensor([4,5,6,4,3])]
y = list()
for i in range(len(x)):
    y.extend([i]*x[i].size(0))
    
#retain only certain number of data
t = torch.randn(7)
z = torch.randn(7,3)
top_k = 5

x,index = t.sort(dim=0, descending = True)
y = z[index][:top_k]

#model.py line576
import torch
t = torch.randn([5,12,4])
q = torch.randn([5,12,4])

for i in range(t.size(0)):
    print(q[i].size(1))
    
#model.py line 603
t = torch.tensor([33, 12, 13, 53, 23, 19, 45, 17, 16])
t1 = torch.tensor([0.1,0.7,0.23,0.74,0.5,0.6,0.2,0.4,0.8])
t[t1<0.5] = 1 
t

#utils.py line 191
x = torch.randn(10,4)
t = torch.tensor([0,0,0,1,1,1,1,2,3,3])
c = z[t==0]
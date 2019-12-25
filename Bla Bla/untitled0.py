# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 00:32:35 2019

@author: GS65 8RF
"""



import torch
true_images = list()
true_labels = [torch.tensor([0,0,1]),torch.tensor([0,3,1,2]),torch.tensor([0,1,1,0,4,3])]
for i in range(len(true_labels)):
    true_images.extend([i] * true_labels[i].size(0))
    
true_images = torch.LongTensor(true_images)
true_labels = torch.cat(true_labels,dim=0)

true_class = true_images[true_labels==0]
test_image = torch.tensor([0,0,1,2,0,1,2,0,2,1,0,1,0])
    
b = [torch.tensor([[0,0,2,1],[2,3,1,0],[4,1,2,3]]),torch.tensor([[1,0,3,2],[0,1,1,5],[2,4,0,1]]),
     torch.tensor([[3,5,1,1],[2,4,0,2],[3,3,1,0]])]

c = torch.cat(b,0)
d = true_images[test_image == true_class[2]] 
e = torch.LongTensor(range (13))[test_image == true_class[2]][3]



import torch

x = torch.zeros((3),dtype=torch.uint8)
y = torch.tensor([[0.3,0.6,0.7],[0.6,0.2,0.1],[0.4,0.7,0.8]])

for c in range(y.size(0)):
    d = y[c]>0.5
    d = torch.tensor(d,dtype=torch.uint8)
    x = torch.max(x,d)
    x[c] = 0


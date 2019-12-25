# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:17:37 2019

@author: GS65 8RF
"""
#libs
import torch

####################################    
#torch.cat
#for label or image
c = [torch.tensor([2,1,3,4]),torch.tensor([6,5,3]),torch.tensor([1,2]),torch.tensor([4,5,6,4,3])]
catc = torch.cat(c, dim=0)
#for box
#note that it is very important to announce that each tensor may be more than one dim
#i.e in 2nd tensor that is 1 dim need to include extra [] if not,you may got error 
b = [torch.tensor([[2,2,1,1],[3,6,7,4],[1,2,1,1]]),torch.tensor([[2,1,2,1]]),torch.tensor([[5,5,7,3],[9,5,4,3]])]
catb = torch.cat(b, dim=0)
####################################
#extract spectifc object and loop in calculate ap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i = [torch.tensor([2,1,4,3]),torch.tensor([1,6,1,3]),torch.tensor([9,6,1,2]),torch.tensor([2,4,3,3])]
y = list()
for x in range(len(i)):
    y.extend([x]*i[x].size(0))
y = torch.LongTensor(y).to(device)
    
l = [torch.tensor([5,3,1,3]),torch.tensor([1,3]),torch.tensor([2,1,1,2]),torch.tensor([4,1,1,4,3])]
catl = torch.cat(l, dim=0)

for i in range(4):
    z = y[catl == i]
#########################################
#PredictedConvolution class in model.py
#test line 262
from torch import nn
import torch

box = {'conv4_3':3}
loc_conv4_3 = nn.Conv2d(128, box['conv4_3']*2, kernel_size=3, padding=1)
conv4_3feat = torch.randn(2,128,4,4)
l_conv4_3 = loc_conv4_3(conv4_3feat)
########################################
#l2norm line 357
import torch

t = torch.randn(3,4,4)
x = t.pow(2).sum(dim =1 , keepdim = True)
z = t/x
#############################################
#F.softmax methodin line 444
import torch.nn.functional as F
import torch

x = torch.randn(4,5,6)
y = F.softmax(x,dim =1) 
###############################
#line 463
import torch

x = torch.randn(3,4,5)
y,z = x[1].max(dim = 0)
###############################
#extract desired variables in line 468 & 469 & 470
import torch

x = torch.randn(3,4,5)
y = x[1][:,2]
xovr = y > -1.8
novr = xovr.sum().item()
########################################################
#line 481 to 487 in nms
#note that I changed little to work that code
import torch

x = torch.zeros((3),dtype=torch.uint8)
y = torch.tensor([[0.3,0.6,0.7],[0.6,0.2,0.1],[0.4,0.7,0.8]])

for c in range(y.size(0)):
    d = y[c]>0.5
    d = torch.tensor(d,dtype=torch.uint8)
    x = torch.max(x,d)
    x[c] = 0
###############################################################
#strange usage in line 595 of model.py
#for testing there will 30 prior boxes and 4 objects
import torch

a = torch.tensor([1,0,2,2,3,2,0,1,2,3,0,1,0,3,2,
                  0,3,1,2,0,3,3,1,0,4,2,3,2,0,1])
b = torch.tensor([24,23,24,24,25])
a[b] = torch.LongTensor(range(5))
#a[b] become range(5) and wrt this 'a' parameters chages.This seem very easy but believe me it misleading very much
#Note that if b has overlap parameter a[b] change wrt range and this overlap
#for example  [b] is [24,23,24,24,25] ,a[b] is [4,0,4,4,2] and the result is [3, 1, 3, 3, 4]coz for range only a[b] is 0,1,2,3,4
#but b overlap 24 in position of 3(the last position) and a[b]changed wrt ovrlapped dim that is 3 in this case.
##########################################################
#assign labels to priors.Former priors are are assign label wrt to no: objects that mean for image that has 4 obj the labels
#will be [0,1,2,3].We need to assign real leabel from 0-20.Line 603 in model.py
import torch

t = torch.tensor([0,1,2,1,3,3,1,3,2,1,3,1,1,2,3,0])
t_l = [torch.tensor([12,2,3,4]),torch.tensor([2,12,7,4])]
t_b = [torch.tensor([[0.12,0.17,0.42,0.71],
                     [0.41,0.25,0.31,0.28],
                     [0.87,0.21,0.66,0.98],
                     [0.77,0.24,0.14,0.58]])]

t_l[0][t]
t_b[0][t]
######################################################
#line 612 and 630
import torch

t = torch.zeros(4,10)
tt = torch.ones(10)
t[1] = tt
r = t !=0
r.sum(0)
r.sum(1)
#########################################################
#line167 utils.py
import torch
true_images = list()
true_labels = [torch.randn(3),torch.randn(4),torch.randn(6)]
for i in range(len(true_labels)):
    true_images.extend([i] * true_labels[i].size(0))



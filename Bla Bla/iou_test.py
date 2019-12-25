# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 08:57:34 2019

@author: GS65 8RF
"""
import torch

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


t = torch.tensor([[-0.0368, -0.0368,  0.0632,  0.0632],
        [-0.0576, -0.0576,  0.0839,  0.0839],
        [-0.0576, -0.0222,  0.0839,  0.0485]])
t1 =  torch.tensor([[0.1578, 0.5322, 0.5553, 0.8148],
         [0.2932, 0.7165, 0.3727, 0.7871],
         [0.1590, 0.6225, 0.2112, 0.6914]])
'''
testing to understand the flow
interset flow
z = t[:,:2].unsqueeze(1)
x = t1[:,:2]
u = torch.max(z,x)
z = t[:,2:].unsqueeze(1)
x = t1[:,2:]
l = torch.max(z,x)
diff = torch.clamp(u-l ,min = 0)
r = diff[:,:,0]*diff[:,:,1]
'''
find_intersection(t,t1)
find_jaccard_overlap(t,t1)
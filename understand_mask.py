#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:25:02 2020

@author: john
"""

import numpy as np

labels = np.random.randint(0,10,8)
indices_equal = np.eye(labels.shape[0])
indices_not_equal = 1-indices_equal
i_not_equal_j = np.expand_dims(indices_not_equal,2)
i_not_equal_k = np.expand_dims(indices_not_equal,1)
j_not_equal_k = np.expand_dims(indices_not_equal,0)

distinct_indices = np.logical_and(np.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k) 

#
distinct_indices1 = np.logical_and(i_not_equal_j, i_not_equal_k) 
distinct_indices2 = np.logical_and(i_not_equal_j, j_not_equal_k) 


distinct_indices0 = np.logical_and(distinct_indices1, distinct_indices2) 

diff = np.sum(distinct_indices.astype(int) - distinct_indices0.astype(int))


label_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))

i_equal_j = np.expand_dims(label_equal, 2)
i_equal_k = np.expand_dims(label_equal, 1)

valid_labels = np.logical_and(i_equal_j, np.logical_not(i_equal_k))

mask = np.logical_and(distinct_indices, valid_labels)

###############################################################################

import torch
#labels = torch.randint(10,(8,1))
labels = torch.from_numpy(labels)
indices_equal = torch.eye(labels.shape[0])
indices_not_equal = 1-indices_equal

i_not_equal_j = torch.unsqueeze(indices_not_equal,2).bool()
i_not_equal_k = torch.unsqueeze(indices_not_equal,1).bool()
j_not_equal_k = torch.unsqueeze(indices_not_equal,0).bool()

distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k 

label_equal = torch.eq(torch.unsqueeze(labels,0),torch.unsqueeze(labels,1))

i_equal_j = torch.unsqueeze(label_equal, 2)
i_equal_k = torch.unsqueeze(label_equal, 1)

valid_labels = i_equal_j & (~i_equal_k)

mask_torch = distinct_indices & valid_labels

diff = np.sum(mask.astype(int) - mask_torch.numpy().astype(int))
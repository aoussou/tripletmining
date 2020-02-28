#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:52:23 2020

@author: john
"""

import numpy as np
import os
import torch
import torch.nn as nn
from models import MoindrotCNN
from utils import MNISTloader
from torch.utils.data import DataLoader
torch.cuda.set_device(0)


root = '../../ssd/mnist/'

Xtrain = np.load(os.path.join(root,'X_train_MNIST.npy'))
Xtrain = np.swapaxes(Xtrain,1,3)
Xtrain = np.swapaxes(Xtrain,2,3)

Xtest = np.load(os.path.join(root,'X_test_MNIST.npy'))
Xtest = np.swapaxes(Xtest,1,3)
Xtest = np.swapaxes(Xtest,2,3)

ytrain = np.load(os.path.join(root,'y_train_MNIST.npy'))
n_train = len(Xtrain)


ytest = np.load(os.path.join(root,'y_test_MNIST.npy'))
n_test = len(Xtest)    

        
train_dataset = MNISTloader(Xtrain,ytrain)
test_dataset = MNISTloader(Xtest,ytest)

x_test = train_dataset[0]
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
num_channels = 32

bce = nn.CrossEntropyLoss()

model = MoindrotCNN(num_channels)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr= 1e-3)

n_epoch = 10

for e in range(n_epoch) :

    total_training_loss = 0   
    correct_ind_train = 0
    model.train()
    for pair in train_loader :
        
        optimizer.zero_grad()   
        x,y = pair        
        x = model(x)
        _, max_ind = torch.max(x,1)
        correct_ind_train += (max_ind == y).sum().cpu().data.numpy()
        loss = bce(x,y)
        loss.backward()
        optimizer.step()
        #total_training_loss += float(loss.cpu())

    correct_ind_test = 0       
    total_test_loss = 0  
    model.eval()
    for pair in test_loader :
 
        x,y = pair        
        x = model(x)
        _, max_ind = torch.max(x,1)
        correct_ind_test += (max_ind == y).sum().cpu().data.numpy()

        #total_training_loss += float(loss.cpu())
        
    
    print(e,correct_ind_train/n_train,correct_ind_test/n_test)

    
    

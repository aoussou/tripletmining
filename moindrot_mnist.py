#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:52:23 2020

@author: john
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

Xtrain = np.load('../../data/MNIST/X_train_MNIST.npy')
Xtrain = np.swapaxes(Xtrain,1,3)
Xtrain = np.swapaxes(Xtrain,2,3)
#Xtrain = np.squeeze(Xtrain)

ytrain = np.load('../../data/MNIST/y_train_MNIST.npy')

n_train = len(Xtrain)

class MNISTloader(Dataset) :
    
    def __init__(self,X,y) :
        
        self.X = X
        self.y = y
        
    def __len__(self) :
        
        return len(self.X)
    
    
    def __getitem__(self,idx) : 
        
        Xtensor = torch.tensor(self.X[idx],device=torch.device('cuda'),dtype=torch.float)
        ytensor = torch.tensor(self.y[idx],device=torch.device('cuda'),dtype=torch.long)
        
        return [Xtensor,ytensor]
        
class MoindrotCNN(nn.Module):
    def __init__(self,n_channels):
        super(MoindrotCNN, self ).__init__( )

        self.n_channels = n_channels
        self.cnn = self.cnn_seq()
        self.dense = self.dense_seq()
        
    def cnn_seq(self) :        
        
        seq = nn.Sequential(
                nn.Conv2d(1, self.n_channels, 3,padding=1),
                nn.BatchNorm2d(self.n_channels),  
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.n_channels, self.n_channels*2, 3,padding=1),
                nn.BatchNorm2d(self.n_channels*2),  
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        
        return seq

    def dense_seq(self) :
 
        seq = nn.Sequential(nn.Linear(3136,64),nn.Linear(64,10))

        return seq
 
                    
    def forward(self,x) :
        
       x = self.cnn(x)
       x = x.view( -1,3136)
       x = self.dense(x)

       return x
        
train_dataset = MNISTloader(Xtrain,ytrain)
x_test = train_dataset[0]
batch_size = 2 #64
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
num_channels = 32

bce = nn.CrossEntropyLoss()

model = MoindrotCNN(num_channels)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr= 1e-3)

n_epoch = 10

for e in range(n_epoch) :

    total_training_loss = 0   
    
    for pair in train_loader :
        
        optimizer.zero_grad()   
        
        x,y = pair        
        x = model(x)
    
        loss = bce(x,y)
        
        loss.backward()
        optimizer.step()
        
        total_training_loss += float(loss.cpu())
        
#        print(float(loss.cpu()))
#        
#    print('*'*10)
    print(e,total_training_loss/n_train)

    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:05:01 2020

@author: john
"""
import torch.nn as nn

class MoindrotCNN(nn.Module):
    def __init__(self,n_channels):
        super(MoindrotCNN, self ).__init__( )

        self.n_channels = n_channels
        self.cnn = self.cnn_seq()
        self.dense = self.dense_seq()
        self.last_linear = nn.Linear(64,10)
        
    def cnn_seq(self) :        
        
        seq = nn.Sequential(
                nn.Conv2d(1, self.n_channels, 3,padding=1),
                nn.BatchNorm2d(self.n_channels,momentum=0.1),  
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.n_channels, self.n_channels*2, 3,padding=1),
                nn.BatchNorm2d(self.n_channels*2),  
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        
        return seq

    def dense_seq(self) :
 
        seq = nn.Sequential(nn.Linear(3136,64))

        return seq
 
    def get_embeddings(self,x) :
        
       x = self.cnn(x)
       x = x.view(-1,3136)
       x = self.dense(x)     
       
       return x
                    
    def forward(self,x) :
        
        x = self.get_embeddings(x)
        x = self.last_linear(x)

        return x
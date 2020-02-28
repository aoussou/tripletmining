#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:03:47 2020

@author: john
"""
from torch.utils.data import Dataset
import torch

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
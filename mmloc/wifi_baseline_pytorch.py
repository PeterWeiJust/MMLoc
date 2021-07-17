#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:20:41 2020

@author: weixijia
"""

import h5py
import utm
import os
import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import torch
from torch.autograd import Variable
import torch.nn as nn
from wifi_data_functions import read_wifi_data,normalisation,WifiDataset,NeuralNet

#Train_X, Train_Y=read_data(1,10)

EPOCH=100
        
train_sensor=WifiDataset(mode='train')
val_sensor=WifiDataset(mode='val')
test_sensor=WifiDataset(mode='test')

train_loader = torch.utils.data.DataLoader(dataset=train_sensor,batch_size=len(train_sensor), shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_sensor,batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cnn=NeuralNet(10)
print(cnn)  # net architecture

optimizer = torch.optim.RMSprop(cnn.parameters(), lr=0.0001)   # optimize all cnn parameters
loss_func = nn.MSELoss()                      # the target label is not one-hotted



for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
       
        b_x = Variable(b_x).float().to(device)            # reshape x to (batch, time_step, input_size)
        b_y = Variable(b_y).float().to(device)  
        b_x = b_x.view(len(train_sensor), 1, train_sensor.trainx.shape[1])
        
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            #test_output, last_layer = cnn(test_x)
           # pred_y = torch.max(test_output, 1)[1].data.numpy()
           
            #accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

test_loader = torch.utils.data.DataLoader(dataset=test_sensor,batch_size=len(test_sensor), shuffle=False)
testingset=torch.Tensor(test_loader.dataset.testx).view(len(test_loader.dataset.testx),1,102)#read testing set and reshape to 3d as tensor
prediction=cnn(testingset)
prediction=(Variable(prediction).data).cpu().numpy()#convert to numpy for plotting
plt.plot(prediction[:,0],prediction[:,1])


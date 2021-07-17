#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:21:59 2020

@author: weixijia
"""

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
from sensor_data_functions import read_data,normalisation

# Hyper-parameters
timestep=100
input_size = 3
hidden_size = 128
num_layers = 1
output_dim = 2
num_epochs = 10
learning_rate = 0.001

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self,train=12,val=13,mode="train",datapath='timestep100/',labelpath='sensordata/sensor_wifi_timestep100_',transform=torch.from_numpy):
        self.mode=mode
        self.datapath=datapath
        self.labelpath=labelpath
        self.transform=transform
        self.trainx,self.trainy=read_data(1,train)
        self.testx,self.testy=read_data(14,14)
        self.length=len(self.trainx)+len(self.testx)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            data=self.trainx[index]
            label=self.trainy[index]
    
        else:
            data=self.testx[index]
            data=self.testy[index]
        
        if self.transform is not None:
            data=self.transform(data)
            label=self.transform(label)
            
        return data,label
        
    def __len__(self):
        if self.mode=="train":
            return len(self.trainx)
        
        else:
            return len(self.testx)
        
train_sensor=SensorDataset(mode='train')

train_loader = torch.utils.data.DataLoader(dataset=train_sensor,batch_size=len(train_sensor), shuffle=False)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)     
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  
        return out

    
lstmmodel = LSTM(input_size, hidden_size, num_layers, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(lstmmodel.parameters(), lr=learning_rate)
total_step=len(train_loader)

# lstmmodel = LSTM(input_size, hidden_size, num_layers, output_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.RMSprop(lstmmodel.parameters(), lr=learning_rate)
# total_step=len(train_loader)

for epoch in range(num_epochs):
    for i, (sensors, labels) in enumerate(train_loader):
        b_x = Variable(sensors).float().to(device)            # reshape x to (batch, time_step, input_size)
        b_y = Variable(labels).float().to(device)                               # batch y
        # Forward pass
        outputs = lstmmodel(b_x)
        loss = criterion(outputs, b_y)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            #test_output, last_layer = cnn(test_x)
           # pred_y = torch.max(test_output, 1)[1].data.numpy()
           
            #accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            
import plotting_functions as pf
suf = "sensor_original"

# ---save the model---
torch.save(lstmmodel.state_dict(),  suf + ".pkl")

# ---load the model---
# lstmmodel.load_state_dict(torch.load("./model/"+suf+".pkl"))
# lstmmodel.eval()


test_sensor = SensorDataset(mode='test')
test_loader = torch.utils.data.DataLoader(dataset=test_sensor, batch_size=len(test_sensor), shuffle=False)
sensor_test_set = torch.Tensor(test_loader.dataset.testx).view(len(test_loader.dataset.testx), 100,
                                                               3)  # read testing set and reshape to 3d as tensor
test_target = torch.Tensor(test_loader.dataset.testy)

prediction = lstmmodel(sensor_test_set)
prediction = (Variable(prediction).data).cpu().numpy()  # convert to numpy for plotting
print(sensor_test_set.shape, test_target.shape)

fig = plt.figure()
plt.scatter(prediction[:, 0], prediction[:, 1], s=1)
fig.savefig(str(suf) + ".png")

# ---visualization of test set performance---
# generate the cdf and error line plot into 'graph_output' directory
pf.error_in_meter_plotcdf(target=test_target, predict=prediction, suffix=suf)
ratio = int(len(prediction))
tar_pre = pf.normalized_data_to_utm(np.hstack((test_target, prediction)))
tar = tar_pre[:ratio, :2]
pre = tar_pre[:ratio, 2:]
pf.visualization(Y_test=tar, Y_pre=pre, suffix=suf)

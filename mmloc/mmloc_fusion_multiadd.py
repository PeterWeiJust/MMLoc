import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from tensorflow import keras
from mmloc_functions import read_data,normalisation,WifiDataset,SensorDataset

train_wifi=WifiDataset(mode="train")
test_wifi=WifiDataset(mode="test")

train_sensor=SensorDataset(mode='train')
test_sensor=SensorDataset(mode='test')

train_sensor_loader = torch.utils.data.DataLoader(dataset=train_sensor,batch_size=len(train_sensor), shuffle=False)
train_wifi_loader = torch.utils.data.DataLoader(dataset=train_wifi,batch_size=len(train_sensor), shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sensor_input_size = 3
wifi_input_size = 102
hidden_size = 128
fusion_dim = 128
num_layers = 1
output_dim = 2
num_epochs = 200
learning_rate = 0.001

class SensorLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(SensorLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
    
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        
        return out[:,-1,:]
    
class WifiLayer(nn.Module):
    def __init__(self, input_size, hidden_size,dropout):
        super(WifiLayer, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, x):
        y_1 = self.relu(self.linear_1(x))
        y_2 = self.relu(self.linear_2(y_1))
        y_3 = self.relu(self.linear_3(y_2))

        return y_3
    
class MMLoc(nn.Module):
    def __init__(self, input_dims, hidden_dims, dropouts, fusion_dim, output_dim):
        super(MMLoc, self).__init__()

        # dimensions are specified in the order of sensor and wifi
        self.sensor_input = input_dims[0]
        self.wifi_input = input_dims[1]
        
        self.sensor_hidden = hidden_dims[0]
        self.wifi_hidden = hidden_dims[1]
        
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        self.sensor_prob = dropouts[0]
        self.wifi_prob = dropouts[1]

        self.sensor_subnet = SensorLayer(self.sensor_input, self.sensor_hidden, self.sensor_prob)
        self.wifi_subnet = WifiLayer(self.wifi_input, self.wifi_hidden, self.wifi_prob)
        
        self.sensor_factor = Parameter(torch.Tensor(self.sensor_hidden, self.fusion_dim)).to(device)
        self.wifi_factor = Parameter(torch.Tensor(self.wifi_hidden, self.fusion_dim)).to(device)
        
        xavier_normal(self.sensor_factor)
        xavier_normal(self.wifi_factor)
        
        self.out=nn.Linear(self.fusion_dim,self.output_dim)

    def forward(self, sensor_x, wifi_x):
        
        sensor_h=self.sensor_subnet(sensor_x)
        wifi_h=self.wifi_subnet(wifi_x)
        
        fusion_sensor = torch.matmul(sensor_h, self.sensor_factor)
        fusion_wifi = torch.matmul(wifi_h, self.wifi_factor)
        
        self.fusions = fusion_sensor + fusion_wifi 

        output = self.out(self.fusions)
        
        return output
    
mmmodel = MMLoc((sensor_input_size,wifi_input_size), (hidden_size,hidden_size), (0.7,0.7), fusion_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(mmmodel.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for sensordata, wifidata in zip(enumerate(train_sensor_loader),enumerate(train_wifi_loader)):
        i=sensordata[0]
        sensors=sensordata[1][0]
        wifi=wifidata[1][0]
        labels=sensordata[1][1]
        b_sensor = Variable(sensors).float().to(device)            
        b_wifi = Variable(wifi).float().to(device)
        b_y = Variable(labels).float().to(device)
        
        outputs = mmmodel(b_sensor,b_wifi)
        
        loss = criterion(outputs, b_y)
        print(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 23:11:07 2021

@author: zhiwei
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import visualization as v
import tensorflow as tf
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from absl import flags
from keras.models import Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard

np.random.seed(7)
'''
# Hyper-parameters
flags.DEFINE_string("scenario", default="scenarioA", help="select scenarioA or scenarioB")
flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
flags.DEFINE_integer("epoch", default="20", help="training epochs")
FLAGS = flags.FLAGS
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self,mode,scenario="scenarioA"):
        if mode=="train":
            self.data=np.load(scenario+"/overlap_timestep1000/sensor_baseline_train.npy")
            self.label=np.load(scenario+"/overlap_timestep1000/location_train.npy")
        elif mode=="val":
            self.data=np.load(scenario+"/overlap_timestep1000/sensor_baseline_val.npy")
            self.label=np.load(scenario+"/overlap_timestep1000/location_val.npy")
        else:
            self.data=np.load(scenario+"/overlap_timestep1000/sensor_baseline_test.npy")
            self.label=np.load(scenario+"/overlap_timestep1000/location_test.npy")
    
    def __getitem__(self,index):        
        return self.data[index],self.label[index]
        
    def __len__(self):
        return len(self.data)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)     
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  
        return out


def main(_):
    '''
    scenario=FLAGS.scenario
    hidden_size=FLAGS.hidden_size
    learning_rate=FLAGS.learning_rate
    batch_size=FLAGS.batch_size
    epoch=FLAGS.epoch
    '''
    scenario="scenarioA"
    input_size=3
    num_layers=1
    hidden_size=128
    output_dim=2
    learning_rate=0.005
    batch_size=100
    num_epochs=20
    
    model_name = "sensor_baseline_scenarioA"
    
    train_sensor=SensorDataset('train',scenario=scenario)
    val_sensor=SensorDataset('val',scenario=scenario)
    test_sensor=SensorDataset('test',scenario=scenario)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_sensor,batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_sensor,batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_sensor,batch_size=len(test_sensor), shuffle=False)

    lstmmodel = LSTM(input_size, hidden_size, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(lstmmodel.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        lstmmodel.train()
        #train step
        train_loss=0.0
        for i, (sensors, labels) in enumerate(train_loader):
            b_x = Variable(sensors).float().to(device)            
            b_y = Variable(labels).float().to(device)                              
            # Forward pass
            outputs = lstmmodel(b_x)
            loss = criterion(outputs, b_y)
            train_loss+=loss.data.numpy()/batch_size
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if i % 50 == 0:
                #test_output, last_layer = cnn(test_x)
               # pred_y = torch.max(test_output, 1)[1].data.numpy()
               
                #accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
        print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)
        
        #evaluation step
        lstmmodel.eval()
        val_loss=0.0
        for i, (sensors, labels) in enumerate(val_loader):
            b_x = Variable(sensors).float().to(device)           
            b_y = Variable(labels).float().to(device) 
            outputs = lstmmodel(b_x)
            loss = criterion(outputs, b_y)
            val_loss+=loss.data.numpy()/batch_size
        print('Epoch: ', epoch, '| val loss: %.4f' % val_loss)
        
        
    #test step
    
    lstmmodel.eval()
    locationtest=test_loader.dataset.label
    test_data=Variable(torch.Tensor(test_loader.dataset.data)).float().to(device)
    locPrediction=lstmmodel(test_data).detach().numpy()
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
    '''
    #save model
    model.save(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict(SensorTest,batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
    '''
if __name__ == "__main__":
    tf.app.run()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:01:12 2020

@author: mwei_archor
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import tensorflow as tf
import plotting_functions as pf
import pandas as pd
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM,Input,Dropout
from keras.optimizers import Adam, RMSprop,SGD
from keras.callbacks import Callback, TensorBoard

np.random.seed(7)

# Hyper-parameters
wifi_input_size = 102
hidden_size = 128
output_dim = 2
batch_size=100
learning_rate = 0.001
epoch=20

model_name = "wifi_scenarioA"
'''
Wifis=[]
locations=[]
for i in range(1,15):
    wifitemp,locationtemp=combine_wifi('sensordata/sensor_wifi_timestep1000_'+str(i)+'.csv',wifi_input_size)
    Wifis.append(wifitemp)
    locations.append(locationtemp)

WifiTrain=np.concatenate((Wifis[0],Wifis[1],Wifis[2],Wifis[3],Wifis[4],Wifis[5],Wifis[6],Wifis[7]),axis=0)
locationlabel=np.concatenate((locations[0],locations[1],locations[2],locations[3],locations[4],locations[5],locations[6],locations[7]),axis=0)

WifiVal=np.concatenate((Wifis[8],Wifis[9],Wifis[10],Wifis[11],Wifis[12]),axis=0)
locationval=np.concatenate((locations[8],locations[9],locations[10],locations[11],locations[12]),axis=0)

WifiTest=Wifis[13]
locationtest=locations[13]
'''
WifiTrain=np.load()
locationlabel=np.load()

WifiVal=np.load()
locationval=np.load()

WifiTest=np.load()
locationtest=np.load()

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
model = Sequential()
model.add(Dense(hidden_size,activation='relu',input_dim=wifi_input_size))
model.add(Dropout(0.5))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(hidden_size,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(activation='tanh',units=output_dim))
model.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=['acc'])

model.fit(WifiTrain, locationlabel,
                       validation_data=(WifiVal,locationval),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

#save model
model.save("scenarioAmodel/"+str(model_name)+".h5")

locPrediction = model.predict(WifiTest, batch_size=batch_size)
aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)

#print location prediction picture
pf.print_locprediction(locationtest,aveLocPrediction,model_name)
#draw cdf picture
pf.draw_cdf_picture(locationtest,locPrediction,model_name)

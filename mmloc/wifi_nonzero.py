# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:46:45 2021

@author: zhiwei
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop,SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from data_functions import combine_wifi

np.random.seed(7)

# Hyper-parameters
wifi_input_size = 102
hidden_size = 128
n_components = 200
output_dim = 2
num_layers_rbm = 3
batch_size=100
learning_rate = 0.001
epoch=100

model_name = "wifi_scenarioA_DNN_all"

Wifis,locations=combine_wifi('wifiloc/wifialloverlap.csv',wifi_input_size)

length=len(Wifis)

#split training, validation and test with 7:2:1
WifiTrain=Wifis[:length//10*7]
locationlabel=locations[:length//10*7]

WifiVal=Wifis[length//10*7:length//10*9]
locationval=locations[length//10*7:length//10*9]

WifiTest=Wifis[length//10*9:]
locationtest=locations[length//10*9:]

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

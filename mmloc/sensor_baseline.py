# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:48:28 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
from data_functions import SensorBaselineDataset
from keras.models import Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard

np.random.seed(7)
# Hyper-parameters
timestep=100
input_size = 3
hidden_size = 128
num_layers = 1
output_dim = 2
batch_size=100
LR = 0.005
epoch=200

model_name = "sensor_baseline_scenarioA"

train_sensor=SensorBaselineDataset()
SensorTrain=train_sensor.sensortrain
locationtrain=train_sensor.labeltrain
SensorVal=train_sensor.sensorval
locationval=train_sensor.labelval
SensorTest=train_sensor.sensortest
locationtest=train_sensor.labeltest

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensorlstm=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=hidden_size)(sensorinput)
sensoroutput=Dense(2)(sensorlstm)
model=Model(inputs=[sensorinput],outputs=[sensoroutput])

model.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])

model.fit(SensorTrain, locationtrain,
                       validation_data=(SensorVal,locationval),
                       epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )
#save model
model.save("scenarioAmodel/"+str(model_name)+".h5")

locPrediction = model.predict(SensorTest, batch_size=batch_size)
aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)

#print location prediction picture
pf.print_locprediction(locationtest,aveLocPrediction,model_name)
#draw cdf picture
pf.draw_cdf_picture(locationtest,locPrediction,model_name)
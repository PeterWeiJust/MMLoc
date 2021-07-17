# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:58:34 2021

@author: zhiwei
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
from data_functions import normalisation,overlap_data,read_overlap_data,downsample_data,DownsampleDataset,overlapping,SimpleDownsampling
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
# fix random seed for reproducibility
np.random.seed(7)
time_step=1000
epoch=50
batch_size=100
LR=0.05
DownSample_num=100

model_name="overlap_downsample_sensor_scenarioA"

SensorTrain1, location1 = overlapping('overlap_timestep1000/1_timestep1000_overlap900.csv',3, time_step)
SensorTrain2, location2 = overlapping('overlap_timestep1000/2_timestep1000_overlap900.csv',3, time_step)
SensorTrain3, location3 = overlapping('overlap_timestep1000/3_timestep1000_overlap900.csv',3, time_step)
SensorTrain4, location4 = overlapping('overlap_timestep1000/4_timestep1000_overlap900.csv',3, time_step)
SensorTrain5, location5 = overlapping('overlap_timestep1000/5_timestep1000_overlap900.csv',3, time_step)
SensorTrain6, location6 = overlapping('overlap_timestep1000/6_timestep1000_overlap900.csv',3, time_step)
SensorTrain7, location7 = overlapping('overlap_timestep1000/7_timestep1000_overlap900.csv',3, time_step)
SensorTrain8, location8 = overlapping('overlap_timestep1000/8_timestep1000_overlap900.csv',3, time_step)
SensorTrain9, location9 = overlapping('overlap_timestep1000/9_timestep1000_overlap900.csv',3, time_step)
SensorTrain10, location10 = overlapping('overlap_timestep1000/10_timestep1000_overlap900.csv',3, time_step)
SensorTrain11, location11 = overlapping('overlap_timestep1000/11_timestep1000_overlap900.csv',3, time_step)
SensorTrain12, location12 = overlapping('overlap_timestep1000/12_timestep1000_overlap900.csv',3, time_step)
SensorTrain13, location13 = overlapping('overlap_timestep1000/13_timestep1000_overlap900.csv',3, time_step)
SensorTrain14, location14 = overlapping('overlap_timestep1000/14_timestep1000_overlap900.csv',3, time_step)

SensorTrain1=SimpleDownsampling(SensorTrain1,DownSample_num)
SensorTrain2=SimpleDownsampling(SensorTrain2,DownSample_num)
SensorTrain3=SimpleDownsampling(SensorTrain3,DownSample_num)
SensorTrain4=SimpleDownsampling(SensorTrain4,DownSample_num)
SensorTrain5=SimpleDownsampling(SensorTrain5,DownSample_num)
SensorTrain6=SimpleDownsampling(SensorTrain6,DownSample_num)
SensorTrain7=SimpleDownsampling(SensorTrain7,DownSample_num)
SensorTrain8=SimpleDownsampling(SensorTrain8,DownSample_num)
SensorTrain9=SimpleDownsampling(SensorTrain9,DownSample_num)
SensorTrain10=SimpleDownsampling(SensorTrain10,DownSample_num)
SensorTrain11=SimpleDownsampling(SensorTrain11,DownSample_num)
SensorTrain12=SimpleDownsampling(SensorTrain12,DownSample_num)
SensorTrain13=SimpleDownsampling(SensorTrain13,DownSample_num)
SensorTrain14=SimpleDownsampling(SensorTrain14,DownSample_num)

SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8),axis=0)
location=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8),axis=0)


Sensor_val=np.concatenate((SensorTrain9,SensorTrain10,SensorTrain11,SensorTrain12,SensorTrain13),axis=0)
loc_val=np.concatenate((location9,location10,location11,location12,location13),axis=0)

model = Sequential()
model.add(LSTM(
    input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
    units=32,
))
model.add(Dense(2))
model.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])


model.fit(SensorTrain, location,
                       validation_data=(Sensor_val,loc_val),
                       epochs=epoch, batch_size=batch_size, verbose=1,
                       #shuffle=False,
                       callbacks=[TensorBoard(log_dir='Tensorboard/downsampling_300'),
                                  #EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')
                                  ]
                       )


#save model
model.save("scenarioAmodel/"+str(model_name)+".h5")

locPrediction = model.predict(SensorTrain14,batch_size=batch_size)
aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)

#print location prediction picture
pf.print_locprediction(location14,aveLocPrediction,model_name)
#draw cdf picture
pf.draw_cdf_picture(location14,locPrediction,model_name)

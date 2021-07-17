import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import plotting_functions as pf
import pandas as pd
import os
from data_functions import overlapping,SimpleDownsampling,combine_wifi
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input,ReLU,Multiply,Add
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard

# Hyper-parameters
sensor_input_size = 3
wifi_input_size = 102
hidden_size = 200
batch_size = 100
output_dim = 2
num_epochs = 200
learning_rate = 0.001
time_step=1000
DownSample_num=100

model_name = "mmloc_scenarioA_overlap"

#load downsample dataset

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

Wifis=[]
for i in range(1,15):
    wifitemp,locationtemp=combine_wifi('sensordata/sensor_wifi_timestep1000_'+str(i)+'.csv',wifi_input_size)
    Wifis.append(wifitemp)


SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8),axis=0)
locationtrain=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8),axis=0)
WifiTrain=np.concatenate((Wifis[0],Wifis[1],Wifis[2],Wifis[3],Wifis[4],Wifis[5],Wifis[6],Wifis[7]),axis=0)

SensorVal=np.concatenate((SensorTrain9,SensorTrain10,SensorTrain11,SensorTrain12,SensorTrain13),axis=0)
locationval=np.concatenate((location9,location10,location11,location12,location13),axis=0)
WifiVal=np.concatenate((Wifis[8],Wifis[9],Wifis[10],Wifis[11],Wifis[12]),axis=0)

SensorTest=SensorTrain14
locationtest=location14
WifiTest=Wifis[13]

#construct mmloc model
sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
sensoroutput=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=hidden_size)(sensorinput)

wifiinput=Input(shape=(wifi_input_size,))
wifi=Dense(hidden_size)(wifiinput)
wifi=ReLU()(wifi)
wifi=Dense(hidden_size)(wifi)
wifi=ReLU()(wifi)
wifioutput=Dense(hidden_size)(wifi)

#merge style: multiply
rssmodel=load_model("edinmodel/rssrate_edin.h5")
wifioutput=Multiply()([wifioutput,rssmodel(wifiinput)])
merge=Add()([sensoroutput,wifioutput])
#merge=concatenate([sensoroutput,wifioutput])
hidden=Dense(hidden_size,activation='relu')(merge)
output=Dense(output_dim,activation='relu')(hidden)
mmloc=Model(inputs=[sensorinput,wifiinput],outputs=[output])

mmloc.compile(optimizer=RMSprop(learning_rate),
                 loss='mse',metrics=['acc'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

mmloc.fit([SensorTrain,WifiTrain], locationtrain,
                       validation_data=([SensorVal,WifiVal],locationval),
                       epochs=num_epochs, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                       #shuffle=False,
                       )

#save model
mmloc.save("scenarioAmodel/"+str(model_name)+".h5")
'''
mmloc=load_model("scenarioAmodel/"+str(model_name)+".h5")
'''
locPrediction = mmloc.predict([SensorTest,WifiTest], batch_size=batch_size)
aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)

#visualization for error line and location prediction
pf.visualization(locationtest,locPrediction,model_name)
#print location prediction picture
pf.print_locprediction(locationtest,aveLocPrediction,model_name)
#draw cdf picture
pf.draw_cdf_picture(locationtest,locPrediction,model_name)


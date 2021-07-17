# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:20:45 2020

@author: mwei_archor
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import plotting_functions as pf
import pandas as pd
from data_functions import normalisation,overlapping,DownsampleDataset,SimpleDownsampling,combine_wifi
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras_visualizer import visualizer 
from ann_visualizer.visualize import ann_viz
from keras.callbacks import EarlyStopping, Callback, TensorBoard

wifi_input_size=102
batch_size=100
time_step=1000
DownSample_num=100

time_step=1000

downsample=DownsampleDataset()

BaselineSensorTest=downsample.sensortest
Baselinelocationtest=downsample.labeltest

WifinonzeroTest,Wifinonzerolocations=combine_wifi('wifiloc/wifialloverlap.csv',102)
length=len(WifinonzeroTest)
WifinonzeroTest=WifinonzeroTest[length//10*9:]
Wifinonzerolocationstest=Wifinonzerolocations[length//10*9:]

WifinonzeroTestms,Wifinonzerolocationsms=combine_wifi('D:/src/mmloc_microsoft/wifidata/intapril/wifi_int_all.csv',750)
length=len(WifinonzeroTestms)
WifinonzeroTestms=WifinonzeroTestms[length//10*9:]
Wifinonzerolocationstestms=Wifinonzerolocationsms[length//10*9:]


SensorTrain14, location14 = overlapping('overlap_timestep1000/14_timestep1000_overlap900.csv',3, time_step)
DownsampleSensorTest=SimpleDownsampling(SensorTrain14,100)

SensorTrain14ms, location14ms = overlapping('D:/src/mmloc_microsoft/overlap_timestep1000/14_timestep1000_overlap900.csv',3, time_step)
DownsampleSensorTestms=SimpleDownsampling(SensorTrain14ms,100)

DownsampleWifioverlapTest,Downsampleoverlaplocationtest=combine_wifi('sensordata/dis0.1/sensor_wifi_timestep1000_14.csv',102)
DownsampleWifiTest,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14.csv',102)
DownsampleWifiTest,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14.csv',102)
DownsampleWifiTest2,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti2.csv',102)
DownsampleWifiTest5,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti5.csv',102)
DownsampleWifiTest10,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti10.csv',102)
DownsampleWifiTest20,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti20.csv',102)
DownsampleWifiTest50,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti50.csv',102)
DownsampleWifiTest100,Downsamplelocationtest=combine_wifi('sensordata/sensor_wifi_timestep1000_14_ti100.csv',102)

sensor_baseline=load_model("edinmodel/sensor_baseline_edin.h5")
sensor_downsample=load_model("edinmodel/overlap_downsample_sensor_edin_mini.h5")
sensor_downsample_ms=load_model("D:/src/mmloc_microsoft/msmodel/overlap_downsample_sensor_ms_mini.h5")
wifi_all=load_model("edinmodel/wifi_edin.h5")
wifi_nonzero=load_model("edinmodel/wifi_edin_DNN_all.h5")
wifi_nonzero_ms=load_model("D:/src/mmloc_microsoft/msmodel/wifi_ms_nonzero_DNN.h5")
wifi_overlap=load_model("edinmodel/wifi_edin_overlap_DNN.h5")
mmloc_baseline=load_model("edinmodel/mmloc_edinburgh_dis0.05.h5")
mmloc_overlap=load_model("edinmodel/mmloc_edinburgh_overlap_0.05.h5")
#visualizer(mmloc_overlap, format='png', view=True)
#ann_viz(mmloc_overlap,title="mmloc")

sensorbaselineloc = sensor_baseline.predict(BaselineSensorTest, batch_size=100)
sensordownsampleloc = sensor_downsample.predict(DownsampleSensorTest, batch_size=100)
sensordownsamplelocms = sensor_downsample_ms.predict(DownsampleSensorTestms, batch_size=100)
wifinonzeromsloc = wifi_nonzero_ms.predict(WifinonzeroTestms,batch_size=100)
wifiloc = wifi_all.predict(DownsampleWifioverlapTest, batch_size=100)
wifinonzeroloc = wifi_nonzero.predict(WifinonzeroTest, batch_size=100)
wifi_overlaploc=wifi_overlap.predict(DownsampleWifiTest,batch_size=100)
mmloc_baselineloc = mmloc_baseline.predict([DownsampleSensorTest,DownsampleWifioverlapTest], batch_size=100)
mmloc_overlaploc = mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest], batch_size=100)
mmloc_ti2loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest2], batch_size=100)
mmloc_ti5loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest5], batch_size=100)
mmloc_ti10loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest10], batch_size=100)
mmloc_ti20loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest20], batch_size=100)
mmloc_ti50loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest50], batch_size=100)
mmloc_ti100loc= mmloc_overlap.predict([DownsampleSensorTest,DownsampleWifiTest100], batch_size=100)

bin_edgeses=[]
cdfs=[]
#names=["sensor_baseline","sensor_downsample","wifiall","wifi_nonzero","mmloc_baseline","mmloc_overlap"]
#colors=['yellow','blue','gray','orange','red','green']
'''
names=["sensor_baseline","wifiall","mmloc_baseline"]
colors=['green','gray','orange']
bin_edge,cdf=pf.cdfdiff(target=Baselinelocationtest, predict=sensorbaselineloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=wifiloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
#bin_edge,cdf=pf.cdfdiff(target=Wifinonzerolocationstest, predict=wifinonzeroloc)
#bin_edgeses.append(bin_edge)
#cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_baselineloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
'''
names=["MM-Loc 10Hz(Default)","MM-Loc 5Hz","MM-Loc 1Hz","Sensor","WiFi"]
colors=['blue','purple','red','green','gold']
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_overlaploc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_ti2loc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=mmloc_ti10loc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)

bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=sensordownsampleloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Wifinonzerolocationstest, predict=wifinonzeroloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
'''

names=["Sensor","WiFi"]
colors=['green','gold']
bin_edge,cdf=pf.cdfdiff(target=Downsamplelocationtest, predict=sensordownsampleloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Wifinonzerolocationstest, predict=wifinonzeroloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)

names=["Sensor","WiFi"]
colors=['green','gold']
bin_edge,cdf=pf.cdfdiff(target=location14ms, predict=sensordownsamplelocms)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
bin_edge,cdf=pf.cdfdiff(target=Wifinonzerolocationstestms, predict=wifinonzeromsloc)
bin_edgeses.append(bin_edge)
cdfs.append(cdf)
'''

fig = plt.figure()
for i in range(len(cdfs)): 
    index=-1
    if bin_edgeses[i][0:-1][-1]>25:
        index=next(x[0] for x in enumerate(bin_edgeses[i][0:-1]) if x[1] > 25)
    else:
        index=len(bin_edgeses[i][0:-1])
    plt.plot(bin_edgeses[i][0:-1][:index], cdfs[i][:index],linestyle='-', label=names[i],color=colors[i],linewidth=2)
    
plt.xlim(xmin = 0)
plt.ylim((0,1))
plt.xlabel("metres",fontsize=10)
plt.ylabel("CDF",fontsize=10)
x_ticks = np.linspace(0, 26, 14)
y_ticks = np.linspace(0, 1, 6)
plt.xticks(x_ticks,fontsize=10)
plt.yticks(y_ticks,fontsize=10)
plt.legend(names,loc='lower right')
plt.grid(True,linestyle='--',linewidth=1)
plt.title("MM-Loc CDF")
fig.savefig("edincdf/scenarioA_CDF.pdf")


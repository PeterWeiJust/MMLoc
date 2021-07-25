import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

timestep=100

class SensorBaselineDataset(torch.utils.data.Dataset):
    def __init__(self,tw=1000,slide=100):
        
        self.sensortrain,self.labeltrain,self.wifitrain=downsample_data(1,1,tw,slide)
        for i in range(2,9):
            sensortrain,labeltrain,wifitrain=downsample_data(i, i, tw, slide)
            self.sensortrain=np.concatenate((self.sensortrain, sensortrain),axis=0)
            self.labeltrain=np.concatenate((self.labeltrain, labeltrain),axis=0)
            self.wifitrain=np.concatenate((self.wifitrain, wifitrain),axis=0)

        self.sensorval,self.labelval,self.wifival=downsample_data(9,9,tw,slide)
        for i in range(10,14):
            sensorval,labelval,wifival=downsample_data(i, i, tw, slide)
            self.sensorval=np.concatenate((self.sensorval, sensorval),axis=0)
            self.labelval=np.concatenate((self.labelval, labelval),axis=0)
            self.wifival=np.concatenate((self.wifival, wifival),axis=0)
            
        self.sensortest,self.labeltest,self.wifitest=downsample_data(14,14,tw,slide)
        self.length=len(self.sensortrain)+len(self.sensorval)+len(self.sensortest)
    
    
    def __getitem__(self, index):
        if self.mode=="train":
            sensor=self.sensortrain[index]
            wifi=self.wifitrain[index]
            label=self.labeltrain[index]
        elif self.mode=="val":
            sensor=self.sensorval[index]
            wifi=self.labelval[index]
            label=self.wifival[index]
        else:
            sensor=self.sensortest[index]
            wifi=self.labeltest[index]
            label=self.wifitest[index]
            
        return sensor,label,wifi

def read_overlap_data(file_start,file_end):
    if file_start==file_end:
        label_num=count_label_num(file_start)
        path='timestep100/'+str(file_start)+'_timestep'+str(timestep)+'.csv'
        label_path='sensordata/sensor_wifi_timestep1000_'+str(file_start)+'.csv'
        dataset = pd.read_csv(path,usecols = [11,12,13,14,15])
        dataset = dataset.dropna()
        wifidata = pd.read_csv(label_path,usecols=[i for i in range(3,105)])
        wifidata = wifidata.dropna()
        dataset_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
        X=dataset.iloc[0:label_num*timestep,0:3]
        X=np.array(X)#convert df to array
        #X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        #X=normalisation(X)
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
        wifidata=np.array(wifidata)
    else:
        res =[]
        res_label =[]
        wifi = []
        for file_num in range (file_start-1,file_end):
            file_num=file_num+1
            label_num=count_label_num(file_num)
            path='timestep100/'+str(file_num)+'_timestep'+str(timestep)+'.csv'
            label_path='sensordata/sensor_wifi_timestep1000_'+str(file_num)+'.csv'
            data = pd.read_csv(path,usecols = [11,12,13,14,15])
            wifidata = pd.read_csv(label_path,usecols=[i for i in range(3,105)])
            data_label = pd.read_csv(label_path,usecols = [11,12,13,14,15])
            data = data.iloc[0:label_num*timestep,0:5]
            data_label = data_label.iloc[:,0:5]
            res.append(data)
            res_label.append(data_label)
            wifi.append(wifidata)
        dataset=pd.concat(res, axis=0)
        dataset_label=pd.concat(res_label, axis=0)
        X=dataset.iloc[:,0:3]
        X=np.array(X)#convert df to array
        #X=normalisation(X)
        #X=X.reshape((X.shape[0]//timestep,timestep,3))#reshape for lstm
        Y=dataset_label.iloc[:,3:5]
        Y=np.array(Y)#convert df to array
        Y=normalisation(Y)#get normalised value
        wifidata=pd.concat(wifi,axis=0)
    return X,Y,dataset,wifidata

def overlap_data(file_start, file_end, tw,slide):
    X,Y,input_data,wifidata=read_overlap_data(file_start, file_end)
    sensor=input_data[['AccTotal','GyrTotal','MagTotal']]
    location=input_data[['lat','lng']]
    sensor=normalisation(sensor)
    location=normalisation(location)
    input_data=np.concatenate((sensor, location), axis=1)
    inout_seq = []
    label_seq = []
    wifi_seq = []
    L = len(input_data)
    total_samples=(L-tw)//slide+1
    input_data=np.array(input_data)
    for i in range (total_samples):
        train_seq = input_data[i*slide:i*slide+tw,0:3]
        #train_seq=train_seq.reshape((1,tw,3))#reshape for lstm
        train_label = input_data[i*slide,3:5]
        inout_seq.append((train_seq))
        label_seq.append((train_label))
        wifi_seq.append((wifidata.iloc[i]))
    inout_seq=np.array(inout_seq)
    wifi_seq=np.array(wifi_seq)
    return inout_seq,label_seq,wifi_seq #return array of sensor,label and wifi

def downsample_data(file_start, file_end, tw,slide):
    X,Y,input_data,wifidata=read_overlap_data(file_start, file_end)
    input_data=np.array(input_data)
    sensor=input_data[:,0:3]
    location=input_data[:,3:5]
    sensor=normalisation(sensor)
    location=normalisation(location)
    input_data=np.concatenate((sensor, location), axis=1)
    inout_seq = []
    label_seq = []
    wifi_seq = []
    wifidata=np.array(wifidata)
    L = len(input_data)
    total_samples=(L-tw)//slide+1
    input_data=np.array(input_data)
    for i in range (total_samples):
        train_seq = input_data[i*slide:i*slide+tw,0:3]# 1000*3
        train_seq = train_seq[[99,199,299,399,499,599,699,799,899,999],:]
        #train_seq = train_seq[[0,100,200,300,400,500,600,700,800,900],:]
        #train_seq=train_seq.reshape((1,tw,3))#reshape for lstm
        train_label = input_data[i*slide,3:5]
        inout_seq.append((train_seq))
        label_seq.append((train_label))
        wifi_seq.append((wifidata[i]))
    inout_seq=np.array(inout_seq)
    label_seq=np.array(label_seq)
    wifi_seq=np.array(wifi_seq)
    return inout_seq,label_seq,wifi_seq #return array of sensor,label and wifi

def normalisation(X):#Scale data to the range of -1:1
    max_range=1
    min_range=0
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    normalized_X = X_std * (max_range- min_range) + min_range
    return normalized_X

def count_label_num(file_num):
    countlabel_path='sensordata/sensor_wifi_timestep100_'+str(file_num)+'.csv'
    dataset = pd.read_csv(countlabel_path,usecols = [11,12,13,14,15])
    label_num=dataset.shape[0]
    return label_num

def overlapping(filepath,feature_num, time_step):
    dataframe = pd.read_csv(filepath, engine='python',skipfooter=0)
    df_length = dataframe.shape[1]
    usecols = [i for i in range(2,df_length)]
    dataframe = pd.read_csv(filepath, usecols=usecols, engine='python',skipfooter=0)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    sensordata = dataset[:,0:(dataset.shape[1]-2)]
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
    location=np.column_stack((lat,lng))
    
    ttt=np.zeros(feature_num)
    if feature_num==3:
        for i in range (len(sensordata)):
            k=sensordata[i,time_step*9:time_step*10];l=sensordata[i,time_step*10:time_step*11];m=sensordata[i,time_step*11:time_step*12]
            k=k.reshape(-1,1);l=l.reshape(-1,1);m=m.reshape(-1,1);
            abc=np.column_stack((k,l,m))
            ttt=np.vstack((ttt,abc))
        ttt=ttt[1:,:]
    elif feature_num==12:
        for i in range (len(sensordata)):
            a=sensordata[i,0:time_step];b=sensordata[i,time_step:time_step*2];c=sensordata[i,time_step*2:time_step*3]
            d=sensordata[i,time_step*3:time_step*4];e=sensordata[i,time_step*4:time_step*5];f=sensordata[i,time_step*5:time_step*6]
            g=sensordata[i,time_step*6:time_step*7];h=sensordata[i,time_step*7:time_step*8];j=sensordata[i,time_step*8:time_step*9]
            k=sensordata[i,time_step*9:time_step*10];l=sensordata[i,time_step*10:time_step*11];m=sensordata[i,time_step*11:time_step*12]
            a=a.reshape(-1,1);b=b.reshape(-1,1);c=c.reshape(-1,1);       
            d=d.reshape(-1,1);e=e.reshape(-1,1);f=f.reshape(-1,1);
            g=g.reshape(-1,1);h=h.reshape(-1,1);j=j.reshape(-1,1);
            k=k.reshape(-1,1);l=l.reshape(-1,1);m=m.reshape(-1,1);    
            abc=np.column_stack((a,b,c,d,e,f,g,h,j,k,l,m))
            ttt=np.vstack((ttt,abc))
        ttt=ttt[1:,:]
        
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    ttt = scaler.fit_transform(ttt)
    location=scaler.fit_transform(location)
    lat=scaler.fit_transform(lat)
    lng=scaler.fit_transform(lng)
    SensorTrain=np.reshape(ttt, ((ttt.shape[0]//time_step), time_step, ttt.shape[1]))
    #SensorTrain=SensorTrain[:SensorTrain.shape[0]//10*10]
    #location=location[:location.shape[0]//10*10]

    return SensorTrain,location

def SimpleDownsampling(SensorTrain, downsample_num):    
    ttt=SensorTrain[0,0,:]
    for i in range(SensorTrain.shape[0]):
        for j in range(1,SensorTrain.shape[1]):
            if j*downsample_num > SensorTrain.shape[1]:
                break
            abc=SensorTrain[i,(j*downsample_num)-1,:]
            ttt=np.vstack((ttt,abc))
    ttt=ttt[1:,:]
    SensorTrain=np.reshape(ttt, (int(ttt.shape[0]/int(SensorTrain.shape[1]/downsample_num)), int(SensorTrain.shape[1]/downsample_num), ttt.shape[1]))
    return SensorTrain

def combine_wifi(filepath,wifi_input_size):
    wifidata=pd.read_csv(filepath,usecols=[i for i in range(3,3+wifi_input_size)])
    dataframe=pd.read_csv(filepath)
    scaler = MinMaxScaler(feature_range=(0, 1))
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
    location=np.column_stack((lat,lng))
    location=scaler.fit_transform(location)
    return np.array(wifidata),location
	
time_step=1000
DownSample_num=100
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
    wifitemp,locationtemp=combine_wifi('sensor_wifi_timestep1000_'+str(i)+'.csv',102)
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

np.save('overlap_timestep1000/overlap_ds_sensor_train.npy',SensorTrain)
np.save('overlap_timestep1000/overlap_ds_location_train.npy',locationtrain)
np.save('overlap_timestep1000/overlap_ds_wifi_train.npy',WifiTrain)

np.save('overlap_timestep1000/overlap_ds_sensor_val.npy',SensorVal)
np.save('overlap_timestep1000/overlap_ds_location_val.npy',locationval)
np.save('overlap_timestep1000/overlap_ds_wifi_val
.npy',WifiVal)

np.save('overlap_timestep1000/overlap_ds_sensor_train.npy',SensorTest)
np.save('overlap_timestep1000/overlap_ds_sensor_train.npy',locationtest)
np.save('overlap_timestep1000/overlap_ds_sensor_train.npy',WifiTest)

train_sensor=SensorBaselineDataset()
SensorTrain=train_sensor.sensortrain
locationtrain=train_sensor.labeltrain
SensorVal=train_sensor.sensorval
locationval=train_sensor.labelval
SensorTest=train_sensor.sensortest
locationtest=train_sensor.labeltest

np.save('overlap_timestep1000/sensor_baseline_train.npy',SensorTrain)
np.save('overlap_timestep1000/location_train.npy',locationtrain)

np.save('overlap_timestep1000/sensor_baseline_val.npy',SensorVal)
np.save('overlap_timestep1000/location_val.npy',locationval)

np.save('overlap_timestep1000/sensor_baseline_test.npy',SensorTest)
np.save('overlap_timestep1000/location_test.npy',locationtest)
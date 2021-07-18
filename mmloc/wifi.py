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
import pandas as pd
import visualization as v
from absl import flags
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM,Input,Dropout
from keras.optimizers import Adam, RMSprop,SGD
from keras.callbacks import Callback, TensorBoard

np.random.seed(7)

# Hyper-parameters
flags.DEFINE_string("scenario", default="scenarioA", help="select scenarioA or scenarioB")
flags.DEFINE_integer("wifi_input_size", default="102", help="wifi rss feature numbers")
flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
flags.DEFINE_integer("epoch", default="20", help="training epochs")
FLAGS = flags.FLAGS

def main(_):
    scenario=FLAGS.scenario
    wifi_input_size=FLAGS.wifi_input_size
    hidden_size=FLAGS.hidden_size
    learning_rate=FLAGS.learning_rate
    batch_size=FLAGS.batch_size
    epoch=FLAGS.epoch
    
    model_name = "wifi_scenarioB"
    
    WifiTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_train.npy")
    locationlabel=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_train.npy")
    
    WifiVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_val.npy")
    locationval=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_val.npy")
    
    WifiTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_test.npy")
    locationtest=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")
    
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    model = Sequential()
    model.add(Dense(hidden_size,activation='relu',input_dim=wifi_input_size))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_size,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_size,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(activation='tanh',units=2))
    model.compile(optimizer=RMSprop(learning_rate),
                     loss='mse',metrics=['acc'])
    
    model.fit(WifiTrain, locationlabel,
                           validation_data=(WifiVal,locationval),
                           epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                           #shuffle=False,
                           )
    #save model
    model.save(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict(WifiTest,batch_size=batch_size)
    #aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)

if __name__ == "__main__":
    tf.app.run()
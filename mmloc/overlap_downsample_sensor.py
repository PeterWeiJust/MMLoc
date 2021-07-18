# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:58:34 2021

@author: zhiwei
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import visualization as v
import pandas as pd
from absl import flags
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM, TimeDistributed,Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard

np.random.seed(7)

# Hyper-parameters
flags.DEFINE_string("scenario", default="scenarioA", help="select scenarioA or scenarioB")
flags.DEFINE_integer("hidden_size", default="128", help="hidden size of deep learning models")
flags.DEFINE_float("learning_rate", default="0.005", help="learning rate")
flags.DEFINE_integer("batch_size", default="100", help="training batch sizes")
flags.DEFINE_integer("epoch", default="20", help="training epochs")
FLAGS = flags.FLAGS

def main(_):
    scenario=FLAGS.scenario
    hidden_size=FLAGS.hidden_size
    learning_rate=FLAGS.learning_rate
    batch_size=FLAGS.batch_size
    epoch=FLAGS.epoch
    
    model_name="overlap_downsample_sensor_scenarioA"
    
    SensorTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_train.npy")
    locationtrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_train.npy")
    
    SensorVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_val.npy")
    locationval=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_val.npy")
    
    SensorTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_test.npy")
    locationtest=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")
    
    model = Sequential()
    model.add(LSTM(
        input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
        units=hidden_size,
    ))
    model.add(Dense(2))
    model.compile(optimizer=RMSprop(learning_rate),
                     loss='mse',metrics=['acc'])
    
    
    model.fit(SensorTrain, locationtrain,
                           validation_data=(SensorVal,locationval),
                           epochs=epoch, batch_size=batch_size, verbose=1,
                           #shuffle=False,
                           callbacks=[TensorBoard(log_dir='Tensorboard/downsampling_300'),
                                      #EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')
                                      ]
                           )
    #save model
    model.save(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict(SensorTest,batch_size=batch_size)
    #aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)

if __name__ == "__main__":
    tf.app.run()
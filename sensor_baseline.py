# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:48:28 2020

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import pandas as pd
import visualization as v
import tensorflow as tf
from absl import flags
from keras.models import Model,load_model
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
    
    model_name = "sensor_baseline_scenarioB"
    
    SensorTrain=np.load(scenario+"/overlap_timestep1000/sensor_baseline_train.npy")
    locationtrain=np.load(scenario+"/overlap_timestep1000/location_train.npy")
    
    SensorVal=np.load(scenario+"/overlap_timestep1000/sensor_baseline_val.npy")
    locationval=np.load(scenario+"/overlap_timestep1000/location_val.npy")
    
    SensorTest=np.load(scenario+"/overlap_timestep1000/sensor_baseline_test.npy")
    locationtest=np.load(scenario+"/overlap_timestep1000/location_test.npy")
    
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
    sensorlstm=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=hidden_size)(sensorinput)
    sensoroutput=Dense(2)(sensorlstm)
    model=Model(inputs=[sensorinput],outputs=[sensoroutput])
    
    model.compile(optimizer=RMSprop(learning_rate),
                     loss='mse',metrics=['acc'])
    
    model.fit(SensorTrain, locationtrain,
                           validation_data=(SensorVal,locationval),
                           epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                           #shuffle=False,
                           )
    #save model
    model.save(scenario+"/model/"+str(model_name)+".h5")
    
    locPrediction = model.predict(SensorTest,batch_size=batch_size)
    aveLocPrediction = v.get_ave_prediction(locPrediction, batch_size)
    
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)
    
if __name__ == "__main__":
    tf.app.run()
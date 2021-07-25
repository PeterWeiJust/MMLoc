import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import visualization as v
import pandas as pd
from absl import flags
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, concatenate, LSTM,Input,ReLU,Multiply,Add
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard

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
    wifi_input_size = FLAGS.wifi_input_size
    hidden_size = FLAGS.hidden_size
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    learning_rate = FLAGS.learning_rate
    
    model_name = "mmloc_scenarioA_overlap"
    
    SensorTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_train.npy")
    locationtrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_train.npy")
    WifiTrain=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_train.npy")
    
    SensorVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_val.npy")
    locationval=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_val.npy")
    WifiVal=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_val.npy")
    
    SensorTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_sensor_test.npy")
    locationtest=np.load(scenario+"/overlap_timestep1000/overlap_ds_location_test.npy")
    WifiTest=np.load(scenario+"/overlap_timestep1000/overlap_ds_wifi_test.npy")

    #construct mmloc model
    sensorinput=Input(shape=(SensorTrain.shape[1], SensorTrain.shape[2]))
    sensoroutput=LSTM(input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),units=hidden_size)(sensorinput)
    
    wifiinput=Input(shape=(wifi_input_size,))
    wifi=Dense(hidden_size)(wifiinput)
    wifi=ReLU()(wifi)
    wifi=Dense(hidden_size)(wifi)
    wifi=ReLU()(wifi)
    wifioutput=Dense(hidden_size)(wifi)
    
    merge=concatenate([sensoroutput,wifioutput])
    hidden=Dense(hidden_size,activation='relu')(merge)
    output=Dense(2,activation='relu')(hidden)
    mmloc=Model(inputs=[sensorinput,wifiinput],outputs=[output])
    
    mmloc.compile(optimizer=RMSprop(learning_rate),
                     loss='mse',metrics=['acc'])
    
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
    
    mmloc.fit([SensorTrain,WifiTrain], locationtrain,
                           validation_data=([SensorVal,WifiVal],locationval),
                           epochs=epoch, batch_size=batch_size, verbose=1,callbacks=[tensorboard]
                           #shuffle=False,
                           )
    
    #save model
    mmloc.save(scenario+"/model/"+str(model_name)+".h5")

    locPrediction = mmloc.predict([SensorTest,WifiTest], batch_size=batch_size)
    aveLocPrediction = pf.get_ave_prediction(locPrediction, batch_size)
    #visualization for error line and location prediction
    v.visualization(locationtest,locPrediction,model_name)
    #print location prediction picture
    v.print_locprediction(locationtest,aveLocPrediction,model_name,scenario)
    #draw cdf picture
    v.draw_cdf_picture(locationtest,locPrediction,model_name,scenario)

if __name__ == "__main__":
    tf.app.run()
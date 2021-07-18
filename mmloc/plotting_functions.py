#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def normalized_data_to_utm(dd):
    min_c1 = 0
    max_c1 = 39.76
    min_c2 = 0
    max_c2 = 32.63

    d1 = dd[:, 0]
    d2 = dd[:, 1]
    d3 = dd[:, 2]
    d4 = dd[:, 3]
    
    inverse_to_utm_x = lambda x: (min_c1 + x * (max_c1 - min_c1))
    inverse_to_utm_y = lambda x: (min_c2 + x * (max_c2 - min_c2))
    
    id1 = inverse_to_utm_x(d1)
    id2 = inverse_to_utm_y(d2)
    id3 = inverse_to_utm_x(d3)
    id4 = inverse_to_utm_y(d4)
    
    return np.transpose(np.vstack((np.vstack((id1, id2)), np.vstack((id3, id4)))))

def cal_error_in_meters(data):
    data = normalized_data_to_utm(data)
    errors = [np.sqrt(np.square(item[0] - item[2]) + np.square(item[1] - item[3])) for item in data]
    return errors


def cdfpic(data):
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)
    
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    
    hist=hist.astype(float)/len(data)

    cdf = np.cumsum(hist)
    
    return bin_edges,cdf

def cdfdiff(target, predict):
    target_and_predict = np.hstack((target, predict))
    error_in_meters = cal_error_in_meters(target_and_predict)    
    return cdfpic(error_in_meters)

def get_ave_prediction(locPrediction, n):
    weights = np.ones(n)
    weights /= weights.sum()
    x = np.asarray(locPrediction[:,0])
    y = np.asarray(locPrediction[:,1])  
    avelatPrediction = np.convolve(x, weights, mode='full')[:len(x)]
    avelngPrediction = np.convolve(y, weights, mode='full')[:len(y)]
    avelatPrediction[:n] = avelatPrediction[n]
    avelngPrediction[:n] = avelngPrediction[n]
    avelatPrediction=avelatPrediction.reshape(-1,1)
    avelngPrediction=avelngPrediction.reshape(-1,1)
    aveLocPrediction=np.column_stack((avelatPrediction,avelngPrediction))
    return aveLocPrediction

def visualization(locationtest, locPrediction, suffix):
    data=normalized_data_to_utm(np.hstack((locationtest, locPrediction)))
    datare1=data[:,0].reshape(-1,1)
    datare2=data[:,1].reshape(-1,1)
    datare3=data[:,2].reshape(-1,1)
    datare4=data[:,3].reshape(-1,1)
    Y_test=np.hstack((datare1,datare2))
    Y_pre=np.hstack((datare3,datare4))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True,linestyle='--',linewidth=1)
    ax.set_xlabel("x-longitude")
    ax.set_ylabel("y-latitude")

    Y_test = np.array(Y_test)
    for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
        plt.plot([pred[0], target[0]], [pred[1], target[1]], color='r',
                 linewidth=0.5, label='error line' if i == 0 else "")
        plt.scatter(pred[0], pred[1], label='prediction' if i == 0 else "", color='b', marker='.')
        plt.scatter(target[0], target[1], label='target' if i == 0 else "", color='c', marker='.')
    ax.set_title("Prection Footpath")
    ax.legend()
    # save error line fig
    fig.savefig("errors_visualization_" + str(suffix) + ".pdf")

def draw_cdf_picture(locationtest,locPrediction,model_name,scenario):
    fig=plt.figure()
    bin_edge,cdf=cdfdiff(target=locationtest,predict=locPrediction)
    plt.plot(bin_edge[0:-1],cdf,linestyle='--',label=str(model_name),color='r')
    plt.xlim(xmin = 0)
    plt.ylim((0,1))
    plt.xlabel("metres")
    plt.ylabel("CDF")
    plt.legend(str(model_name),loc='upper right')
    plt.grid(True)
    plt.title((str(model_name)+' CDF'))
    fig.savefig(scenario+"/cdf/"+str(model_name)+"_CDF.pdf")
    
def print_locprediction(locationtest,aveLocPrediction,model_name,scenario):
    fig=plt.figure()
    data=normalized_data_to_utm(np.hstack((locationtest, aveLocPrediction)))
    plt.plot(data[:,0],data[:,1],'b',data[:,2],data[:,3],'r')
    plt.legend(['target','prediction'],loc='upper right')
    plt.xlabel("x-latitude")
    plt.ylabel("y-longitude")
    plt.title(str(model_name)+" Prediction")
    fig.savefig(scenario+"/predictionpng/"+str(model_name)+"_locprediction.png")
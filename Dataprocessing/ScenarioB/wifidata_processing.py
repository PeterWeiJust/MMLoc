import os
import pandas as pd
import numpy as np
import scipy.io
import xml.dom.minidom
import collections
import sys
import warnings
import os
import xml.dom.minidom
import collections
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import matplotlib.pyplot as plt
from tqdm import tqdm

folder="wifi"

class WifiRecord(object):
    world_wifi = {}
    world_ordered_wifi = {}

    def __init__(self, path):
        self.traverse_all(path)
        self.generate_ordered_wifi()

    def scan_wifi(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
            for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
                if j % 2:
                    ap = item.childNodes[j].getAttribute("b")
                    if ap not in self.world_wifi.keys():
                        self.world_wifi[ap] = 1
                    else:
                        self.world_wifi[ap] = self.world_wifi[ap] + 1

    def traverse_all(self, path):
        dirs = os.listdir(path)
        for dir in dirs:
            if dir != '.DS_Store':
                dir = os.path.join(path, dir)
                self.scan_wifi(dir)

    def generate_ordered_wifi(self):
        self.world_ordered_wifi = collections.OrderedDict(
            sorted(self.world_wifi.items(), key=lambda t: t[1], reverse=True))


def write_ap_to_file(wr, filename):
    file = open(filename, 'w')
    for wifi_id, times, rank in zip(wr.world_ordered_wifi.keys(), wr.world_ordered_wifi.values(),
                                    range(len(wr.world_ordered_wifi))):
        file.write('{}\t{}\t{}\n'.format(str(wifi_id), str(times), str(rank + 1)))
    file.close()

def processwifi(folder,dir):
    logfile=pd.read_table(str(folder)+'/'+ dir,sep='\s+',delimiter=';',comment='%',dtype='string',header=None,error_bad_lines=False,warn_bad_lines=False,engine='python')
    wifi_id=pd.read_table('wifi_id.txt',delimiter=',',dtype='string',engine='python')
    posi_info=pd.read_table('posi_info.txt',delimiter=',')
    router_names=wifi_id.iloc[:,1]
    
    posi=logfile[logfile[0]=="POSI"]
    extract_sensertime_index=posi.index-1 #get posi previous time index as current posi sensortime index
    posi_sensortime=logfile.loc[(n for n in extract_sensertime_index),2]
    posi_sensortime=pd.DataFrame(posi_sensortime)
    posi[2]=posi_sensortime[2].values
    posi[2]=posi[2].astype(float).astype(int)
    posi[3]=posi_info['lat'].values
    posi[4]=posi_info['lng'].values
    posi.reset_index(inplace=True,drop=True)
    posi.drop([6, 5], axis=1,inplace=True)
    
    
    wifi=logfile[logfile[0]=="WIFI"]
    startime=np.min(wifi.iloc[:,2].astype('float64'))
    endtime=np.max(wifi.iloc[:,2].astype('float64'))
    wifi=wifi.iloc[:,0:7]
    wifi=wifi.dropna() #drop NaN ssid name if router name is not in English
    wifi.set_index([0], inplace=True)
    wifi.reset_index(inplace=True,drop=True)
    
    
    record_time=wifi[2].unique() #get unique time records
    
    print('start extract wifi data...')
    wifi_dataset = pd.DataFrame(columns=[n.strip() for n in router_names],index=[m.strip() for m in record_time])
    for i in tqdm( range (len(wifi))  ):
        for time in record_time:
            if wifi.iloc[i,1]==time:
                for ssid in router_names:
                    if wifi.iloc[i,3]==ssid:
                        wifi_dataset.loc[time,ssid]=wifi.iloc[i,5]
                       # print( True == pd.notna(wifi.iloc[i,5])) check if real rss value is matched
                        break
    wifi_dataset.reset_index(inplace=True)
    wifi_dataset=wifi_dataset.rename(columns={'index': "ST"}) #rename first col as sensortime after reset_index
    wifi_dataset.iloc[:,0]=wifi_dataset.iloc[:,0].astype(np.float64) #convert ST from str to float64
    wifi_dataset.sort_values(by='ST', ascending=True, inplace=True)   #sort
    
    integer_st=wifi_dataset.iloc[:,0].astype(int).unique()
    dec1_st=round(wifi_dataset.iloc[:,0],1).unique()
    densewifi=pd.DataFrame(columns=[n.strip() for n in router_names],index= integer_st)
    densewifi.reset_index(inplace=True)
    densewifi.rename(columns={'index': "ST"},inplace=True) #rename first col as sensortime after reset_index
    
    print('start pinching...')
    for i in tqdm(  range(len(wifi_dataset))  ):
        #print('current i: ',i)
        for j in range(len(densewifi)):
            if int(wifi_dataset.iloc[i,0])==densewifi.iloc[j,0]:
                for q in range(1,len(router_names)+1):
                    if pd.notna(wifi_dataset.iloc[i,q]):
                        densewifi.iloc[j,q]=wifi_dataset.iloc[i,q]
                        # break
    densewifi["lat"] = np.nan
    densewifi["lng"] = np.nan
    
    print('matching lables...')
    indexes_matched=[]
    for st in posi[2]:
        index = abs(densewifi['ST'] - st).idxmin()
        indexes_matched.append(index)
        densewifi.iloc[index,-2]=posi[posi[2]==st].iloc[:,3].values
        densewifi.iloc[index,-1]=posi[posi[2]==st].iloc[:,4].values
    
    finaldensewifi=densewifi.iloc[indexes_matched[0]:indexes_matched[-1]+1,:]
    finaldensewifi['lat']=finaldensewifi['lat'].interpolate(method='linear')
    finaldensewifi['lng']=finaldensewifi['lng'].interpolate(method='linear')
    finaldensewifi.reset_index(inplace=True,drop=True)
    finaldensewifi.to_csv('wifi/wifi'+dir+'.csv')
    print('wifi'+dir+'.csv has finished!')
    
dirs = os.listdir(folder)

wifi_filename = "wifi_id.txt"
wr1 = WifiRecord("rawdata/")
write_ap_to_file(wr1, wifi_filename)
for dir in dirs:
    if dir != '.DS_Store':
        print('Processing: ',dir)
        processwifi(folder,dir)
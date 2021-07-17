# encoding: utf-8
import os
import re
import xml.dom.minidom
import collections
import numpy as np
from scipy import interpolate
# import pandas as pd
import pyproj
import h5py
import pandas as pd
import utm
import matplotlib.pyplot as plt
# from collections import Counter

# define some constant variables
north_west = (55.945139, -3.18781)
south_east = (55.944600, -3.186537)
num_grid_y = 30  # latitude
num_grid_x = 40  # longitude
max_lat = abs(north_west[0] - south_east[0])  # 0.0006 # 0.0005393
max_lng = abs(north_west[1] - south_east[1])  # 0.002  # 0.001280
delta_lat = max_lat / num_grid_y  # 3e-06
delta_lng = max_lng / num_grid_x  # 1e-05

# ROW_SIZE = 2000
NUM_COLUMNS = 1*102

# This script processing the original log file and write in 'out_in_accor_wifi(* indicates the number of wifi characters).h5',
# 'out_in_accor_wifi(* indicate the number of wifi characters)'.txt

# *****************************************************************************************************
# 2. Read the distinct access point id from file("wifi_filename") into dictionary

# ** VERSION 1 **: 102 dimensions(Xingji)/135 dimensions(Simon)
wifi_filename = "wifi_id1.txt"


def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict


WIFI_DICT = read_ap_to_dict(wifi_filename)


# Normalise each APs strength to [0,1]
def normalize_wifi_inputs(wr_inputs):

    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    wifi_max = -40
    wifi_min = -100
    wr_inputs = (wr_inputs - wifi_min) / (wifi_max - wifi_min)

    return wr_inputs


# convert the original lat&lng to [-1,1]
def latlng_to_cor(outputs):
    north_west = (55.945139, -3.18781)  # A
    south_east = (55.944600, -3.186537)  # B

    # lat-y
    max0 = north_west[0]
    min0 = south_east[0]
    outputs[:, 0] = 2 * (outputs[:, 0] - min0) / (max0 - min0) - 1

    # lng-x
    max1 = south_east[1]
    min1 = north_west[1]
    outputs[:, 1] = 2 * (outputs[:, 1] - min1) / (max1 - min1) - 1

    # now the outputs[lat, lng], that is [y, x]
    # we would like to reverse the order of the 2 columns, and become [x,y] as follow:
    outputs[:, [0, 1]] = outputs[:, [1, 0]]

    return outputs


def latlng_to_utm(outputs):
    p1 = pyproj.Proj(init="epsg:4326")  # 定义数据地理坐标系 // WGS84，GPS使用的地理坐标系统，EPSG Code为4326
    p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
    x1, y1 = p1(outputs[:, 1], outputs[:, 0])
    x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=False)
    return np.hstack((x2, y2))


# Generate the interpolated location according to the input time list<t_list>
def get_label_list(t_list, loc_dict):

    loc_list = np.zeros((len(loc_dict), 3))
    for k, v in loc_dict.items():
        loc_list[k[0], 0] = k[1]
        loc_list[k[0], 1] = v[0]
        loc_list[k[0], 2] = v[1]

    f_lat = interpolate.interp1d(loc_list[:, 0], loc_list[:, 1], kind='linear', fill_value="extrapolate")
    f_lng = interpolate.interp1d(loc_list[:, 0], loc_list[:, 2], kind='linear', fill_value="extrapolate")
    c123 = []

    for _t in t_list:
        interpolated_lat = f_lat(_t).ravel()[0]
        interpolated_lng = f_lng(_t).ravel()[0]
        t_lat_lng = [_t, interpolated_lat, interpolated_lng]
        c123.append(t_lat_lng)

    c123 = np.array(c123)

    return c123


def plot(f_outputs):
    print("*********  " + str(SensorFile.file_rank))

    plt.scatter(f_outputs[:, 1], f_outputs[:, 0], s=1, marker='x')

    # plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    # plt.show()
    # plt.close()
    print("\n")


class SensorFile(object):
    # Class variable
    world_ap_dict = WIFI_DICT
    file_rank = 0

    def __init__(self, file_path, file_name, i):
        SensorFile.file_rank += 1
        # Member variables
        self.wr_dict = collections.OrderedDict()
        self.loc_dict = collections.OrderedDict()
        self.fn = file_name
        self.index = i

        # Transfer the data from raw file into internal data structure
        self.first_parse_file(file_path)

        # Filter out intermediate wifi according to the start and end location
        self.wr_t_list = self.filter_wifi()
        self.sample_num = len(self.wr_dict)
        self.f_inputs = np.zeros((self.sample_num, NUM_COLUMNS))
        self.f_outputs = np.zeros((self.sample_num, 2))

        # output
        t_lat_lng = get_label_list(self.wr_t_list, self.loc_dict)
        # normalize output
        # t_lat_lng[:, [1, 2]] = t_lat_lng[:, [2, 1]]

        # 1 2 3切换原始经纬度/标准化的local坐标/utm坐标
        self.f_outputs = t_lat_lng[:, 1:]
        # self.f_outputs = latlng_to_cor(t_lat_lng[:, 1:])
        # self.f_outputs = latlng_to_utm(t_lat_lng[:, 1:])

        # input
        wifi_array = np.zeros((self.sample_num, NUM_COLUMNS))
        for v, i in zip(self.wr_dict.values(), range(len(self.wr_dict.items()))):
            wifi_array[i, :] = self.formalize_wr(v)
        # normalize input
        self.f_inputs = normalize_wifi_inputs(wifi_array)

        # Save standard input and output into files
        self.save_overall_txt()

        print("********************")

    def first_parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        loc_list = root.getElementsByTagName('loc')

        print("# wifi record:", wr_list.length)
        print("# loc record:", loc_list.length)
     
        # location(user input)
        for item, i in zip(loc_list, range(len(loc_list))):
            try:
                t = int(item.getAttribute("t"))
                lat = float(item.getAttribute("lat"))
                lng = float(item.getAttribute("lng"))
            except ValueError:
                print('invalid input %d: %s,%s'.format(i, lat, lng))
            self.loc_dict[(i, t)] = (lat, lng)

        # all wifi records in log file
        # for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
        for item in wr_list:  # for each time step
            t = int(item.getAttribute("t"))
            # print(i, "->", t, len(item.childNodes)//2)

            # ap_list是一个ap的列表，一个ap_list表示一个<wr>，代表一个time step记录下来的一个ap的列表
            ap_list = list()
            for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
                if j % 2:
                    ap = item.childNodes[j].getAttribute("b")
                    s = item.childNodes[j].getAttribute("s")
                    if ap not in self.world_ap_dict.keys():
                        pass
                        # self.world_wifi[ap] = 1
                        # print("{} not in world ap dict".format(ap))
                    else:
                        ap_list.append((ap, s))
            # self.wr_dict[(i, t)] = ap_list
            self.wr_dict[t] = ap_list

    def filter_wifi(self):

        tt1 = next(iter(self.loc_dict))[1]  # get the first key
        tt2 = next(reversed(self.loc_dict))[1]  # get the last key

        print(len(self.wr_dict))
        t_list = list()
        _wr_dict = self.wr_dict.copy()
        for k, v in _wr_dict.items():
            if tt1 > k or k > tt2:
                self.wr_dict.pop(k)
            else:
                t_list.append(k)
        print(len(self.wr_dict))
        print(len(t_list))
        return t_list



    # ----------------------------------------------------------------------------------------


    @staticmethod
    def formalize_wr(wr):
        ap_num = len(SensorFile.world_ap_dict)  # standard input need same number of input ap
        element = np.zeros(ap_num)
        for ap in wr:
            ap_id = ap[0]
            ap_val = ap[1]
            # find out the index（column index in element） of this ap_id
            ap_index = int(SensorFile.world_ap_dict[ap_id][1]) - 1
            element[ap_index] = ap_val
        return element

    def save_overall_txt(self):
        write_text = np.hstack((self.f_outputs, self.f_inputs))
        zero=[488278.27,6199937.61]
        ux=utm.from_latlon(write_text[:,0],write_text[:,1])[0]-zero[0]
        uy=utm.from_latlon(write_text[:,0],write_text[:,1])[1]-zero[1]
        write_text[:,0]=ux
        write_text[:,1]=uy
        cs=['lat','lng']
        for i in range(102):
            cs.append('ap'+str(i))
        df=pd.DataFrame(write_text,columns=cs)
        df.to_csv("wifiloc/wifi"+self.fn+".csv",index=None)

    def plot(self):
        SensorFile.file_rank += 1
        print("*********  "+ str(SensorFile.file_rank))
        fig = plt.figure()
        # north_west = (55.945139, -3.18781)  # A
        # south_east = (55.944600, -3.186537)  # B
        plt.axis([-3.18781, -3.186580, 55.944630, 55.945100])

        plt.scatter(self.f_outputs[:, 1], self.f_outputs[:, 0], s=1, marker='x')

        # plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
        # plt.show()
        fig.savefig("./scatter/scatterPlot_"+ str(SensorFile.file_rank) +".png")
        plt.close()
        print("\n")


# Iterate over all the background file in the directory "background"
def iterate(path):
    dirs = os.listdir(path)
    for i,dir in enumerate(dirs):
        fi_d = os.path.join(path, dir)
        SensorFile(fi_d,dir,i)
            # using "continue" here is the same as using "pass"

iterate("rawdata")

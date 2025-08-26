import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import os

import time




def read_csv_files(data_path):
    data = pd.read_csv(data_path)
    points_array = data[['buoy name','buoy X','pred X','buoy Y', 'pred Y', 'SIC']].values
    return points_array

def read_data(data_path):
    data_path_18_h = os.path.join(data_path, '18_h')
    data_path_18_v = os.path.join(data_path, '18_v')
    data_path_36_h = os.path.join(data_path, '36_h')
    data_path_36_v = os.path.join(data_path, '36_v')

    csv_files = [os.path.join(data_path_18_h, file) for file in os.listdir(data_path_18_h) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    points_mag = []
    points_dir = []
    SIC=[]
    for basename in csv_files_basename:
        result_18v=read_csv_files(data_path_18_v+'/'+basename)
        result_18h=read_csv_files(data_path_18_h+'/'+basename)
        result_36v=read_csv_files(data_path_36_v+'/'+basename)
        result_36h=read_csv_files(data_path_36_h+'/'+basename)

        result_18v_dict = {}
        result_18h_dict = {}
        result_36v_dict = {}
        result_36h_dict = {}
        for i in range(len(result_18v)):
            result_18v_dict[str(int(result_18v[i][0]))] = result_18v[i][1:]
        for i in range(len(result_18h)):
            result_18h_dict[str(int(result_18h[i][0]))] = result_18h[i][1:]
        for i in range(len(result_36v)):
            result_36v_dict[str(int(result_36v[i][0]))] = result_36v[i][1:]
        for i in range(len(result_36h)):
            result_36h_dict[str(int(result_36h[i][0]))] = result_36h[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_18v_dict.keys()) & set(result_18h_dict.keys()) & set(result_36v_dict.keys()) & set(result_36h_dict.keys())
        for key in keys:
            if abs(result_18v_dict[key][0]) > 5 or abs(result_18v_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_18v_dict[key][0], result_18v_dict[key][1],result_18h_dict[key][1],result_36v_dict[key][1],result_36h_dict[key][1]]
            point_y= [result_18v_dict[key][2], result_18v_dict[key][3],result_18h_dict[key][3],result_36v_dict[key][3],result_36h_dict[key][3]]
            point_mag= [np.sqrt(point_x[0]**2+point_y[0]**2),
                        np.sqrt(point_x[1]**2+point_y[1]**2),np.sqrt(point_x[2]**2+point_y[2]**2),
                        np.sqrt(point_x[3]**2+point_y[3]**2),np.sqrt(point_x[4]**2+point_y[4]**2)]
            point_dir= [np.arctan2(point_y[0],point_x[0]),
                        np.arctan2(point_y[1],point_x[1]),np.arctan2(point_y[2],point_x[2]),
                        np.arctan2(point_y[3],point_x[3]),np.arctan2(point_y[4],point_x[4])]
            points_x.append(point_x)
            points_y.append(point_y)
            points_mag.append(point_mag)
            points_dir.append(point_dir)
            SIC.append(result_18v_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    points_mag = np.array(points_mag)
    points_dir = np.array(points_dir)
    SIC = np.array(SIC)

    return points_x,points_y,points_mag,points_dir,SIC



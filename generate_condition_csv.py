import itertools
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import time
from sklearn.svm import SVR
from ML_method import RF,Liner,svm,poly,DT,xgboost
from group_truth_pred import read_data_for_2_winter, read_data_for_2_summer, read_data_for_3_summer, read_data_for_3_winter, read_data_for_4_winter,read_data_all_summer,read_data_all_winter,read_data_all_5_winter,read_data_all_5_summer,read_data_all_4_summer,read_data_all_4_winter
from regress_read_file import read_data

def read_csv_files(data_path):
    data = pd.read_csv(data_path)
    points_array = data[['buoy name','buoy X','pred X','buoy Y', 'pred Y','SIC']].values
    return points_array

def r2_analyse(x,y):
    x = x.reshape(-1, 1)  # 浮标 X 坐标
    # 创建线性回归模型
    model_fit = LinearRegression()
    model_fit.fit(x, y)

    # 进行预测
    y_pred = model_fit.predict(x)

    # 计算 R^2 值
    r_squared = r2_score(y, y_pred)

    plt.plot(x, y_pred, color='red', label='Fitted line')

    # plt.plot(x, x, color='red', label='y=x')

    return model_fit.coef_,model_fit.intercept_


#按照日期读取csv文件
data_path_train='./result/train/excel'
data_path_test='./result/test/excel'


bands = ['18v', '18h', '18P', '36v', '36h', '36P']
#========================================6 condition=======================================
train_x,train_y, train_SIC=read_data_all_winter(data_path_train,mode='train')
test_x,test_y, test_SIC=read_data_all_winter(data_path_test,mode='test')
train_x,train_y, train_SIC=read_data_all_summer(data_path_train,mode='train')
test_x,test_y, test_SIC=read_data_all_summer(data_path_test,mode='test')
#========================================5 condition=======================================
freq_combinations = [('18v', '18h', '18P', '36v', '36h'), ('18v', '18h', '18P', '36v', '36P'), ('18v', '18h', '18P', '36h', '36P'), ('18v', '18h', '36v', '36h', '36P'), ('18v', '18P', '36v', '36h', '36P'), ('18h', '18P', '36v', '36h', '36P')]
for freq1, freq2, freq3, freq4, freq5 in freq_combinations:
    train_x, train_y, train_SIC = read_data_all_5_winter(data_path_train, freq1, freq2, freq3, freq4, freq5, mode='train')
    test_x, test_y, test_SIC = read_data_all_5_winter(data_path_test, freq1, freq2, freq3, freq4, freq5, mode='test')
    train_x, train_y, train_SIC = read_data_all_5_summer(data_path_train, freq1, freq2, freq3, freq4, freq5, mode='train')
    test_x, test_y, test_SIC = read_data_all_5_summer(data_path_test, freq1, freq2, freq3, freq4, freq5, mode='test')

#========================================4 condition=======================================
combinations = list(itertools.combinations(bands, 4))
for freq1, freq2, freq3, freq4 in combinations:
    train_x, train_y, train_SIC = read_data_all_4_winter(data_path_train, freq1, freq2, freq3, freq4, mode='train')
    test_x, test_y, test_SIC = read_data_all_4_winter(data_path_test, freq1, freq2, freq3, freq4, mode='test')
    train_x, train_y, train_SIC = read_data_all_4_summer(data_path_train, freq1, freq2, freq3, freq4, mode='train')
    test_x, test_y, test_SIC = read_data_all_4_summer(data_path_test, freq1, freq2, freq3, freq4, mode='test')
#========================================3 condition=======================================
combinations = list(itertools.combinations(bands, 3))
for freq1, freq2, freq3 in combinations:
    train_x, train_y, train_SIC = read_data_for_3_winter(data_path_train, freq1, freq2, freq3, mode='train')
    test_x, test_y, test_SIC = read_data_for_3_winter(data_path_test, freq1, freq2, freq3, mode='test')
    train_x, train_y, train_SIC = read_data_for_3_summer(data_path_train, freq1, freq2, freq3, mode='train')
    test_x, test_y, test_SIC = read_data_for_3_summer(data_path_test, freq1, freq2, freq3, mode='test')
#========================================2 condition=======================================
combinations = list(itertools.combinations(bands, 2))
for freq1, freq2 in combinations:
    train_x, train_y, train_SIC = read_data_for_2_winter(data_path_train, freq1, freq2, mode='train')
    test_x, test_y, test_SIC = read_data_for_2_winter(data_path_test, freq1, freq2, mode='test')
    train_x, train_y, train_SIC = read_data_for_2_summer(data_path_train, freq1, freq2, mode='train')
    test_x, test_y, test_SIC = read_data_for_2_summer(data_path_test, freq1, freq2, mode='test')



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
import joblib

# from regress_read_file import read_data

def read_csv_files_6(data_path):
    data = pd.read_csv(data_path)
    points_array = data[['# Truth',' 18v',' 18h',' 18P',' 36v', ' 36h',' 36P',' sic']].values  	 	 
    return points_array

def read_csv_files_5(data_path,type1, type2, type3, type4, type5):
    data = pd.read_csv(data_path)
    points_array = data[['# Truth',f' {type1}',f' {type2}', f' {type3}', f' {type4}', f' {type5}',' sic']].values	 	 
    return points_array

def read_csv_files_4(data_path,type1, type2, type3, type4):
    data = pd.read_csv(data_path)
    points_array = data[['# Truth',f' {type1}',f' {type2}', f' {type3}', f' {type4}',' sic']].values  	 	 
    return points_array

def read_csv_files_3(data_path,type1, type2, type3):
    data = pd.read_csv(data_path)
    points_array = data[['# Truth',f' {type1}',f' {type2}', f' {type3}',' sic']].values  	 	 
    return points_array

def read_csv_files_2(data_path,type1, type2):
    data = pd.read_csv(data_path)
    points_array = data[['# Truth',f' {type1}', f' {type2}', ' sic']].values  	 	 
    return points_array

def Liner(X_train, X_test, Y_train, Y_test, path):
    # 创建线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(X_train, Y_train)

    #保存模型
    joblib.dump(model, path)

    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def RF(X_train, X_test, Y_train, Y_test, path, tre_num=100):
    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=tre_num, max_depth=7, random_state=5)
    # 训练模型
    model.fit(X_train, Y_train)

    #保存模型
    joblib.dump(model, path)

    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def average(x_test, y_test):
    x_test = x_test.reshape(-1, 1)

    return x_test


condition='2'
bands = ['18v', '18h', '18P', '36v', '36h', '36P']
if condition=='6':
    for season in ['summer','winter']:
        data_path_train_x=f'./result/condition_csv/{season}/{condition}/train_{condition}_x.csv'
        data_path_train_y=f'./result/condition_csv/{season}/{condition}/train_{condition}_y.csv'
        data_path_test_x=f'./result/condition_csv/{season}/{condition}/test_{condition}_x.csv'
        data_path_test_y=f'./result/condition_csv/{season}/{condition}/test_{condition}_y.csv'
        model_save_path=f'./result/condition_csv/{season}/{condition}/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        train_x=read_csv_files_6(data_path_train_x)
        train_y=read_csv_files_6(data_path_train_y)
        test_x=read_csv_files_6(data_path_test_x)
        test_y=read_csv_files_6(data_path_test_y)

        print(f'========================{season}========================')
        print('========================X========================')
        X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_x[:,1:7], test_x[:,1:7], train_x[:,0], test_x[:,0], train_x[-1], test_x[-1]

        r2_18v_x,rmse_18v_x,mae_18v_x=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
        r2_18h_x,rmse_18h_x,mae_18h_x=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
        r2_18P_x,rmse_18P_x,mae_18P_x=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
        r2_36v_x,rmse_36v_x,mae_36v_x=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))
        r2_36h_x,rmse_36h_x,mae_36h_x=r2_score(Y_test, X_test[:,4]),np.sqrt(mean_squared_error(Y_test, X_test[:,4])),np.mean(np.abs(Y_test-X_test[:,4]))
        r2_36P_x,rmse_36P_x,mae_36P_x=r2_score(Y_test, X_test[:,5]),np.sqrt(mean_squared_error(Y_test, X_test[:,5])),np.mean(np.abs(Y_test-X_test[:,5]))

        r2_average_x,rmse_average_x,mae_average_x=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
        r2_liner_x,rmse_liner_x,mae_liner_x=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_liner.pkl')
        start_time = time.time()
        r2_RF1_x,rmse_RF1_x,mae_RF1_x=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_RF.pkl',tre_num=100)
        end_time = time.time()

        print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
        print(' 18V 18H 18p 36V 36H 36P average Liner RF')
        print('R2',r2_18v_x,r2_18h_x,r2_18P_x,r2_36v_x,r2_36h_x,r2_36P_x,r2_average_x, r2_liner_x, r2_RF1_x)
        print('RMSE',rmse_18v_x*12.5,rmse_18h_x*12.5,rmse_18P_x*12.5,rmse_36v_x*12.5,rmse_36h_x*12.5,rmse_36P_x*12.5,rmse_average_x*12.5, rmse_liner_x*12.5, rmse_RF1_x*12.5)



        print('========================Y========================')
        X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_y[:,1:7], test_y[:,1:7], train_y[:,0], test_y[:,0], train_y[-1], test_y[-1]

        r2_18v_y,rmse_18v_y,mae_18v_y=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
        r2_18h_y,rmse_18h_y,mae_18h_y=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
        r2_18P_y,rmse_18P_y,mae_18P_y=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
        r2_36v_y,rmse_36v_y,mae_36v_y=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))
        r2_36h_y,rmse_36h_y,mae_36h_y=r2_score(Y_test, X_test[:,4]),np.sqrt(mean_squared_error(Y_test, X_test[:,4])),np.mean(np.abs(Y_test-X_test[:,4]))
        r2_36P_y,rmse_36P_y,mae_36P_y=r2_score(Y_test, X_test[:,5]),np.sqrt(mean_squared_error(Y_test, X_test[:,5])),np.mean(np.abs(Y_test-X_test[:,5]))

        r2_average_y,rmse_average_y,mae_average_y=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
        r2_liner1_y,rmse_liner1_y,mae_liner1_y=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_liner.pkl')
        start_time = time.time()
        r2_RF1_y,rmse_RF1_y,mae_RF1_y=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_RF.pkl',tre_num=100)
        end_time = time.time()

        print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
        print(' 18V 18H 18p 36V 36H 36P average Liner RF')
        print('R2',r2_18v_y,r2_18h_y,r2_18P_y,r2_36v_y,r2_36h_y,r2_36P_y,r2_average_y, r2_liner1_y, r2_RF1_y)
        print('RMSE',rmse_18v_y*12.5,rmse_18h_y*12.5,rmse_18P_y*12.5,rmse_36v_y*12.5,rmse_36h_y*12.5,rmse_36P_y*12.5,rmse_average_y*12.5, rmse_liner1_y*12.5, rmse_RF1_y*12.5)


elif condition =='5':
    combinations = list(itertools.combinations(bands, 5))
    for season in ['summer','winter']:
        print(f'========================{season}========================')
        for type1, type2, type3, type4, type5 in combinations:
            print(f'======================== {type1} {type2} {type3} {type4} {type5} ========================')
            data_path_train_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}_{type5}/train_{type1}_{type2}_{type3}_{type4}_{type5}_x.csv'
            data_path_train_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}_{type5}/train_{type1}_{type2}_{type3}_{type4}_{type5}_y.csv'
            data_path_test_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}_{type5}/test_{type1}_{type2}_{type3}_{type4}_{type5}_x.csv'
            data_path_test_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}_{type5}/test_{type1}_{type2}_{type3}_{type4}_{type5}_y.csv'
            model_save_path=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}_{type5}/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_x=read_csv_files_5(data_path_train_x, type1, type2, type3, type4, type5)
            train_y=read_csv_files_5(data_path_train_y, type1, type2, type3, type4, type5)
            test_x=read_csv_files_5(data_path_test_x, type1, type2, type3, type4, type5)
            test_y=read_csv_files_5(data_path_test_y, type1, type2, type3, type4, type5)

            print('========================X========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_x[:,1:6], test_x[:,1:6], train_x[:,0], test_x[:,0], train_x[-1], test_x[-1]

            r2_type1_x,rmse_type1_x,mae_type1_x=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_x,rmse_type2_x,mae_type2_x=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_x,rmse_type3_x,mae_type3_x=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
            r2_type4_x,rmse_type4_x,mae_type4_x=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))
            r2_type5_x,rmse_type5_x,mae_type5_x=r2_score(Y_test, X_test[:,4]),np.sqrt(mean_squared_error(Y_test, X_test[:,4])),np.mean(np.abs(Y_test-X_test[:,4]))

            r2_average_x,rmse_average_x,mae_average_x=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner_x,rmse_liner_x,mae_liner_x=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_liner.pkl')
            start_time = time.time()
            r2_RF1_x,rmse_RF1_x,mae_RF1_x=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_{type4}_{type5}_x_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_x,r2_type2_x,r2_type3_x,r2_type4_x,r2_type5_x,r2_average_x, r2_liner_x, r2_RF1_x)
            print('RMSE',rmse_type1_x*12.5,rmse_type2_x*12.5,rmse_type3_x*12.5,rmse_type4_x*12.5,rmse_type5_x*12.5,rmse_average_x*12.5, rmse_liner_x*12.5, rmse_RF1_x*12.5)
            print('MAE',mae_type1_x*12.5,mae_type2_x*12.5,mae_type3_x*12.5,mae_type4_x*12.5,mae_type5_x*12.5,mae_average_x*12.5, mae_liner_x*12.5, mae_RF1_x*12.5)
            

            print('========================Y========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_y[:,1:6], test_y[:,1:6], train_y[:,0], test_y[:,0], train_y[-1], test_y[-1]

            r2_type1_y,rmse_type1_y,mae_type1_y=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_y,rmse_type2_y,mae_type2_y=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_y,rmse_type3_y,mae_type3_y=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
            r2_type4_y,rmse_type4_y,mae_type4_y=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))
            r2_type5_y,rmse_type5_y,mae_type5_y=r2_score(Y_test, X_test[:,4]),np.sqrt(mean_squared_error(Y_test, X_test[:,4])),np.mean(np.abs(Y_test-X_test[:,4]))

            r2_average_y,rmse_average_y,mae_average_y=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner1_y,rmse_liner1_y,mae_liner1_y=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_liner.pkl')
            start_time = time.time()
            r2_RF1_y,rmse_RF1_y,mae_RF1_y=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_{type4}_{type5}_y_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_y,r2_type2_y,r2_type3_y,r2_type4_y,r2_type5_y,r2_average_y, r2_liner1_y, r2_RF1_y)
            print('RMSE',rmse_type1_y*12.5,rmse_type2_y*12.5,rmse_type3_y*12.5,rmse_type4_y*12.5,rmse_type5_y*12.5,rmse_average_y*12.5, rmse_liner1_y*12.5, rmse_RF1_y*12.5)
            print('MAE',mae_type1_y*12.5,mae_type2_y*12.5,mae_type3_y*12.5,mae_type4_y*12.5,mae_type5_y*12.5,mae_average_y*12.5, mae_liner1_y*12.5, mae_RF1_y*12.5)


elif condition =='4':
    combinations = list(itertools.combinations(bands, 4))
    for season in ['summer','winter']:
        print(f'========================{season}========================')
        for type1, type2, type3, type4 in combinations:
            print(f'======================== {type1} {type2} {type3} {type4} ========================')
            data_path_train_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}/train_{type1}_{type2}_{type3}_{type4}_x.csv'
            data_path_train_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}/train_{type1}_{type2}_{type3}_{type4}_y.csv'
            data_path_test_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}/test_{type1}_{type2}_{type3}_{type4}_x.csv'
            data_path_test_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}/test_{type1}_{type2}_{type3}_{type4}_y.csv'
            model_save_path=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}_{type4}/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_x=read_csv_files_4(data_path_train_x, type1, type2, type3, type4)
            train_y=read_csv_files_4(data_path_train_y, type1, type2, type3, type4)
            test_x=read_csv_files_4(data_path_test_x, type1, type2, type3, type4)
            test_y=read_csv_files_4(data_path_test_y, type1, type2, type3, type4)

            print('========================X========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_x[:,1:5], test_x[:,1:5], train_x[:,0], test_x[:,0], train_x[-1], test_x[-1]

            r2_type1_x,rmse_type1_x,mae_type1_x=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_x,rmse_type2_x,mae_type2_x=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_x,rmse_type3_x,mae_type3_x=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
            r2_type4_x,rmse_type4_x,mae_type4_x=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))

            r2_average_x,rmse_average_x,mae_average_x=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner_x,rmse_liner_x,mae_liner_x=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_liner.pkl')
            start_time = time.time()
            r2_RF1_x,rmse_RF1_x,mae_RF1_x=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_{type4}_x_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_x,r2_type2_x,r2_type3_x,r2_type4_x,r2_average_x, r2_liner_x, r2_RF1_x)
            print('RMSE',rmse_type1_x*12.5,rmse_type2_x*12.5,rmse_type3_x*12.5,rmse_type4_x*12.5,rmse_average_x*12.5, rmse_liner_x*12.5, rmse_RF1_x*12.5)
            print('MAE',mae_type1_x*12.5,mae_type2_x*12.5,mae_type3_x*12.5,mae_type4_x*12.5,mae_average_x*12.5, mae_liner_x*12.5, mae_RF1_x*12.5)
            
            print('========================Y========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_y[:,1:5], test_y[:,1:5], train_y[:,0], test_y[:,0], train_y[-1], test_y[-1]

            r2_type1_y,rmse_type1_y,mae_type1_y=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_y,rmse_type2_y,mae_type2_y=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_y,rmse_type3_y,mae_type3_y=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))
            r2_type4_y,rmse_type4_y,mae_type4_y=r2_score(Y_test, X_test[:,3]),np.sqrt(mean_squared_error(Y_test, X_test[:,3])),np.mean(np.abs(Y_test-X_test[:,3]))

            r2_average_y,rmse_average_y,mae_average_y=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner1_y,rmse_liner1_y,mae_liner1_y=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_liner.pkl')
            start_time = time.time()
            r2_RF1_y,rmse_RF1_y,mae_RF1_y=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_{type4}_y_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_y,r2_type2_y,r2_type3_y,r2_type4_y,r2_average_y, r2_liner1_y, r2_RF1_y)
            print('RMSE',rmse_type1_y*12.5,rmse_type2_y*12.5,rmse_type3_y*12.5,rmse_type4_y*12.5,rmse_average_y*12.5, rmse_liner1_y*12.5, rmse_RF1_y*12.5)
            print('MAE',mae_type1_y*12.5,mae_type2_y*12.5,mae_type3_y*12.5,mae_type4_y*12.5,mae_average_y*12.5, mae_liner1_y*12.5, mae_RF1_y*12.5)

elif condition =='3':
    combinations = list(itertools.combinations(bands, 3))
    for season in ['summer','winter']:
        print(f'========================{season}========================')
        for type1, type2, type3 in combinations:
            print(f'======================== {type1} {type2} {type3} ========================')
            data_path_train_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}/train_{type1}_{type2}_{type3}_x.csv'
            data_path_train_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}/train_{type1}_{type2}_{type3}_y.csv'
            data_path_test_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}/test_{type1}_{type2}_{type3}_x.csv'
            data_path_test_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}/test_{type1}_{type2}_{type3}_y.csv'
            model_save_path=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}_{type3}/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_x=read_csv_files_3(data_path_train_x, type1, type2, type3)
            train_y=read_csv_files_3(data_path_train_y, type1, type2, type3)
            test_x=read_csv_files_3(data_path_test_x, type1, type2, type3)
            test_y=read_csv_files_3(data_path_test_y, type1, type2, type3)


            print('========================X========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_x[:,1:4], test_x[:,1:4], train_x[:,0], test_x[:,0], train_x[-1], test_x[-1]

            r2_type1_x,rmse_type1_x,mae_type1_x=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_x,rmse_type2_x,mae_type2_x=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_x,rmse_type3_x,mae_type3_x=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))

            r2_average_x,rmse_average_x,mae_average_x=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner_x,rmse_liner_x,mae_liner_x=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_liner.pkl')
            start_time = time.time()
            r2_RF1_x,rmse_RF1_x,mae_RF1_x=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_x_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('type1 type2 type3 average Liner RF')
            print('R2',r2_type1_x,r2_type2_x,r2_type3_x,r2_average_x, r2_liner_x, r2_RF1_x)
            print('RMSE',rmse_type1_x*12.5,rmse_type2_x*12.5,rmse_type3_x*12.5,rmse_average_x*12.5, rmse_liner_x*12.5, rmse_RF1_x*12.5)
            print('MAE',mae_type1_x*12.5,mae_type2_x*12.5,mae_type3_x*12.5,mae_average_x*12.5, mae_RF1_x*12.5)

            print('========================Y========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_y[:,1:4], test_y[:,1:4], train_y[:,0], test_y[:,0], train_y[-1], test_y[-1]

            r2_type1_y,rmse_type1_y,mae_type1_y=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_y,rmse_type2_y,mae_type2_y=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))
            r2_type3_y,rmse_type3_y,mae_type3_y=r2_score(Y_test, X_test[:,2]),np.sqrt(mean_squared_error(Y_test, X_test[:,2])),np.mean(np.abs(Y_test-X_test[:,2]))

            r2_average_y,rmse_average_y,mae_average_y=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner1_y,rmse_liner1_y,mae_liner1_y=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_liner.pkl')
            r2_RF1_y,rmse_RF1_y,mae_RF1_y=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_{type3}_y_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_y,r2_type2_y,r2_type3_y,r2_average_y, r2_liner1_y, r2_RF1_y)
            print('RMSE',rmse_type1_y*12.5,rmse_type2_y*12.5,rmse_type3_y*12.5,rmse_average_y*12.5, rmse_liner1_y*12.5, rmse_RF1_y*12.5)
            print('MAE',mae_type1_y*12.5,mae_type2_y*12.5,mae_type3_y*12.5,mae_average_y*12.5, mae_liner1_y*12.5, mae_RF1_y*12.5)


elif condition =='2':
    combinations = list(itertools.combinations(bands, 2))
    for season in ['summer','winter']:
        print(f'========================{season}========================')
        for type1, type2 in combinations:
            print(f'======================== {type1} {type2} ========================')
            data_path_train_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}/train_{type1}_{type2}_x.csv'
            data_path_train_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}/train_{type1}_{type2}_y.csv'
            data_path_test_x=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}/test_{type1}_{type2}_x.csv'
            data_path_test_y=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}/test_{type1}_{type2}_y.csv'
            model_save_path=f'./result/condition_csv/{season}/{condition}/{type1}_{type2}/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_x=read_csv_files_2(data_path_train_x, type1, type2)
            train_y=read_csv_files_2(data_path_train_y, type1, type2)
            test_x=read_csv_files_2(data_path_test_x, type1, type2)
            test_y=read_csv_files_2(data_path_test_y, type1, type2)

            print('========================X========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_x[:,1:3], test_x[:,1:3], train_x[:,0], test_x[:,0], train_x[-1], test_x[-1]

            r2_type1_x,rmse_type1_x,mae_type1_x=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_x,rmse_type2_x,mae_type2_x=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))

            r2_average_x,rmse_average_x,mae_average_x=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner_x,rmse_liner_x,mae_liner_x=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_x_liner.pkl')
            start_time = time.time()
            r2_RF1_x,rmse_RF1_x,mae_RF1_x=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_x_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_x,r2_type2_x,r2_average_x, r2_liner_x, r2_RF1_x)
            print('RMSE',rmse_type1_x*12.5,rmse_type2_x*12.5,rmse_average_x*12.5, rmse_liner_x*12.5, rmse_RF1_x*12.5)
            print('MAE',mae_type1_x*12.5,mae_type2_x*12.5,mae_average_x*12.5, mae_RF1_x*12.5)

            print('========================Y========================')
            X_train, X_test, Y_train, Y_test, sic_train, sic_test = train_y[:,1:3], test_y[:,1:3], train_y[:,0], test_y[:,0], train_y[-1], test_y[-1]

            r2_type1_y,rmse_type1_y,mae_type1_y=r2_score(Y_test, X_test[:,0]),np.sqrt(mean_squared_error(Y_test, X_test[:,0])),np.mean(np.abs(Y_test-X_test[:,0]))
            r2_type2_y,rmse_type2_y,mae_type2_y=r2_score(Y_test, X_test[:,1]),np.sqrt(mean_squared_error(Y_test, X_test[:,1])),np.mean(np.abs(Y_test-X_test[:,1]))

            r2_average_y,rmse_average_y,mae_average_y=r2_score(Y_test, np.mean(X_test, axis=1)),np.sqrt(mean_squared_error(Y_test, np.mean(X_test, axis=1))),np.mean(np.abs(Y_test-np.mean(X_test, axis=1)))
            r2_liner1_y,rmse_liner1_y,mae_liner1_y=Liner(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{condition}_y_liner.pkl')
            start_time = time.time()
            r2_RF1_y,rmse_RF1_y,mae_RF1_y=RF(X_train, X_test, Y_train, Y_test, model_save_path+f'model_{type1}_{type2}_y_RF.pkl',tre_num=100)
            end_time = time.time()

            print(f"{condition}: RF time: {end_time - start_time:.2f}秒")
            print('R2',r2_type1_y,r2_type2_y,r2_average_y, r2_liner1_y, r2_RF1_y)
            print('RMSE',rmse_type1_y*12.5,rmse_type2_y*12.5,rmse_average_y*12.5, rmse_liner1_y*12.5, rmse_RF1_y*12.5)
            print('MAE',mae_type1_y*12.5,mae_type2_y*12.5,mae_average_y*12.5, mae_liner1_y*12.5, mae_RF1_y*12.5)




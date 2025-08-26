import csv
import os

import numpy as np

from regress_read_file import read_csv_files


def read_data_for_4_winter(data_path,mode='train'):
    data_path_18_h = os.path.join(data_path, '18_h')
    data_path_18_v = os.path.join(data_path, '18_v')
    data_path_36_h = os.path.join(data_path, '36_h')
    data_path_36_v = os.path.join(data_path, '36_v')

    csv_files = [os.path.join(data_path_18_h, file) for file in os.listdir(data_path_18_h) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    SIC=[]
    for basename in csv_files_basename:
        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['06','07','08','09']:
            continue
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
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_18v_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/4'):
        os.makedirs(f'./result/condition_csv/winter/4')
    np.savetxt(f'./result/condition_csv/winter/4/{mode}_4_x.csv', points_x_csv, delimiter=',', header='Truth, 18v, 18h, 36v, 36h, sic')
    np.savetxt(f'./result/condition_csv/winter/4/{mode}_4_y.csv', points_y_csv, delimiter=',', header='Truth, 18v, 18h, 36v, 36h, sic')

    return points_x,points_y,SIC


def read_data_for_3_summer(data_path,type1,type2,type3,mode='train'): 
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)


    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month not in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)


        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]


        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys())
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3]]
            

            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/summer/3/{type1}_{type2}_{type3}'):
        os.makedirs(f'./result/condition_csv/summer/3/{type1}_{type2}_{type3}')
    np.savetxt(f'./result/condition_csv/summer/3/{type1}_{type2}_{type3}/{mode}_{type1}_{type2}_{type3}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, sic')
    np.savetxt(f'./result/condition_csv/summer/3/{type1}_{type2}_{type3}/{mode}_{type1}_{type2}_{type3}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, sic')
    return points_x,points_y,SIC

def read_data_for_3_winter(data_path,type1,type2,type3,mode='train'): 
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)


    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['05','06','07','08','09']:
            continue

        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)


        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]


        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys())
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3]]
            

            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/3/{type1}_{type2}_{type3}'):
        os.makedirs(f'./result/condition_csv/winter/3/{type1}_{type2}_{type3}')
    np.savetxt(f'./result/condition_csv/winter/3/{type1}_{type2}_{type3}/{mode}_{type1}_{type2}_{type3}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, sic')
    np.savetxt(f'./result/condition_csv/winter/3/{type1}_{type2}_{type3}/{mode}_{type1}_{type2}_{type3}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, sic')
    return points_x,points_y,SIC




def read_data_for_2_winter(data_path,type1,type2,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]


    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)


    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    SIC=[]
    for basename in csv_files_basename:
        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)

        result_type1_dict = {}
        result_type2_dict = {}
        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) 
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)

            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)

    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)

    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/2/{type1}_{type2}'):
        os.makedirs(f'./result/condition_csv/winter/2/{type1}_{type2}')
    np.savetxt(f'./result/condition_csv/winter/2/{type1}_{type2}/{mode}_{type1}_{type2}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, sic')
    np.savetxt(f'./result/condition_csv/winter/2/{type1}_{type2}/{mode}_{type1}_{type2}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, sic')

    return points_x,points_y,SIC

def read_data_for_2_summer(data_path,type1,type2,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]


    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)


    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    SIC=[]
    for basename in csv_files_basename:
        date=basename.split('.')[0]
        month=date[4:6]
        if month not in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)

        result_type1_dict = {}
        result_type2_dict = {}
        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) 
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)

            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)

    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)

    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/summer/2/{type1}_{type2}'):
        os.makedirs(f'./result/condition_csv/summer/2/{type1}_{type2}')
    np.savetxt(f'./result/condition_csv/summer/2/{type1}_{type2}/{mode}_{type1}_{type2}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, sic')
    np.savetxt(f'./result/condition_csv/summer/2/{type1}_{type2}/{mode}_{type1}_{type2}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, sic')

    return points_x,points_y,SIC



def read_data_all_summer(data_path,mode='train'):
    data_path_18_h = os.path.join(data_path, '18_h')
    data_path_18_v = os.path.join(data_path, '18_v')
    data_path_18_P = os.path.join(data_path, '18_P')
    data_path_36_h = os.path.join(data_path, '36_h')
    data_path_36_v = os.path.join(data_path, '36_v')
    data_path_36_P = os.path.join(data_path, '36_P')

    csv_files = [os.path.join(data_path_18_h, file) for file in os.listdir(data_path_18_h) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    SIC=[]
    for basename in csv_files_basename:
        date=basename.split('.')[0]
        month=date[4:6]
        if month not in ['05','06','07','08','09']:
            continue
        result_18v=read_csv_files(data_path_18_v+'/'+basename)
        result_18h=read_csv_files(data_path_18_h+'/'+basename)
        result_18P=read_csv_files(data_path_18_P+'/'+basename)
        result_36v=read_csv_files(data_path_36_v+'/'+basename)
        result_36h=read_csv_files(data_path_36_h+'/'+basename)
        result_36P=read_csv_files(data_path_36_P+'/'+basename)

        result_18v_dict = {}
        result_18h_dict = {}
        result_18P_dict = {}
        result_36v_dict = {}
        result_36h_dict = {}
        result_36P_dict = {}
        for i in range(len(result_18v)):
            result_18v_dict[str(int(result_18v[i][0]))] = result_18v[i][1:]
        for i in range(len(result_18h)):
            result_18h_dict[str(int(result_18h[i][0]))] = result_18h[i][1:]
        for i in range(len(result_18P)):
            result_18P_dict[str(int(result_18P[i][0]))] = result_18P[i][1:]
        for i in range(len(result_36v)):
            result_36v_dict[str(int(result_36v[i][0]))] = result_36v[i][1:]
        for i in range(len(result_36h)):
            result_36h_dict[str(int(result_36h[i][0]))] = result_36h[i][1:]
        for i in range(len(result_36P)):
            result_36P_dict[str(int(result_36P[i][0]))] = result_36P[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_18v_dict.keys()) & set(result_18h_dict.keys()) & set(result_18P_dict.keys()) & set(result_36v_dict.keys()) & set(result_36h_dict.keys()) & set(result_36P_dict.keys())
        for key in keys:
            if abs(result_18v_dict[key][0]) > 5 or abs(result_18v_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_18v_dict[key][0], result_18v_dict[key][1],result_18h_dict[key][1],result_18P_dict[key][1],result_36v_dict[key][1],result_36h_dict[key][1],result_36P_dict[key][1]]
            point_y= [result_18v_dict[key][2], result_18v_dict[key][3],result_18h_dict[key][3],result_18P_dict[key][3],result_36v_dict[key][3],result_36h_dict[key][3],result_36P_dict[key][3]]

            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_18v_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/summer/6'):
        os.makedirs(f'./result/condition_csv/summer/6')
    np.savetxt(f'./result/condition_csv/summer/6/{mode}_6_x.csv', points_x_csv, delimiter=',', header='Truth, 18v, 18h, 18P, 36v, 36h, 36P, sic')
    np.savetxt(f'./result/condition_csv/summer/6/{mode}_6_y.csv', points_y_csv, delimiter=',', header='Truth, 18v, 18h, 18P, 36v, 36h, 36P, sic')

    return points_x,points_y,SIC

def read_data_all_winter(data_path,mode='train'):
    data_path_18_h = os.path.join(data_path, '18_h')
    data_path_18_v = os.path.join(data_path, '18_v')
    data_path_18_P = os.path.join(data_path, '18_P')
    data_path_36_h = os.path.join(data_path, '36_h')
    data_path_36_v = os.path.join(data_path, '36_v')
    data_path_36_P = os.path.join(data_path, '36_P')

    csv_files = [os.path.join(data_path_18_h, file) for file in os.listdir(data_path_18_h) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []
    SIC=[]
    for basename in csv_files_basename:
        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['05','06','07','08','09']:
            continue
        result_18v=read_csv_files(data_path_18_v+'/'+basename)
        result_18h=read_csv_files(data_path_18_h+'/'+basename)
        result_18P=read_csv_files(data_path_18_P+'/'+basename)
        result_36v=read_csv_files(data_path_36_v+'/'+basename)
        result_36h=read_csv_files(data_path_36_h+'/'+basename)
        result_36P=read_csv_files(data_path_36_P+'/'+basename)

        result_18v_dict = {}
        result_18h_dict = {}
        result_18P_dict = {}
        result_36v_dict = {}
        result_36h_dict = {}
        result_36P_dict = {}
        for i in range(len(result_18v)):
            result_18v_dict[str(int(result_18v[i][0]))] = result_18v[i][1:]
        for i in range(len(result_18h)):
            result_18h_dict[str(int(result_18h[i][0]))] = result_18h[i][1:]
        for i in range(len(result_18P)):
            result_18P_dict[str(int(result_18P[i][0]))] = result_18P[i][1:]
        for i in range(len(result_36v)):
            result_36v_dict[str(int(result_36v[i][0]))] = result_36v[i][1:]
        for i in range(len(result_36h)):
            result_36h_dict[str(int(result_36h[i][0]))] = result_36h[i][1:]
        for i in range(len(result_36P)):
            result_36P_dict[str(int(result_36P[i][0]))] = result_36P[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_18v_dict.keys()) & set(result_18h_dict.keys()) & set(result_18P_dict.keys()) & set(result_36v_dict.keys()) & set(result_36h_dict.keys()) & set(result_36P_dict.keys())
        for key in keys:
            if abs(result_18v_dict[key][0]) > 5 or abs(result_18v_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_18v_dict[key][0], result_18v_dict[key][1],result_18h_dict[key][1],result_18P_dict[key][1],result_36v_dict[key][1],result_36h_dict[key][1],result_36P_dict[key][1]]
            point_y= [result_18v_dict[key][2], result_18v_dict[key][3],result_18h_dict[key][3],result_18P_dict[key][3],result_36v_dict[key][3],result_36h_dict[key][3],result_36P_dict[key][3]]

            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_18v_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/6'):
        os.makedirs(f'./result/condition_csv/winter/6')
    np.savetxt(f'./result/condition_csv/winter/6/{mode}_6_x.csv', points_x_csv, delimiter=',', header='Truth, 18v, 18h, 18P, 36v, 36h, 36P, sic')
    np.savetxt(f'./result/condition_csv/winter/6/{mode}_6_y.csv', points_y_csv, delimiter=',', header='Truth, 18v, 18h, 18P, 36v, 36h, 36P, sic')

    return points_x,points_y,SIC


def read_data_all_5_winter(data_path,type1,type2,type3,type4,type5,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]
    type4_read=type4[:2]+'_'+type4[2:]
    type5_read=type5[:2]+'_'+type5[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)
    data_path_4 = os.path.join(data_path, type4_read)
    data_path_5 = os.path.join(data_path, type5_read)

    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)
        result_type4=read_csv_files(data_path_4+'/'+basename)
        result_type5=read_csv_files(data_path_5+'/'+basename)


        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        result_type4_dict = {}
        result_type5_dict = {}

        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]
        for i in range(len(result_type4)):
            result_type4_dict[str(int(result_type4[i][0]))] = result_type4[i][1:]
        for i in range(len(result_type5)):
            result_type5_dict[str(int(result_type5[i][0]))] = result_type5[i][1:]


        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys()) & set(result_type4_dict.keys()) & set(result_type5_dict.keys())
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1],result_type4_dict[key][1],result_type5_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3],result_type4_dict[key][3],result_type5_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/5/{type1}_{type2}_{type3}_{type4}_{type5}'):
        os.makedirs(f'./result/condition_csv/winter/5/{type1}_{type2}_{type3}_{type4}_{type5}')
    np.savetxt(f'./result/condition_csv/winter/5/{type1}_{type2}_{type3}_{type4}_{type5}/{mode}_{type1}_{type2}_{type3}_{type4}_{type5}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, {type5}, sic')
    np.savetxt(f'./result/condition_csv/winter/5/{type1}_{type2}_{type3}_{type4}_{type5}/{mode}_{type1}_{type2}_{type3}_{type4}_{type5}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, {type5}, sic')
    return points_x,points_y,SIC

def read_data_all_5_summer(data_path,type1,type2,type3,type4,type5,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]
    type4_read=type4[:2]+'_'+type4[2:]
    type5_read=type5[:2]+'_'+type5[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)
    data_path_4 = os.path.join(data_path, type4_read)
    data_path_5 = os.path.join(data_path, type5_read)

    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month not in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)
        result_type4=read_csv_files(data_path_4+'/'+basename)
        result_type5=read_csv_files(data_path_5+'/'+basename)


        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        result_type4_dict = {}
        result_type5_dict = {}

        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]
        for i in range(len(result_type4)):
            result_type4_dict[str(int(result_type4[i][0]))] = result_type4[i][1:]
        for i in range(len(result_type5)):
            result_type5_dict[str(int(result_type5[i][0]))] = result_type5[i][1:]


        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys()) & set(result_type4_dict.keys()) & set(result_type5_dict.keys())
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1],result_type4_dict[key][1],result_type5_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3],result_type4_dict[key][3],result_type5_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/summer/5/{type1}_{type2}_{type3}_{type4}_{type5}'):
        os.makedirs(f'./result/condition_csv/summer/5/{type1}_{type2}_{type3}_{type4}_{type5}')
    np.savetxt(f'./result/condition_csv/summer/5/{type1}_{type2}_{type3}_{type4}_{type5}/{mode}_{type1}_{type2}_{type3}_{type4}_{type5}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, {type5}, sic')
    np.savetxt(f'./result/condition_csv/summer/5/{type1}_{type2}_{type3}_{type4}_{type5}/{mode}_{type1}_{type2}_{type3}_{type4}_{type5}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, {type5}, sic')
    return points_x,points_y,SIC

def read_data_all_4_winter(data_path,type1,type2,type3,type4,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]
    type4_read=type4[:2]+'_'+type4[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)
    data_path_4 = os.path.join(data_path, type4_read)

    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)
        result_type4=read_csv_files(data_path_4+'/'+basename)


        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        result_type4_dict = {}

        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]
        for i in range(len(result_type4)):
            result_type4_dict[str(int(result_type4[i][0]))] = result_type4[i][1:]

        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys()) & set(result_type4_dict.keys())
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1],result_type4_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3],result_type4_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/winter/4/{type1}_{type2}_{type3}_{type4}'):
        os.makedirs(f'./result/condition_csv/winter/4/{type1}_{type2}_{type3}_{type4}')
    np.savetxt(f'./result/condition_csv/winter/4/{type1}_{type2}_{type3}_{type4}/{mode}_{type1}_{type2}_{type3}_{type4}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, sic')
    np.savetxt(f'./result/condition_csv/winter/4/{type1}_{type2}_{type3}_{type4}/{mode}_{type1}_{type2}_{type3}_{type4}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, sic')
    return points_x,points_y,SIC

def read_data_all_4_summer(data_path,type1,type2,type3,type4,mode='train'):
    type1_read=type1[:2]+'_'+type1[2:]
    type2_read=type2[:2]+'_'+type2[2:]
    type3_read=type3[:2]+'_'+type3[2:]
    type4_read=type4[:2]+'_'+type4[2:]

    data_path_1 = os.path.join(data_path, type1_read)
    data_path_2 = os.path.join(data_path, type2_read)
    data_path_3 = os.path.join(data_path, type3_read)
    data_path_4 = os.path.join(data_path, type4_read)

    csv_files = [os.path.join(data_path_1, file) for file in os.listdir(data_path_1) if file.endswith('.csv')]
    #转换成basename
    csv_files_basename = [os.path.basename(file) for file in csv_files]

    points_x = []
    points_y = []

    SIC=[]
    for basename in csv_files_basename:

        date=basename.split('.')[0]
        month=date[4:6]
        if month not in ['05','06','07','08','09']:
            continue
        result_type1=read_csv_files(data_path_1+'/'+basename)
        result_type2=read_csv_files(data_path_2+'/'+basename)
        result_type3=read_csv_files(data_path_3+'/'+basename)
        result_type4=read_csv_files(data_path_4+'/'+basename)

        result_type1_dict = {}
        result_type2_dict = {}
        result_type3_dict = {}
        result_type4_dict = {}

        for i in range(len(result_type1)):
            result_type1_dict[str(int(result_type1[i][0]))] = result_type1[i][1:]
        for i in range(len(result_type2)):
            result_type2_dict[str(int(result_type2[i][0]))] = result_type2[i][1:]
        for i in range(len(result_type3)):
            result_type3_dict[str(int(result_type3[i][0]))] = result_type3[i][1:]
        for i in range(len(result_type4)):
            result_type4_dict[str(int(result_type4[i][0]))] = result_type4[i][1:]


        #寻找四个字典中都有的键
        keys = set(result_type1_dict.keys()) & set(result_type2_dict.keys()) & set(result_type3_dict.keys()) & set(result_type4_dict.keys()) 
        for key in keys:
            if abs(result_type1_dict[key][0]) > 5 or abs(result_type1_dict[key][2]) > 5:
                continue
            # truth 18v 18h 36v 36h
            point_x= [result_type1_dict[key][0], result_type1_dict[key][1],result_type2_dict[key][1],result_type3_dict[key][1],result_type4_dict[key][1]]
            point_y= [result_type1_dict[key][2], result_type1_dict[key][3],result_type2_dict[key][3],result_type3_dict[key][3],result_type4_dict[key][3]]
            
            points_x.append(point_x)
            points_y.append(point_y)
            SIC.append(result_type1_dict[key][4])

        print('finish '+basename)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    SIC = np.array(SIC)
    SIC = SIC.reshape((-1, 1))

    points_x_csv = np.concatenate((points_x, SIC), axis=1)
    points_y_csv = np.concatenate((points_y, SIC), axis=1)
    #points_x输出成csv文件,第一行为标题行'Truth', '18v', '18h', '36v', '36h'，sic
    if not os.path.exists(f'./result/condition_csv/summer/4/{type1}_{type2}_{type3}_{type4}'):
        os.makedirs(f'./result/condition_csv/summer/4/{type1}_{type2}_{type3}_{type4}')
    np.savetxt(f'./result/condition_csv/summer/4/{type1}_{type2}_{type3}_{type4}/{mode}_{type1}_{type2}_{type3}_{type4}_x.csv', points_x_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, sic')
    np.savetxt(f'./result/condition_csv/summer/4/{type1}_{type2}_{type3}_{type4}/{mode}_{type1}_{type2}_{type3}_{type4}_y.csv', points_y_csv, delimiter=',', header=f'Truth, {type1}, {type2}, {type3}, {type4}, sic')
    return points_x,points_y,SIC
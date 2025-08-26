import os
import cv2
import numpy as np
from buoy_process import calculate_SIC, search_txt_files, read_buoy_data
from datetime import datetime, timedelta
from img_process import find_img_and_read, find_img_and_read_89, img16to8, Histogram_equal, Histogram_equal_8,find_SIC_and_read,LoG_32F,find_SIC_and_read_89
from CMCC_block import CMCC, CMCC_modify

import csv
from tqdm import tqdm
import time

from post_process import filter_outlier

def day_of_year_to_date(year, day_of_year):
    # 创建指定年份的第一天的日期对象
    date_start = datetime(year, 1, 1)
    # 通过日序数减1来计算日期（因为年积日从1开始，所以要减去1）
    result_date = date_start + timedelta(days=day_of_year - 1)
    #再获得下一天的日期
    result_date_2 = result_date + timedelta(days=1)
    return result_date.strftime('%Y%m%d'), result_date_2.strftime('%Y%m%d') # 格式化日期为 YYYY-MM-DD

def find_point(x,y):
    x_floor = np.floor(x)
    x_ceil = np.ceil(x)
    y_floor = np.floor(y)
    y_ceil = np.ceil(y)
    if x_floor == x_ceil:
        x_ceil += 1
    if y_floor == y_ceil:
        y_ceil += 1
    return int(x_floor), int(x_ceil), int(y_floor), int(y_ceil)

directory = './buoy_data/train'  # test or train

pol='h' # h v P
fre='36' # 18 36

if pol=='P':
    img_base_path_h = './bremen/bremen_{}/bremen_tif/h'.format(fre)
    img_base_path_v = './bremen/bremen_{}/bremen_tif/v'.format(fre)
else:  
    img_base_path = './bremen/bremen_{}/bremen_tif/{}'.format(fre,pol)
    
SIC_base_path = './sic'

found_txt_files = search_txt_files(directory)
exist=1
#判断csv文件是是否存在 D:\study\02_MCC\result
# if not os.path.exists(f'D:/study/02_MCC/result/train/result_{fre}_{pol}.csv'):
#     with open(f'D:/study/02_MCC/result/train/result_{fre}_{pol}.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         # 写入标题行
#         writer.writerow(['Filename', 'Total Count', 'MAE X', 'RMSE X', 'MAE Y', 'RMSE Y', 'MAE Magnitude', 'RMSE Magnitude', 'MAE Direction', 'RMSE Direction','Time'])
#     exist=0
    
# 打印找到的所有 txt 文件路径
for i, txt_file in enumerate(found_txt_files):
    start=time.time()
    filename = os.path.basename(txt_file)
    year = os.path.splitext(filename)[0].split('_')[1]
    day_of_year = os.path.splitext(filename)[0].split('_')[2]
    date1, date2 = day_of_year_to_date(int(year), int(day_of_year))
    
    if pol=='P':
        folder_path_h=img_base_path_h+'/'+year+'_h'
        folder_path_v=img_base_path_v+'/'+year+'_v'
        SIC_folder_path=SIC_base_path+'/'+year
        if fre == '89':
            filename1_h='amsr2-n6250-'+date1+'-'+fre+'h.tif'
            filename1_v='amsr2-n6250-'+date1+'-'+fre+'v.tif'
            filename2_h='amsr2-n6250-'+date2+'-'+fre+'h.tif'
            filename2_v='amsr2-n6250-'+date2+'-'+fre+'v.tif'
        else:
            filename1_h='amsr2-n12500-'+date1+'-'+fre+'h.tif'
            filename1_v='amsr2-n12500-'+date1+'-'+fre+'v.tif'
            filename2_h='amsr2-n12500-'+date2+'-'+fre+'h.tif'
            filename2_v='amsr2-n12500-'+date2+'-'+fre+'v.tif'
        filename_SIC='asi-AMSR2-n6250-'+date1+'-v5.4.tif'
        if date1[4:]=='1231':
            folder_path_for2_h=img_base_path_h+'/'+str(int(year)+1)+'_h'
            folder_path_for2_v=img_base_path_v+'/'+str(int(year)+1)+'_v'
            if fre == '89':
                img1_h = find_img_and_read_89(filename1_h,folder_path_h)
                img1_v = find_img_and_read_89(filename1_v,folder_path_v)
                img2_h = find_img_and_read_89(filename2_h,folder_path_for2_h)
                img2_v = find_img_and_read_89(filename2_v,folder_path_for2_v)
                img1= img1_v - img1_h
                img2= img2_v - img2_h
                SIC, mask = find_SIC_and_read_89(filename_SIC,SIC_folder_path)
            else:
                img1_h = find_img_and_read(filename1_h,folder_path_h)
                img1_v = find_img_and_read(filename1_v,folder_path_v)
                img2_h = find_img_and_read(filename2_h,folder_path_for2_h)
                img2_v = find_img_and_read(filename2_v,folder_path_for2_v)
                img1= img1_v - img1_h
                img2= img2_v - img2_h
                SIC, mask = find_SIC_and_read(filename_SIC,SIC_folder_path)
        else:
            #读取图像数据
            if fre == '89':
                img1_h = find_img_and_read_89(filename1_h,folder_path_h)
                img1_v = find_img_and_read_89(filename1_v,folder_path_v)
                img2_h = find_img_and_read_89(filename2_h,folder_path_h)
                img2_v = find_img_and_read_89(filename2_v,folder_path_v)
                img1= img1_v - img1_h
                img2= img2_v - img2_h
                SIC, mask = find_SIC_and_read_89(filename_SIC,SIC_folder_path)
            else:
                img1_h = find_img_and_read(filename1_h,folder_path_h)
                img1_v = find_img_and_read(filename1_v,folder_path_v)
                img2_h = find_img_and_read(filename2_h,folder_path_h)
                img2_v = find_img_and_read(filename2_v,folder_path_v)
                img1= img1_v - img1_h
                img2= img2_v - img2_h
                SIC, mask = find_SIC_and_read(filename_SIC,SIC_folder_path)
    else:
        folder_path=img_base_path+'/'+year+'_'+pol
        SIC_folder_path=SIC_base_path+'/'+year
        if fre == '89':
            filename1='amsr2-n6250-'+date1+'-'+fre+pol+'.tif'
            filename2='amsr2-n6250-'+date2+'-'+fre+pol+'.tif'
        else:
            filename1='amsr2-n12500-'+date1+'-'+fre+pol+'.tif'
            filename2='amsr2-n12500-'+date2+'-'+fre+pol+'.tif'
        filename_SIC='asi-AMSR2-n6250-'+date1+'-v5.4.tif'
        if date1[4:]=='1231':
            folder_path_for2=img_base_path+'/'+str(int(year)+1)+'_'+pol
            if fre == '89':
                img1 = find_img_and_read_89(filename1,folder_path)
                img2 = find_img_and_read_89(filename2,folder_path_for2)
                SIC, mask = find_SIC_and_read_89(filename_SIC,SIC_folder_path)
            else:
                img1 = find_img_and_read(filename1,folder_path)
                img2 = find_img_and_read(filename2,folder_path_for2)
                SIC, mask = find_SIC_and_read(filename_SIC,SIC_folder_path)
        else:
            #读取图像数据
            if fre == '89':
                img1 = find_img_and_read_89(filename1,folder_path)
                img2 = find_img_and_read_89(filename2,folder_path)
                SIC, mask = find_SIC_and_read_89(filename_SIC,SIC_folder_path)
            else:
                img1 = find_img_and_read(filename1,folder_path)
                img2 = find_img_and_read(filename2,folder_path)
                SIC, mask = find_SIC_and_read(filename_SIC,SIC_folder_path)
    
    #将img1保存成三通道8比他的图像，作为后续的背景影像
    
    if pol=='P':
        img1_8 = img16to8(img1_v)
        img1_background =np.stack((img1_8,img1_8,img1_8),axis=2)
    else:
        img1_8 = img16to8(img1)
        img1_background =np.stack((img1_8,img1_8,img1_8),axis=2)


    #处理图像数据
    img1= Histogram_equal(img1,mask)
    img2= Histogram_equal(img2,mask)
    LOG_img1 = LoG_32F(img1,3)
    LOG_img2 = LoG_32F(img2,3)
    
    img1_CMCC = LOG_img1
    img2_CMCC = LOG_img2

    drift_map = CMCC_modify(img1_CMCC,img2_CMCC,SIC,date1,template_size=11)

    #post filter
    drift_map = filter_outlier(drift_map)

    #将drift_map保存为npy文件 D:/study/02_MCC/MCC实验
    if not os.path.exists(f'./result/train/drift_map_{fre}_{pol}'):
        os.makedirs(f'./result/train/drift_map_{fre}_{pol}')
    np.save(f'./result/train/drift_map_{fre}_{pol}/{date1}.npy', drift_map)
    
    #处理浮标数据
    buoy_data = read_buoy_data(txt_file)
    error_x, error_y, error_magnitude, error_direction= [], [], [], []
    vaild_count = 0
    if not os.path.exists(f'./result/train/excel/{fre}_{pol}'):
        os.makedirs(f'./result/train/excel/{fre}_{pol}')
    
    with open(f'./result/train/excel/{fre}_{pol}/{date1}.csv', 'a', newline='') as file_sub:
        writer_sub = csv.writer(file_sub)
        writer_sub.writerow(['buoy name', 'position x', 'position y', 'buoy X', 'pred X', 'buoy Y', 'pred Y', 'SIC'])

    with open(f'./result/train/excel/{fre}_{pol}/{date1}.csv', 'a', newline='') as file_sub:
        for key, value in buoy_data.items():
            
            x, y, buoy_dx, buoy_dy = value
            x_nearest = round(x)
            y_nearest = round(y)
            if x_nearest < 0 or x_nearest >= drift_map.shape[0] or y_nearest < 0 or y_nearest >= drift_map.shape[1]:
                continue
            estimate_x = drift_map[x_nearest][y_nearest][0]
            estimate_y = drift_map[x_nearest][y_nearest][1]
            sic_point = calculate_SIC(x_nearest,y_nearest,SIC) 

            if np.isnan(estimate_x) or np.isnan(estimate_y):
                flag=0
                x_floor, x_ceil, y_floor, y_ceil=find_point(x,y)
                if x_floor < 0 or x_ceil < 0 or y_floor < 0 or y_ceil < 0:
                    continue
                if x_floor >= drift_map.shape[0] or x_ceil >= drift_map.shape[0] or y_floor >= drift_map.shape[1] or y_ceil >= drift_map.shape[1]:
                    continue
                
                if ~np.isnan(drift_map[x_floor][y_floor][0]) and ~np.isnan(drift_map[x_floor][y_floor][1]):
                    estimate_x = drift_map[x_floor][y_floor][0]
                    estimate_y = drift_map[x_floor][y_floor][1]
                    sic_point = calculate_SIC(x_nearest,y_nearest,SIC)
                    flag=1
                if ~np.isnan(drift_map[x_floor][y_ceil][0]) and ~np.isnan(drift_map[x_floor][y_ceil][1]):
                    estimate_x = drift_map[x_floor][y_ceil][0]
                    estimate_y = drift_map[x_floor][y_ceil][1]
                    sic_point = calculate_SIC(x_nearest,y_nearest,SIC)   
                    flag=1
                if ~np.isnan(drift_map[x_ceil][y_floor][0]) or ~np.isnan(drift_map[x_ceil][y_floor][1]):
                    estimate_x = drift_map[x_ceil][y_floor][0]
                    estimate_y = drift_map[x_ceil][y_floor][1]
                    sic_point = calculate_SIC(x_nearest,y_nearest,SIC)
                    flag=1
                if ~np.isnan(drift_map[x_ceil][y_ceil][0]) or ~np.isnan(drift_map[x_ceil][y_ceil][1]):
                    estimate_x = drift_map[x_ceil][y_ceil][0]
                    estimate_y = drift_map[x_ceil][y_ceil][1]
                    sic_point = calculate_SIC(x_nearest,y_nearest,SIC)
                    flag=1
                
                if flag==0:
                    continue
            
            buoy_magnitude = np.sqrt(buoy_dx ** 2 + buoy_dy ** 2)
            buoy_direction = np.arctan2(buoy_dy, buoy_dx) * 180 / np.pi
            estimate_magnitude = np.sqrt(estimate_x ** 2 + estimate_y ** 2)
            estimate_direction = np.arctan2(estimate_y, estimate_x) * 180 / np.pi

            error_x.append(estimate_x - buoy_dx)
            error_y.append(estimate_y - buoy_dy)
            error_magnitude.append(estimate_magnitude - buoy_magnitude)
            error_direction.append(estimate_direction - buoy_direction)
            vaild_count += 1

            # for i in range(-1,2):
            #     for j in range(-1,2):
            #         if x_nearest+i < 0 or x_nearest+i >= drift_map.shape[0] or y_nearest+j < 0 or y_nearest+j >= drift_map.shape[1]:
            #             continue
            #         if ~np.isnan(drift_map[x_nearest+i][y_nearest+j][0]) and ~np.isnan(drift_map[x_nearest+i][y_nearest+j][1]):
            #             cv2.arrowedLine(img1_background, (y_nearest+j, x_nearest+i), (y_nearest+j+int(10 * drift_map[x_nearest+i][y_nearest+j][1]), x_nearest+i+int(10 * drift_map[x_nearest+i][y_nearest+j][0])), (255, 0, 0), 1,tipLength=0.5)
            # #画真值箭头
            # cv2.arrowedLine(img1_background, (y_nearest, x_nearest), (y_nearest+int(10 * buoy_dy), x_nearest+int(10 * buoy_dx)), (0, 0, 255), 1,tipLength=0.5)

            writer_sub = csv.writer(file_sub)
            # 写入标题行
            writer_sub.writerow([key, x, y, buoy_dx, estimate_x, buoy_dy, estimate_y, sic_point])
    
    #计算总验证次数
    total_count = vaild_count

    mae_x = np.mean(np.abs(error_x))
    rmse_x = np.sqrt(np.mean(np.square(error_x)))
    
    mae_y = np.mean(np.abs(error_y))
    rmse_y = np.sqrt(np.mean(np.square(error_y)))

    mae_magnitude = np.mean(np.abs(error_magnitude))
    rmse_magnitude = np.sqrt(np.mean(np.square(error_magnitude)))

    mae_direction = np.mean(np.abs(error_direction))
    rmse_direction = np.sqrt(np.mean(np.square(error_direction)))

    end=time.time()
    
    with open(f'./result/train/result_{fre}_{pol}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow([
                filename, total_count, 
                f"{mae_x:.4f}", f"{rmse_x:.4f}", 
                f"{mae_y:.4f}", f"{rmse_y:.4f}", 
                f"{mae_magnitude:.4f}", f"{rmse_magnitude:.4f}", 
                f"{mae_direction:.4f}", f"{rmse_direction:.4f}",
                f"{end-start:.4f}"
            ])
        print(f'{filename} done!')


    # if not os.path.exists(f'./result/train/png/{fre}_{pol}'):
    #     os.makedirs(f'./result/train/png/{fre}_{pol}')
    # cv2.imwrite(f'./result/train/png/{fre}_{pol}/{year}_{day_of_year}.png',img1_background)


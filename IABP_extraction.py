import pandas as pd
import glob
import os

dat_files = glob.glob('./IABP/L1/*.dat')
#sort the file list base on number
dat_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
for file_path in dat_files:

    file_name = os.path.basename(file_path)
    file_name = file_name.split('.')[0]

    parsed_data = []
    with open(file_path, 'r') as file:
        column_names = file.readline().strip().split()
        
        for line in file:
            split_line = line.strip().split()
            parsed_data.append(split_line)

    data = pd.DataFrame(parsed_data, columns=column_names)
    
    # 创建DateTime列，合并年、DOY、小时和分钟
    data['DOY_int'] = data['DOY'].apply(lambda x: int(float(x)))  
    try:
        data['DateTime'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['DOY_int'].astype(str), format='%Y-%j') \
                        + pd.to_timedelta(data['Hour'].astype(int), unit='h') + pd.to_timedelta(data['Min'].astype(int), unit='m')
    except ValueError as e:
        print('Error in file:', file_name)
        #有问题的文件名写入txt文件
        with open("error_file.txt", "a") as f:
            f.write(file_name + "\n")
        continue
    data = data[(data['DateTime'].dt.year >= 2013) & (data['DateTime'].dt.year <= 2020)] #train
    # data = data[(data['DateTime'].dt.year >= 2021) & (data['DateTime'].dt.year <= 2023)] #test
    if data.empty:
        continue
    # 提取浮标编号、时间和位置信息
    selected_data = data[['BuoyID', 'DateTime', 'Lat', 'Lon']]

    # 计算数据的起始和终止日期
    start_date = selected_data['DateTime'].min()
    end_date = selected_data['DateTime'].max()

    # 生成每一天的日期范围
    date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')

    # 初始化空DataFrame存储每天最接近12点的数据
    optimized_nearest_to_noon = pd.DataFrame()

    for single_date in date_range:
        # 计算当天12点的具体时间
        noon_time = pd.Timestamp(single_date.year, single_date.month, single_date.day, 12, 0, 0)
        # 当天的数据
        day_data = selected_data[(selected_data['DateTime'] >= single_date) & (selected_data['DateTime'] < single_date + pd.Timedelta(days=1))]
        # 如果当天有数据
        if not day_data.empty:
            # 计算每个数据点与当天12点的时间差
            day_data['TimeDiff'] = abs(day_data['DateTime'] - noon_time)
            # 找到最小时间差的数据
            nearest_row = day_data.loc[day_data['TimeDiff'].idxmin()]
            # 添加到结果DataFrame
            optimized_nearest_to_noon = pd.concat([optimized_nearest_to_noon, nearest_row.to_frame().T], ignore_index=True)

    # 删除辅助列
    optimized_nearest_to_noon.drop(columns=['TimeDiff', 'DOY_int'], inplace=True, errors='ignore')

    # 写入新的Excel文件
    nearest_to_noon_excel_path = './buoy_excel_to20_train/{}.xlsx'.format(file_name) #test or train
    optimized_nearest_to_noon.to_excel(nearest_to_noon_excel_path, index=False)

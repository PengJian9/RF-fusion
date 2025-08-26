from PIL import Image
import numpy as np
import cv2
from pyproj import Proj, transform
from datetime import datetime
import pandas as pd
import os
import re


def get_truth_txt(file_path,output_file_path):

    def get_truthdata_from_one_point(latitude,longitude):
        def get_projected_coordinates(latitude, longitude):
            x, y = polar_stereographic(longitude, latitude)
            return x, y

        def get_pixel_coordinates(x, y, corners, resolution):
            pixel_y = (x - corners['upper_left'][0]) / resolution
            pixel_x = (corners['upper_left'][1] - y) / resolution 
            return pixel_x, pixel_y

        proj_params = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        polar_stereographic = Proj(proj_params)
        x, y = get_projected_coordinates(latitude, longitude)

        corners = {
            'upper_left': (-3850000, 5850000),
            'upper_right': (3750000, 5850000),
            'lower_right': (3750000, -5350000),
            'lower_left': (-3850000, -5350000)
        }
        
        #36
        resolution = 12.5 * 1000  
        # #89
        # resolution = 6.25 * 1000  

        pixel_coordinates = get_pixel_coordinates(x, y, corners, resolution)

        point_x = pixel_coordinates[0]  
        point_y = pixel_coordinates[1]
         
        point_x_subregion=point_x - 248
        point_y_subregion=point_y - 104
        
        if point_x_subregion < 0 or point_x_subregion > 400 or point_y_subregion < 0 or point_y_subregion > 400:
            return (-1,-1)
        else:
            return (point_x_subregion , point_y_subregion)
    
    df = pd.read_excel(file_path)
    data_dict = {row['DateTime']: (row['Lat'], row['Lon']) for _, row in df.iterrows()}

    def add_day_of_year_to_dict(data_dict):

        for key in data_dict.keys():
            date_obj = datetime.fromisoformat(str(key))
            day_of_year = date_obj.timetuple().tm_yday
            date = str(date_obj.year) + str(date_obj.month).zfill(2) + str(date_obj.day).zfill(2)
            data_dict[key] = (data_dict[key][0], data_dict[key][1], date, day_of_year)

        return data_dict

    data_with_day_of_year = add_day_of_year_to_dict(data_dict)
    with open(output_file_path, 'w') as file:
        for key in data_with_day_of_year.keys():
            latitude, longitude, date, day_of_year = data_with_day_of_year[key]
            grid_position = get_truthdata_from_one_point(latitude, longitude)
            if grid_position == (-1,-1):
                continue       
            file.write(f"Year: {date} Day of Year: {day_of_year}\n")
            file.write(f"{grid_position}\n")
            file.write("\n") 


import os
import glob


root_folder = './buoy_excel_to20_train' #test or train
count = 0
file_path_list = []

for folder, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder, file)
            file_path_list.append(file_path)
            count += 1
            
for path in file_path_list:
    file_name, _ = os.path.splitext(os.path.basename(path))

    output_folder= './MCC_truth/buoy_pixel_pos/train/' #test or train
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = output_folder + file_name + '.txt'
    get_truth_txt(path,output_file_path)
    print('successfully save text to:' +output_file_path)
from PIL import Image
import numpy as np
import cv2
from pyproj import Proj, transform
from datetime import datetime
import pandas as pd
import os
import re






def process_buoy_data(file_paths):
    combined_buoy_data = []

    # Process each file
    for file_path in file_paths:
        buoy_number = os.path.basename(file_path).split('.')[0]

        with open(file_path, 'r') as file:
            file_contents = file.readlines()

        # Process each line in the file
        for line in file_contents:
            if line.strip():
                if 'Year:' in line:
                    year, day_of_year = re.findall(r'\d+', line)
                else:
                    if 'inf' in line:
                        continue
                    if '(nan, nan)' in line:
                        continue
                    buoy_positions = eval(line.strip())
                    x_pos, y_pos = buoy_positions
                    combined_buoy_data.append({
                        'buoy_number': buoy_number,
                        'position': (x_pos, y_pos),
                        'date': year,
                        'day_of_year': day_of_year
                    })

    reformatted_data = []
    for entry in combined_buoy_data:

        year = entry['date']
        day_of_year = entry['day_of_year']
        buoy_number = entry['buoy_number']
        position = entry['position']

        existing_entry = next((item for item in reformatted_data if item[0] == year and item[1] == day_of_year), None)

        if existing_entry:
            existing_entry[2][buoy_number] = position
        else:
            reformatted_data.append([year, day_of_year, {buoy_number: position}])

    return reformatted_data

def write_regions_to_files(reformatted_data, output_directory):
    output_files = []
    for region_index, data in reformatted_data.items():
        output_file_path = os.path.join(output_directory, f'region_{region_index}.txt')
        with open(output_file_path, 'w') as file:
            file.write(f"Region Index {region_index}:\n")
            for entry in data:
                file.write(f"\tYear: {entry[0]}, Day of Year: {entry[1]}, Buoys: {entry[2]}\n")
        output_files.append(output_file_path)





root_folder= './MCC_truth/buoy_pixel_pos/train' #test or train
output_truth_path='./MCC_truth/buoy_data/train' #test or train

if not os.path.exists(output_truth_path):
    os.makedirs(output_truth_path)
file_path_list = []

for folder, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(folder, file)
            file_path_list.append(file_path)


result = process_buoy_data(file_path_list)
# write_regions_to_files(result,output_file_path)

def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

for entry in result:
        displacements = {}
        year = entry[0][:4]
        day_of_year = entry[1]
        buoy_data = entry[2]
        flag=0
        for item in result: #迭代找到下一天的数据
            if is_leap_year(int(year)):
                if day_of_year == '366':
                    if item[0][:4] == str(int(year)+1) and item[1] == '1':
                        next_day_buoys = item[2]
                        for buoy_number, current_position in buoy_data.items():
                            if buoy_number in next_day_buoys:
                                next_position = next_day_buoys[buoy_number]
                                displacement = (current_position[0], current_position[1], next_position[0] - current_position[0], next_position[1] - current_position[1])
                                displacements[buoy_number] = displacement
                        flag=1    
                        break
                else:
                    if item[0][:4] == year and item[1] == str(int(day_of_year) + 1):
                        next_day_buoys = item[2]
                        for buoy_number, current_position in buoy_data.items():
                            if buoy_number in next_day_buoys:
                                next_position = next_day_buoys[buoy_number]
                                displacement = (current_position[0], current_position[1], next_position[0] - current_position[0], next_position[1] - current_position[1])
                                displacements[buoy_number] = displacement
                        flag=1  
                        break
            else:
                if day_of_year == '365':
                    if item[0][:4] == str(int(year)+1) and item[1] == '1':
                        next_day_buoys = item[2]
                        for buoy_number, current_position in buoy_data.items():
                            if buoy_number in next_day_buoys:
                                next_position = next_day_buoys[buoy_number]
                                displacement = (current_position[0], current_position[1], next_position[0] - current_position[0], next_position[1] - current_position[1])
                                displacements[buoy_number] = displacement
                        flag=1  
                        break
                else:
                    if item[0][:4] == year and item[1] == str(int(day_of_year) + 1):
                        next_day_buoys = item[2]
                        for buoy_number, current_position in buoy_data.items():
                            if buoy_number in next_day_buoys:
                                next_position = next_day_buoys[buoy_number]
                                displacement = (current_position[0], current_position[1], next_position[0] - current_position[0], next_position[1] - current_position[1])
                                displacements[buoy_number] = displacement
                        flag=1  
                        break
        if flag==0:
            continue
        else:                       
            file_path_folder=os.path.join(output_truth_path, f'region_{year}')
            if not os.path.exists(file_path_folder):
                os.makedirs(file_path_folder)
            filename=f'region_{year}_{day_of_year}.txt'
            file_path_displacement=os.path.join(file_path_folder, filename)
            with open(file_path_displacement, 'w') as file:
                for buoy_number, displacement in displacements.items():
                    file.write(f"Buoy Number: {buoy_number}, Displacement: {displacement}\n")
            print(f"File {filename} written successfully")

    
        








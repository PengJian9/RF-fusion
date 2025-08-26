import os
from math import floor, ceil
import numpy as np

def search_txt_files(directory):
    txt_files = []

    # 遍历指定目录及其子目录下的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        # 在当前目录下搜索所有的 txt 文件
        for file in files:
            if file.endswith(".txt"):
                # 构造文件的绝对路径
                file_path = os.path.join(root, file)
                txt_files.append(file_path)

    return txt_files

def read_buoy_data(filepath):
    buoy_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            if "Buoy Number" in line and "Displacement" in line:
                parts = line.strip().split(", Displacement: ")
                number_part = parts[0].split("Buoy Number: ")[1]
                displacement_part = parts[1].strip("()")
                # 转换位移数据为浮点数元组
                displacement = tuple(map(float, displacement_part.split(", ")))
                buoy_data[number_part] = displacement
    return buoy_data

def calculate_SIC(x,y,SIC):
    # 计算x,y处11*11邻域的SIC均值，x,y为整数
    SIC_neighborhood = SIC[x-5:x+6, y-5:y+6]
    #取小于100的值计算均值
    SIC_vaild = SIC_neighborhood[SIC_neighborhood <= 100]
    mean_SIC = np.mean(SIC_vaild)
    return mean_SIC


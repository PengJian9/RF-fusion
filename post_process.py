import numpy as np


def filter_outlier(drift_map):
    original_drift_map = drift_map.copy() 
    for x in range(0, drift_map.shape[0]):
            for y in range(0, drift_map.shape[1]):
                if ~np.isnan(drift_map[x][y][0]) and ~np.isnan(drift_map[x][y][1]):
                    count = 0
                    x_list = []
                    y_list = []
                    for i in range(-2,3):
                        for j in range(-2,3):
                            if x+i < 0 or x+i >= drift_map.shape[0] or y+j < 0 or y+j >= drift_map.shape[1]:
                                continue
                            if ~np.isnan(drift_map[x+i][y+j][0]) and ~np.isnan(drift_map[x+i][y+j][1]):
                                count += 1
                                x_list.append(drift_map[x+i][y+j][0])
                                y_list.append(drift_map[x+i][y+j][1])

                    if count < 10:
                        drift_map[x][y][0] = np.nan
                        drift_map[x][y][1] = np.nan
                    else:
                        
                        x_mean, y_mean= np.mean(x_list),np.mean(y_list)
                        x_std, y_std = np.std(x_list), np.std(y_list)
                        if abs(drift_map[x][y][0] - x_mean) > 3 * x_std or abs(drift_map[x][y][1] - y_mean) > 3 * y_std:
                            drift_map[x][y][0] = np.nan
                            drift_map[x][y][1] = np.nan
    return drift_map
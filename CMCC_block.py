import cv2
import numpy as np
from skimage.transform import resize
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
import img_process as ip

def CMCC(image1, image2, SIC, index,template_size):
    img_h,img_w=image1.shape

    drift=[]
    drift_map=np.full((img_h,img_w,2),np.nan) # 2:x,y
    
    for x in range(0, image1.shape[0] - template_size + 1):
        for y in range(0, image1.shape[1] - template_size + 1):
            template = image1[x:x+template_size, y:y+template_size]
            SIC_vaild = SIC[x:x+template_size, y:y+template_size]
            # Filter out pixels in SIC with values between 0 and 100
            
            if np.sum(SIC_vaild[SIC_vaild==120]) >= 0.7 * template_size ** 2:
                continue
            else:
                SIC_vaild_filtered = SIC_vaild[SIC_vaild <= 100]
                if np.mean(SIC_vaild_filtered) < 30:
                    continue

            if x-5<0 or y-5<0 or x+5+template_size>image1.shape[0] or y+5+template_size>image1.shape[1]:
                continue

            corr_each_temp=[]
            search_range=5
            image2_search_range=image2[x - search_range : x + search_range + template_size, y - search_range : y + search_range + template_size]
            h_search, w_search = image2_search_range.shape
            h_step = np.arange(0, float(h_search - template_size + 1), 0.5)
            w_step = np.arange(0, w_search - template_size + 1, 0.5)
            corr_map=np.zeros((len(h_step),len(w_step)))
            for i in range(len(h_step)):
                for j in range(len(w_step)):
                    col_grid, row_grid=ip.generate_coordinate_grid(w_step[j],h_step[i],template_size)
                    interpolated_image = cv2.remap(image2_search_range, col_grid.astype(np.float32), row_grid.astype(np.float32), cv2.INTER_LINEAR)  #注意这里的x,y 先是col 再是 row 对应y，x
                    result = cv2.matchTemplate(interpolated_image, template, cv2.TM_CCORR_NORMED)
                    corr_each_temp.append((h_step[i], w_step[j], result))
                    corr_map[i,j] = -result ##用于计算最小值，所以用了相反数
        
            # 找到corr_each_temp中最大的相关性，及其像素位置
            max_corr = max(corr_each_temp, key=lambda x: x[2])
            max_corr_value_ini = max_corr[2]
            max_corr_position_ini = (max_corr[0], max_corr[1])  # x,y

            # 构建样条插值模型
            interpolator = RectBivariateSpline(h_step, w_step, corr_map,bbox=[min(h_step), max(h_step), min(w_step), max(w_step)])
            bounds = [(np.min(h_step), np.max(h_step)), (np.min(w_step), np.max(w_step))]
            
            def objective_function(params):
                x, y = params
                return interpolator(x, y)

            # 设置初始猜测值
            x0 = [max_corr[0], max_corr[1]]

            # 使用 Nelder-Mead 算法进行优化
            result = minimize(objective_function, x0, method='nelder-mead',bounds=bounds)
            
            max_corr_value = - result.fun

            # 获取对应的参数值大小
            max_corr_position = result.x

            if max_corr_value > 0.6:
                ori_des=[search_range, search_range]
                distance = (max_corr_position[1] - ori_des[1], max_corr_position[0] - ori_des[0])  #(y, x)
                drift.append([(y + template_size // 2, x + template_size // 2), distance])  #为了画图，所以x,y反了
                drift_map[x + template_size // 2,y + template_size // 2, 0]=distance[1] #x
                drift_map[x + template_size // 2,y + template_size // 2, 1]=distance[0] #y
                
                print(str(index)+ ' ' + str(len(drift)) + " 加入了一个运动")
    return drift_map



def CMCC_modify(image1, image2, SIC, index,template_size):
    img_h,img_w=image1.shape

    drift=[]
    drift_map=np.full((img_h,img_w,2),np.nan) # 2:x,y
    
    for x in range(0, image1.shape[0] - template_size + 1):
        for y in range(0, image1.shape[1] - template_size + 1):
            template = image1[x:x+template_size, y:y+template_size]
            SIC_vaild = SIC[x:x+template_size, y:y+template_size]
            # Filter out pixels in SIC with values between 0 and 100
            
            if np.sum(SIC_vaild[SIC_vaild==120]) >= 0.7 * template_size ** 2:
                continue
            else:
                SIC_vaild_filtered = SIC_vaild[SIC_vaild <= 100]
                if np.mean(SIC_vaild_filtered) < 30:
                    continue

            if x-5<0 or y-5<0 or x+5+template_size>image1.shape[0] or y+5+template_size>image1.shape[1]:
                continue

            corr_each_temp=[]
            search_range=5
            image2_search_range=image2[x - search_range : x + search_range + template_size, y - search_range : y + search_range + template_size]
            h_search, w_search = image2_search_range.shape

            resize_image2_search_range = cv2.resize(image2_search_range, (h_search * 2, w_search * 2), interpolation=cv2.INTER_LINEAR)
            resize_template=cv2.resize(template, (template_size * 2, template_size * 2), interpolation=cv2.INTER_LINEAR)

            result = cv2.matchTemplate(resize_image2_search_range, resize_template, cv2.TM_CCORR_NORMED)
            result = np.transpose(result)
            
            #找到最大值及最大值的位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)



            resize_h, resize_w = resize_image2_search_range.shape
            resize_template_h, resize_template_w = resize_template.shape
            h_step = np.arange(0, resize_h - resize_template_h + 1)
            w_step = np.arange(0, resize_w - resize_template_w + 1)
            

            # 构建样条插值模型
            interpolator = RectBivariateSpline(h_step, w_step, -result,bbox=[min(h_step), max(h_step), min(w_step), max(w_step)])
            bounds = [(np.min(h_step), np.max(h_step)), (np.min(w_step), np.max(w_step))]
            
            def objective_function(params):
                x, y = params
                return interpolator(x, y)

            # 设置初始猜测值
            x0 = [max_loc[0], max_loc[1]]

            # 使用 Nelder-Mead 算法进行优化
            result = minimize(objective_function, x0, method='nelder-mead',bounds=bounds)
            
            max_corr_value = - result.fun

            # 获取对应的参数值大小
            max_corr_position = result.x
            y_match = float(max_corr_position[0]) / 2
            x_match = float(max_corr_position[1]) / 2

            if max_corr_value > 0.6:
                ori_des=[search_range, search_range]
                distance = (y_match - ori_des[1], x_match - ori_des[0])  #(y, x)
                drift.append([(y + template_size // 2, x + template_size // 2), distance])  #为了画图，所以x,y反了
                drift_map[x + template_size // 2,y + template_size // 2, 0]=distance[1] #x
                drift_map[x + template_size // 2,y + template_size // 2, 1]=distance[0] #y
                
                print(str(index)+ ' ' + str(len(drift)) + " 加入了一个运动")
    return drift_map

def CMCC_modify_for_comparision(image1, image2, SIC, index,template_size):
    img_h,img_w=image1.shape

    drift=[]
    drift_map=np.full((img_h,img_w,2),np.nan) # 2:x,y
    
    for x in range(0, image1.shape[0] - template_size + 1):
        for y in range(0, image1.shape[1] - template_size + 1):
            template = image1[x:x+template_size, y:y+template_size]
            SIC_vaild = SIC[x:x+template_size, y:y+template_size]
            # Filter out pixels in SIC with values between 0 and 100
            
            if np.sum(SIC_vaild[SIC_vaild==120]) >= 0.7 * template_size ** 2:
                continue
            else:
                SIC_vaild_filtered = SIC_vaild[SIC_vaild <= 100]
                if np.mean(SIC_vaild_filtered) < 30:
                    continue

            if x-5<0 or y-5<0 or x+5+template_size>image1.shape[0] or y+5+template_size>image1.shape[1]:
                continue

            corr_each_temp=[]
            search_range=5
            image2_search_range=image2[x - search_range : x + search_range + template_size, y - search_range : y + search_range + template_size]
            h_search, w_search = image2_search_range.shape

            resize_image2_search_range = cv2.resize(image2_search_range, (h_search * 2, w_search * 2), interpolation=cv2.INTER_LINEAR)
            resize_template=cv2.resize(template, (template_size * 2, template_size * 2), interpolation=cv2.INTER_LINEAR)

            result = cv2.matchTemplate(resize_image2_search_range, resize_template, cv2.TM_CCORR_NORMED)
            result = np.transpose(result)
            
            #找到最大值及最大值的位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)



            resize_h, resize_w = resize_image2_search_range.shape
            resize_template_h, resize_template_w = resize_template.shape
            h_step = np.arange(0, resize_h - resize_template_h + 1)
            w_step = np.arange(0, resize_w - resize_template_w + 1)
            

            # 构建样条插值模型
            interpolator = RectBivariateSpline(h_step, w_step, -result,bbox=[min(h_step), max(h_step), min(w_step), max(w_step)])
            bounds = [(np.min(h_step), np.max(h_step)), (np.min(w_step), np.max(w_step))]
            
            def objective_function(params):
                x, y = params
                return interpolator(x, y)

            # 设置初始猜测值
            x0 = [max_loc[0], max_loc[1]]

            # 使用 Nelder-Mead 算法进行优化
            result = minimize(objective_function, x0, method='nelder-mead',bounds=bounds)
            
            max_corr_value = - result.fun

            # 获取对应的参数值大小
            max_corr_position = result.x
            y_match = float(max_corr_position[0]) / 2
            x_match = float(max_corr_position[1]) / 2

            if max_corr_value > 0.3:
                ori_des=[search_range, search_range]
                distance = (y_match - ori_des[1], x_match - ori_des[0])  #(y, x)
                drift.append([(y + template_size // 2, x + template_size // 2), distance])  #为了画图，所以x,y反了
                drift_map[x + template_size // 2,y + template_size // 2, 0]=distance[1] #x
                drift_map[x + template_size // 2,y + template_size // 2, 1]=distance[0] #y
                
                print(str(index)+ ' ' + str(len(drift)) + " 加入了一个运动")
    return drift_map
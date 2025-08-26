import cv2
import numpy as np
import os
from PIL import Image
import rasterio
from skimage.transform import resize


def find_img_and_read(filename,base_path):
    file_path = base_path+'/'+filename
    image=Image.open(file_path)
    image=100 * np.array(image,dtype='uint16')
    image=image[248:648,104:504]
    
    return image

def find_img_and_read_89(filename,base_path):
    file_path = base_path+'/'+filename
    image=Image.open(file_path)
    image=100 * np.array(image,dtype='uint16')
    image=image[496:1296,208:1008]
    
    return image

def find_SIC_and_read(filename,base_path):
    file_path = base_path+'/'+filename
    with rasterio.open(file_path) as dataset:
        # 读取第一波段的数据
        SIC = dataset.read(1)
    #SIC降采样
    SIC = resize(SIC, (SIC.shape[0]/2, SIC.shape[1]/2), 
                        anti_aliasing=True, 
                        preserve_range=True)    
    SIC=SIC[248:648:,104:504]
    
    mask = np.zeros_like(SIC)
    for i in range(SIC.shape[0]):
        for j in range(SIC.shape[1]):
            if SIC[i][j] <= 100 and SIC[i][j] >= 15:
                mask[i][j] = 1


    return SIC,mask

def find_SIC_and_read_89(filename,base_path):
    file_path = base_path+'/'+filename
    with rasterio.open(file_path) as dataset:
        # 读取第一波段的数据
        SIC = dataset.read(1)
    #SIC降采样  
    SIC=SIC[496:1296,208:1008]
    
    mask = np.zeros_like(SIC)
    for i in range(SIC.shape[0]):
        for j in range(SIC.shape[1]):
            if SIC[i][j] <= 100 and SIC[i][j] >= 15:
                mask[i][j] = 1


    return SIC,mask



def img16to8(image_16bit):
    # 计算最小和最大灰度值
    min_val, max_val, _, _ = cv2.minMaxLoc(image_16bit)

    # 缩放和截断灰度值
    image_8bit = np.uint8((image_16bit - min_val) * 255 / (max_val - min_val))
    return image_8bit



def Histogram_equal(image,mask):
    # 计算直方图,只统计mask中为1的像素点
    
    image = image.astype(np.uint16)
    hist, bins = np.histogram(image[mask == 1].flatten(), bins=2**16, range=(0, 2**16-1))

    # 计算累积直方图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # 构建直方图均衡化的映射表
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(image.dtype)

    equalized_image = cdf[image]
    # cv2.imwrite('1/equalized_image.png',equalized_image)
    # cv2.imwrite('1/image.png',image)
    # equalized_image = equalized_image.astype(np.float32)
    return equalized_image

def Histogram_equal_8(image,mask):
    # 计算直方图,只统计mask中为1的像素点
    
    hist, bins = np.histogram(image[mask == 1].flatten(), bins=2**8, range=(0, 2**8-1))

    # 计算累积直方图
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # 构建直方图均衡化的映射表
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(image.dtype)

    equalized_image = cdf[image]

    return equalized_image


def generate_coordinate_grid(start_x, start_y, patch_size):
    # 生成11*11的坐标网格
    x_grid, y_grid = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    
    # 考虑起始坐标可能不是整数，生成浮点数类型的网格
    x_grid = x_grid + start_x
    y_grid = y_grid + start_y
    
    return x_grid, y_grid

def LoG_8U(src, ksize, sigma=0):
    blur_img = cv2.GaussianBlur(src, (ksize, ksize), sigmaX=sigma)
    LoG_img = cv2.Laplacian(blur_img, cv2.CV_8U)
    return LoG_img

def LoG_64F(src, ksize, sigma=0):
    blur_img = cv2.GaussianBlur(src, (ksize, ksize), sigmaX=sigma)
    LoG_img = cv2.Laplacian(blur_img, cv2.CV_64F)
    return LoG_img

def LoG_32F(src, ksize, sigma=0):
    blur_img = cv2.GaussianBlur(src, (ksize, ksize), sigmaX=sigma)
    LoG_img = cv2.Laplacian(blur_img, cv2.CV_32F)
    # LoG_img = cv2.Laplacian(src, cv2.CV_32F)


    laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)



    laplacian_output = cv2.filter2D(blur_img.astype(np.float32), -1, laplacian_kernel)
    sharpening_image = blur_img + laplacian_output
    # cv2.imwrite('2/sharpening_image_2.png',sharpening_image)
    # cv2.imwrite('2/LoG_img_self_1.png',laplacian_output)
    # cv2.imwrite('2/LoG_img_1.png',LoG_img)
    # cv2.imwrite('2/blur_img.png',blur_img)
    # cv2.imwrite('2/src.png',src)
    return LoG_img


def sobel(img, ksize,sigma=0):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

    # 计算x方向的梯度
    grad_x = cv2.Sobel(blur_img, cv2.CV_32F, 1, 0, ksize=ksize)
    # 计算y方向的梯度
    grad_y = cv2.Sobel(blur_img, cv2.CV_32F, 0, 1, ksize=ksize)
    return grad_x + grad_y

def sobel_8u(img, ksize,sigma=0):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

    # 计算x方向的梯度
    grad_x = cv2.Sobel(blur_img, cv2.CV_8U, 1, 0, ksize=ksize)
    # 计算y方向的梯度
    grad_y = cv2.Sobel(blur_img, cv2.CV_8U, 0, 1, ksize=ksize)
    return grad_x + grad_y


def canny(img,ksize,sigma=0):
    blur_img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    canny = cv2.Canny(blur_img, 50, 150) 

    return canny
# -*- coding: utf-8 -*-
# @Time	: 2019.07.15
# @Author  : Bohrium Kwong
# @Licence : bio-totem

import numpy as np
import cv2
import sys
import scipy.ndimage

sys.path.append('../')
from utils.opencv_utils import OpenCV

from utils.tissue_utils import get_tissue



# patch_size = 224
def matrix_resize(slide,region_prediction_result,patch_size,svs_level=2):
    """将区域分类结果的矩阵resize成svs图片对应尺寸.

    # Arguments
        region_prediction_result: Numpy array;the result of region classifition by deep learning models.
        patch_size: the size of region you read in the svs_file.
            either "224" or "512".
        file_format: the value of openslide 's leve arguments.
            almost set a default value by 2
    """

    slide_width, slide_height = slide.get_level_dimension(0)

    w_count = slide_width // patch_size
    h_count = slide_height // patch_size

    level_downsamples = int(slide.get_level_downsample(level = svs_level))


    out_img = OpenCV(region_prediction_result).resize(
        (w_count * patch_size /level_downsamples),
        (h_count * patch_size /level_downsamples),
        interpolation=cv2.INTER_NEAREST)

    out_img = cv2.copyMakeBorder(out_img,0,int(slide.get_level_dimension(level=svs_level)[1]-out_img.shape[0]),0,
                                     int(slide.get_level_dimension(level=svs_level)[0]-out_img.shape[1]),cv2.INTER_NEAREST)
    return out_img


"""
去除矩阵中的离散点
matrix: Numpy矩阵，浮点类型
size: OPENCV开操作卷积核大小
"""


def remove_discrete_point(matrix, size):
    # 尽可能保证matrix类型为uint8类型
    # 因为在当前项目的测试下，该函数处理float类型耗费的时间比uint8类型高出100倍
    # float类型实测为17秒，uint8类型实测为0.1秒
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    matrix = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel)

    return matrix


"""
获取组织区域MASK（0为非组织区域，1为组织区域）
bgimg_rgb: RGB三通道的图片矩阵
"""
def get_mask(bgimg_rgb):
    mask, _ = get_tissue(bgimg_rgb, contour_area_threshold=10000)
    mask = remove_discrete_point(mask, 50)
    # 仅获取外部轮廓
    # contours = get_contours(mask,cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')
    mask = fill_contours(mask.shape, contours)

    return mask


'''
填充轮廓到二维矩阵中，轮廓区域为1，非轮廓区域为0
shape：二维矩阵的shape
cnt_list：轮廓列表
'''


def fill_contours(shape, cnt_list):
    result_matrix = np.full((shape[0], shape[1]), 0, dtype=np.uint8)
    cv2.drawContours(result_matrix, cnt_list, -1, 1, cv2.FILLED)
    return result_matrix



'''
求中心点（轮廓质心)坐标x，y
cnt:单个轮廓
'''
def get_centerpoint(cnt):
    M = cv2.moments(cnt)
    return [int(M["m10"] / M["m00"]) ,int(M["m01"] / M["m00"])]



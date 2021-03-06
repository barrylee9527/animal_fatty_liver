#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 06:11:39 2018

@author: Bohrium Kwong
"""

import numpy as np
import cv2
import skimage.io
import skimage.color
import copy
from cell_classifition import deconvolution
import tensorflow as tf
from keras.models import *
from keras.models import Model
from keras import optimizers
from scipy.misc import imresize
from keras.utils import multi_gpu_model


from utils.log_utils import get_logger

#    stains = {
#     'hematoxylin': [0.63100845,  0.63630474,    0.44378445],
#     'dab':         [0.40348947,  0.59615797,    0.6941123],
#     'res':         [0.6625893,   0.48960388,    0.5668011 ]
#     }


def read_process(size_img,im_input,mask_data,coordinate,large_flag,model,patch_size):
    """
    cell caculating process of each input image
    :param size_img: the patch size of processing on the Whole slide svs iamges
    :param im_input: INput image,notice that the size of image is bigger than the size of mask
    :param mask_data: the out put  of maskrcnn_detection
    :param seg_model: mask-rcnn segmentation model
    :param coordinate: the coordinate of each imput image on Whole slide svs iamges,tupple,the 1st element for x-weight,2nd for y-height
    :param large_flag: tuppple,the 1st element input image reach bottom edge or not,2nd element fot reach right edge or not,if do,value is 0 other is 1
    :param model: the keras model for cell classification ,the lable tag is [epithelial,fibroblast,inflammatory,miscellaneous]
    :param patch_size:define by model,eche nuclei cut pach size
    :return:the array of nuclei_info, include the predict_porba of classification 
    """

    cell_sum = 0
    region_raw_tensor = np.zeros((1,72,72,3))
    nuclei_info = np.ones((1,4))
    image_x = size_img
    image_y = size_img
    for c in range(mask_data.shape[2]):

        _, contours, _, = cv2.findContours(np.uint8(mask_data[:,:,c]),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        
        if len(contours) > 0:
            for r in range(len(contours)):
                cx,cy,w,h = cv2.boundingRect(contours[r])
                y1 = cx + w/2
                x1 = cy + h/2
                y_c = copy.deepcopy(y1)
                x_c = copy.deepcopy(x1)
                nuclei_flag = 1
                bottom_x = 1
                bottom_y = 1
                if x1 < 10:
                    bottom_x = 0
                elif x1 <= patch_size/2 + 1:
                    x1 = int(patch_size/2 + 1)
                elif x1 > image_x and large_flag[0] == 1:
                    bottom_x = 0
                elif x1 > image_x - int(patch_size/2) and large_flag[0] == 0:
                    bottom_x = 0
                
                if y1 < 10:
                    bottom_y = 0
                elif y1 <= patch_size/2 + 1:
                    y1 = int(patch_size/2 + 1)
                elif y1 > image_y and large_flag[1] == 1:
                    bottom_y = 0
                elif y1 > image_y - int(patch_size/2) and large_flag[1] == 0:
                    bottom_y = 0                                 
                    
                if bottom_x * bottom_y ==1:
                    contour_area = cv2.contourArea(contours[r])
                    
                    region = im_input[int(x1 - patch_size/2):int(x1 - patch_size/2) + patch_size,
                                                     int(y1 - patch_size/2):int(y1 - patch_size/2) + patch_size,:]
                    region = cv2.resize(region,(72,72), interpolation=cv2.INTER_AREA)
                    region_contous = im_input[cy:cy+h,cx:cx+w,:]
                    
                    region_flag = (abs(region_contous[:,:,0]-234)<=17) & (abs(region_contous[:,:,1]-234)<=17) & (abs(region_contous[:,:,2]-234)<=17)
#                    if np.sum(region_flag) <= w * h * 100:
                    nuclei_flag = np.sum(region_flag)/ w / h

                    if cell_sum ==0:
                        region_raw_tensor[0,:,:,:] = region/255 
                        
                        nuclei_info[0,0] = y_c + coordinate[0]
                        nuclei_info[0,1] = x_c + coordinate[1]
                        nuclei_info[0,2] = contour_area
                        nuclei_info[0,3] = nuclei_flag
                    else:
                        region_raw_tensor = np.row_stack((region_raw_tensor,np.expand_dims(region/255, axis=0)))
                        nuclei_info = np.row_stack((nuclei_info,np.array([y_c + coordinate[0],x_c + coordinate[1],
                                                                              contour_area,nuclei_flag])))
                        
                    cell_sum = cell_sum + 1
    if region_raw_tensor.shape[0]==1 and nuclei_info[0,0]==1:
        preds_c_l = np.zeros((1,2))
    else:                
        preds_c_l = model.predict_proba(region_raw_tensor)                        
                                                     
    return np.column_stack((nuclei_info,preds_c_l))

def cancer_cell_caculating(size_img,image_list,mask_list,coordinate_list,large_flag_list,model,patch_size):
    #print ("Img numbers : "+str(len(image_list)))
    #print ("mask numbers: "+str(len(mask_list)))
    is_success_flag_list =[]
    image_result_list=[]
 
    for i in range(len(image_list)):
        is_success_flag = 1
        im_input = image_list[i]
        mask_data = mask_list[i]
        large_flag = large_flag_list[i]
        coordinate = coordinate_list[i]
        try:
            result = read_process(size_img,im_input,mask_data,coordinate,large_flag,model,patch_size)
        except Exception as e: #if program with wrong
            error_logger = get_logger(level="error")
            error_logger.error('Cell Classification Error', exc_info=True)
            is_success_flag =0
            result = []
            image_result_list.append(result)
            is_success_flag_list.append(is_success_flag)
        else:
            if len(result)==1 and result[0,0]==1:
               is_success_flag =0
               result = []
            image_result_list.append(result)
            is_success_flag_list.append(is_success_flag)
                 
    
    return image_result_list,is_success_flag_list

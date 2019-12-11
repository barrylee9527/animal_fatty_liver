#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 07:29:57 2019

@author: barry
"""

import cv2
import os
import numpy as np
import skimage
import math
from matplotlib import pyplot as plt
import glob
import sys
sys.path.append('/cptjack/totem/barrylee/cv')
import openslide as opsl
from t import Faststainsep
from skimage import io
from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.miscellaneous.optical_density_conversion import convert_OD_to_RGB
from staintools.miscellaneous.get_concentrations import get_concentrations
import SimpleITK as sitk
import pickle
import keras.backend as K
import random


def takeSecond(elem):
    return elem[1]


def remove_discrete_point(matrix, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    matrix = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel)
    return matrix
# 计算轮廓公共面积
def commonArea(rec1, rec2):
    l_x = max(rec1[0], rec2[0])
    d_y = max(rec1[1], rec2[1])
    r_x = min(rec1[2], rec2[2])
    u_y = min(rec1[3], rec2[3])
    length = r_x - l_x
    wide = u_y - d_y
    if length > 0 and wide > 0:
        return length * wide / ((rec2[2] - rec2[0]) * (rec2[3] - rec2[1]))
    return 0

extractor = VahadaneStainExtractor
# extractor = MacenkoStainExtractor
img_name = 'LIV057003 STZ+HFD #123'
img_dir = '/cptjack/totem/barrylee/NASH annotation/raw/'
biaozhu = '/cptjack/totem/barrylee/NASH annotation/new-annoatation/'
mask_dir = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/output/mask_dir/'
img_num = os.listdir(img_dir)
random.shuffle(img_num)
print(img_num)
colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 0), (128, 42, 42)]
# out_mask = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/mask_out/inflammation/'
own_slide = 54
contours_num = 0
skip_num = 0
slide = 27
q=0
# if not os.path.exists(out_mask):
#     os.makedirs(out_mask)
for i in range(len(img_num)):
    # out_path = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/real/ballooning/' + img_num[i].split('.')[0] + '.png'
    # our_mask_np = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/output/inflammation/'+ img_num[i].split('.')[0] + '.npy'
    biaozhu_img = biaozhu+ img_num[i].split('.')[0] + '-marking'+ '.png'
    pict_name = img_num[i]
    picture_name1 = img_num[i].split('-')
    picture_name2 = ''
    for s in picture_name1[:-2]:
        picture_name2 = picture_name2 + s + '-'
    picture_name2 = picture_name2[:-4]
    print(picture_name2)
    sum_dir = '/cptjack/totem/barrylee/cut_small_cell/cell_tri-classification/train54-'+img_name+'/'
    I_raw = cv2.imread(img_dir + pict_name)
    biaozhu_raw = cv2.imread(biaozhu_img)
    print(biaozhu_img)
    flag_path = pict_name.split('.')[0]
    I = cv2.cvtColor(I_raw, cv2.COLOR_BGR2RGB)
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    stain_matrix_source = extractor.get_stain_matrix(I)  # ,regularizer=0.8)
    source_concentrations = get_concentrations(I, stain_matrix_source)  # , regularizer=0.1)
    b = source_concentrations.reshape((I.shape[0], I.shape[1], 2)).astype(np.uint8)
    # 打开保存的npy格式的mask文件
    with open(
            mask_dir + pict_name.split('.')[
                0] + '.npy',
            'rb') as fr:
        print(
            mask_dir + pict_name.split('.')[
                0] + '.npy')
        mask = np.load(fr)
    mask_sum = []
    cnt_list = []
    # 将轮廓和面积存入一个变量
    for j in range(mask.shape[2]):
        outline = mask[:, :, j]
        outline = outline.astype(np.uint8)
        # print(outline)
        image, contours, hierarchy = cv2.findContours(outline, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for k in range(len(contours)):
            area = cv2.contourArea(contours[k])
            mask_sum.append((contours[k], area))
        mask_sum.sort(key=takeSecond, reverse=True)
    mask_coordinate = []
    print(len(mask_sum))
    # 如果两个轮廓重叠占比大于0.8则需要过滤
    for k in range(len(mask_sum)):
        x, y, w, h = cv2.boundingRect(mask_sum[k][0])
        cx = round(x + w / 2)
        cy = round(y + h / 2)
        r = math.floor(min(w, h) / 2)
        flag = False
        mask_coordinate.append((x, y, x + w, y + h))
        for l in range(k):
            if commonArea(mask_coordinate[l], mask_coordinate[k]) > 0.8:
                #                cv2.drawContours(I_raw, mask_sum[k][0], -1, colors[1], 1)
                flag = True
                break
        if flag:
            continue
        def fill_contours(matrix, cnt):
            result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.uint8)
            cv2.drawContours(result_matrix, [cnt], -1, 1, cv2.FILLED)
            result_matrix *= matrix
            return result_matrix
        matrix = fill_contours(I_gray, mask_sum[k][0]).flatten()
        matrix = matrix[matrix != 0]
        bound_point = 205
        bound_line = 0.75
        bound_median = 195
        f = False
        # print(matrix[matrix > bound_point].shape[0])
        # 过滤掉轮廓过白的不是细胞核的轮廓
        if (matrix[matrix > bound_point].shape[0] / matrix.shape[0]) > bound_line or np.median(
                matrix) > bound_median:
            f = True
        if (cx == 300) or (cy == 300):
            cx, cy = 299, 299
        region_contous_0 = b[cy, cx, 0] if r == 0 else b[cy - r:cy + r, cx - r:cx + r, 0]
        region_contous_1 = b[cy, cx, 1] if r == 0 else b[cy - r:cy + r, cx - r:cx + r, 1]
        #            print('np.max(region_contous_0):',np.max(region_contous_0))
        #            print('np.max(region_contous_1):',np.max(region_contous_1))
        #            print('r',r)
        #        cv2.drawContours(I_raw, mask_sum[k][0], -1, (0,0,255), 1)
        if np.size(region_contous_0) != 0:
            if (np.max(region_contous_0) != 0) and f == False:  # and(np.max(region_contous_1)==0)
                M1 = cv2.moments(mask_sum[k][0])
                try:
                    cent_x = int(M1['m10'] / M1['m00'])
                    cent_y = int(M1['m01'] / M1['m00'])
                except ZeroDivisionError:
                    cent_x = round(x + w / 2)
                    cent_y = round(y + h / 2)
                slide = min(w, h)
                b_cx, b_cy, b_w, b_h = cv2.boundingRect(mask_sum[k][0])
                x1 = cent_x - math.ceil(slide / 2)
                y1 = cent_y - math.ceil(slide / 2)
                x2 = cent_x + math.ceil(slide / 2)
                y2 = cent_y + math.ceil(slide / 2)
                # if (cent_x < math.ceil(own_slide / 2)) or (cent_y < math.ceil(own_slide / 2)):
                #     continue
                pixcel_space = 0
                # 得到原图的RGB
                region_r = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 2]
                region_g = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 1]
                region_b = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 0]
                # 计算标注的颜色像素
                red = ((region_r==255) & (region_g==0) & (region_b==0))
                yellow = ((region_r==255) & (region_g==255) & (region_b==0))
                blue = ((region_r==0) & (region_g==0) & (region_b==255))
                purple = ((region_r==255) & (region_g==0) & (region_b==255))
                # print(np.sum(red),np.sum(yellow),np.sum(blue),np.sum(purple))
                red,yellow,blue,purple = np.sum(red),np.sum(yellow),np.sum(blue),np.sum(purple)
                if picture_name2==img_name:
                    sum_dir = '/cptjack/totem/barrylee/cut_small_cell/cell_tri-classification/val54-'+img_name+'/'
                if(red>30 and red>yellow):
                    dir_name = sum_dir + 'hepatocyte'
                elif (purple > 30 and purple > yellow):
                    dir_name = sum_dir + 'hepatocyte'
                elif (blue > 30 and blue > yellow):
                    dir_name = sum_dir + 'hepatocyte'
                elif(yellow>30 and yellow>red and yellow>blue and yellow>purple):
                    dir_name = sum_dir + 'immune_cells'
                elif(red <=30 and blue <=30 and yellow <=30 and purple <=30):
                    dir_name = sum_dir + 'other_cells'
                x1 = cent_x - math.ceil(own_slide / 2)
                y1 = cent_y - math.ceil(own_slide / 2)
                x2 = cent_x + math.ceil(own_slide / 2)
                y2 = cent_y + math.ceil(own_slide / 2)
                w_ = y2-y1
                h_ = x2-x1
                print(w_,h_)
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    if w_==own_slide and h_==own_slide:
                        cropImg = I_raw[y1:y2, x1:x2]
                        # moved_pixel_img = I_raw[y1:y2, x1+6:x2+6]
                        # re_move_img = I_raw[y1+6:y2+6, x1:x2]
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name)
                        ss = dir_name + '/' + str(flag_path) + str(k) + '.png'
                        # print(ss)
                        # vv = dir_name + '/' + str(flag_path) + str(k) +'-r6p-'+ '.png'
                        # print(vv)
                        # tt = dir_name + '/' + str(flag_path) + str(k) +'-u6p-'+ '.png'
                        # print(tt)
                        cv2.imwrite(ss, cropImg)
                        # cv2.imwrite(vv, moved_pixel_img)
                        # cv2.imwrite(tt, re_move_img)
                        # iii = cv2.imread(dir_name + '/' + str(flag_path) + str(k) + '.png')
                        # if cropImg.shape[0] != own_slide or cropImg.shape[1] != own_slide:
                        #     print(ss)
                        #     os.remove(ss)
                        # if moved_pixel_img.shape[0] != own_slide or moved_pixel_img.shape[1] != own_slide:
                        #     print(vv)
                        #     os.remove(vv)
                        if cropImg.shape[0] != own_slide or cropImg.shape[1] != own_slide:
                            print(ss)
                            os.remove(ss)
                    else:
                        print('is not equal')
                else:
                    print('image scale is not ')
                # print(x1, x2)
                # print(y1, y2)
                # HSV = cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV)
                # print(HSV)
                # purple = yellow = red = 0
                # try:
                #     row_num, col_num = HSV.shape[:2]
                # except AttributeError:
                #     continue
                # # card_img_count = row_num * col_num
                # for i in range(row_num):
                #     for j in range(col_num):
                #         H = HSV.item(i, j, 0)
                #         S = HSV.item(i, j, 1)
                #         V = HSV.item(i, j, 2)
                #         if 0 < H <= 10 and S > 43 and V > 46:  # 图片分辨率调整
                #             red += 1
                #         elif 26 < H <= 34 and S > 43 and V > 46:  # 图片分辨率调整
                #             yellow += 1
                #         elif 125 < H <= 155 and S > 43 and V > 46:  # 图片分辨率调整
                #             purple += 1
                # print(purple,yellow,red)
                # dir_name = sum_dir + 'other_cells'

                # print(x1, x2)
                # print(y1, y2)
                # print(len(c))
                # dir_name = sum_dir + 'other_cells'
                # for s in range(len(c)):
                #     biao_x, biao_y, biao_w, biao_h = cv2.boundingRect(c[s])
                #     # slide = min(w,h)
                #     M = cv2.moments(c[s])
                #     try:
                #         biao_cx = int(M['m10'] / M['m00'])
                #         biao_cy = int(M['m01'] / M['m00'])
                #     except ZeroDivisionError:
                #         biao_cx = round(biao_x + biao_w / 2)
                #         biao_cy = round(biao_y + biao_h / 2)
                #     # print(biao_cx, biao_cy)
                #     if (cent_x < math.ceil(max_slide / 2)) or (cent_x < math.ceil(max_slide / 2)) or (
                #             cent_x > (I_raw.shape[1] - math.ceil(max_slide / 2))) or (
                #             cent_y > (I_raw.shape[0] - math.ceil(max_slide / 2))):
                #         skip_num = skip_num + 1
                #         continue
                #
                #     if (x1 < biao_cx < x2) and (y1 < biao_cy < y2):
                #         dir_name = sum_dir + 'hepatocyte'
                #
                #         cropImg = I_raw[y1:y2, x1:x2]
                #         # print(cropImg)
                #         q = q + 1
                #         # print(x1, x2)
                #         # print(y1, y2)
                #         if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                #             if not os.path.exists(dir_name):
                #                 os.makedirs(dir_name)
                #             print(dir_name + '/' + str(flag_path) + str(q) + '.png')
                #             cv2.imwrite(dir_name + '/' + str(flag_path) + str(k) + '.png', cropImg)
                # x1 = cent_x - math.ceil(max_slide / 2)
                # y1 = cent_y - math.ceil(max_slide / 2)
                # x2 = cent_x + math.ceil(max_slide / 2)
                # y2 = cent_y + math.ceil(max_slide / 2)

                #     # else:
                #     #     cropImg = I_raw[y1:y2, x1:x2]
                #     #     # print(cropImg)
                #     #     q = q + 1
                #     #     print(x1, x2)
                #     #     print(y1, y2)
                #     #     if not os.path.exists(dir_name):
                #     #         os.makedirs(dir_name)
                #     #     print(dir_name + '/' + str(flag_path) + str(q) + '.png')
                #     #     cv2.imwrite(dir_name + '/' + str(flag_path) + str(q) + '.png', cropImg)
                #
                # #                cv2.circle(I_raw, (cx, cy), 1, (255,0,0), -1)
                # cv2.drawContours(biaozhu_raw, mask_sum[k][0], -1, colors[3], 1)
                # cv2.imwrite('/cptjack/totem/barrylee/temp/'+str(i)+'.png', biaozhu_raw)
                # np.save(our_mask_np,mask_sum[k][0])
        #                if f:
        #         cv2.drawContours(I_raw, mask_sum[k][0], -1, (255,255), 1)
        # #        cv2.imwrite('/cptjack/totem/youguangyin/red/'+img_num[i].split('.')[0]+'.png', I_raw)
        # cv2.imwrite(out_path, I_raw)
    #        cv2.imwrite('/cptjack/totem/youguangyin/stain_result/'+'123'+'.png', I_raw)
    print(i)


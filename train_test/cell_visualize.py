#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 07:29:57 2019

@author: root
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
from keras.models import load_model
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
Xception_model=load_model('/cptjack/totem/barrylee/codes/find-7003-54/xception-all-tri.h5')
def takeSecond(elem):
    return elem[1]

def remove_discrete_point(matrix, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    matrix = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel)
    return matrix

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
img_dir = '/cptjack/totem/barrylee/NASH-ndpi/test/'
biaozhu = '/cptjack/totem/barrylee/NASH-ndpi/new-annoatation/'
mask_dir = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/output/mask_dir/'
img_num = os.listdir(img_dir)
random.shuffle(img_num)
print(img_num)
colors = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 0), (128, 42, 42)]
# out_mask = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/mask_out/inflammation/'
own_slide = 54
min_slide = 53
contours_num = 0
skip_num = 0
slide = 27
q=0
for i in range(len(img_num)):
    # out_path = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/real/ballooning/' + img_num[i].split('.')[0] + '.png'
    # our_mask_np = '/cptjack/totem/barrylee/cell_classification_codes/mask-rcnn/output/inflammation/'+ img_num[i].split('.')[0] + '.npy'
    biaozhu_img = biaozhu+ img_num[i].split('.')[0] + '-marking'+ '.png'
    picture_name1 = img_num[i].split('.')[0][:-2]
    picture_name2 = img_num[i].split('.')[0][:-3]
    print(picture_name1)
    print(picture_name2)
    sum_dir = '/cptjack/totem/barrylee/cut_small_cell/visua-98'
    I_raw = cv2.imread(img_dir + img_num[i])
    img_2 = I_raw.copy()
    biaozhu_raw = cv2.imread(biaozhu_img)
    print(biaozhu_img)
    flag_path = img_num[i].split('.')[0]
    I = cv2.cvtColor(I_raw, cv2.COLOR_BGR2RGB)
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    stain_matrix_source = extractor.get_stain_matrix(I)  # ,regularizer=0.8)
    source_concentrations = get_concentrations(I, stain_matrix_source)  # , regularizer=0.1)
    b = source_concentrations.reshape((I.shape[0], I.shape[1], 2)).astype(np.uint8)
    with open(
            mask_dir + img_num[i].split('.')[
                0] + '.npy',
            'rb') as fr:
        print(
            mask_dir + img_num[i].split('.')[
                0] + '.npy')
        mask = np.load(fr)
    mask_sum = []
    cnt_list = []
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
        bound_point = 215
        bound_line = 0.75
        bound_median = 195
        f = False
        # print(matrix[matrix > bound_point].shape[0])
        if (matrix[matrix > bound_point].shape[0] / matrix.shape[0]) > bound_line or np.median(
                matrix) > bound_median:
            f = True
        if (cx == 300) or (cy == 300):
            cx, cy = 299, 299
        region_contous_0 = b[cy, cx, 0] if r == 0 else b[cy - r:cy + r, cx - r:cx + r, 0]
        region_contous_1 = b[cy, cx, 1] if r == 0 else b[cy - r:cy + r, cx - r:cx + r, 1]
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
                # if (cent_x < math.ceil(own_slide / 2)) or (cent_y < math.ceil(own_slide / 2)):
                #     continue
                x1 = cent_x - math.ceil(own_slide / 2)
                y1 = cent_y - math.ceil(own_slide / 2)
                x2 = cent_x + math.ceil(own_slide / 2)
                y2 = cent_y + math.ceil(own_slide / 2)
                real_pre = {}
                pixcel_space = 0
                region_r = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 2]
                region_g = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 1]
                region_b = biaozhu_raw[b_cy - pixcel_space:b_cy + pixcel_space + b_h,
                           b_cx - pixcel_space:b_cx + b_w + pixcel_space, 0]
                red = ((region_r == 255) & (region_g == 0) & (region_b == 0))
                yellow = ((region_r == 255) & (region_g == 255) & (region_b == 0))
                blue = ((region_r == 0) & (region_g == 0) & (region_b == 255))
                purple = ((region_r == 255) & (region_g == 0) & (region_b == 255))
                # print(np.sum(red),np.sum(yellow),np.sum(blue),np.sum(purple))
                red, yellow, blue, purple = np.sum(red), np.sum(yellow), np.sum(blue), np.sum(purple)
                label = 0
                if (red > 30 and red >= yellow):
                    label = 0
                elif(purple>30 and purple>yellow):
                    label = 0
                elif (blue > 30 and blue > yellow):
                    label = 0
                elif (yellow > 30 and yellow > red and yellow >blue and yellow > purple):
                    label = 1
                elif (red <= 30 and blue <= 30 and yellow <= 30 and purple <= 30):
                    label = 2
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    w_ = y2 - y1
                    h_ = x2 - x1
                    cropImg = I_raw[y1:y2, x1:x2]
                    if cropImg.shape[0]==54 and cropImg.shape[1]==54:
                        cv2.imwrite('/cptjack/totem/barrylee/NASH-ndpi/22/'+str(flag_path) + str(i)+str(k)+'.png',cropImg)
                        s = cropImg.copy()
                        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
                        real_pre = {'img':cropImg,'label':label}
                        I = cv2.resize(real_pre['img'], (72, 72))
                        I = I/255
                        cropImg_1 = np.expand_dims(I, axis=0)
                        semple = Xception_model.predict(cropImg_1)
                        preds = np.argmax(semple)
                        print('label is %d,pred is %d'%(real_pre['label'],preds))
                        # if preds == 0:
                        #     cv2.circle(img_2, (cent_x, cent_y), 1, (0, 0, 255))
                        # elif preds ==1:
                        #     cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 255))
                        # elif preds ==2:
                        #     cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 0))
                        if real_pre['label'] == 0:

                            if preds==0:
                                print('predicted success')
                            elif preds==1:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 0, 255), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 255),2)
                            elif preds==2:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 0, 255), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 0),2)
                        elif real_pre['label'] == 1:
                            if preds==0:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 255, 255), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 0, 255),2)
                            elif preds==1:
                                print('predicted success')
                            elif preds==2:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 255, 255), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 0),2)
                        elif real_pre['label'] == 2:
                            if preds==0:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 255, 0), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 0, 255),2)
                            elif preds==1:
                                cv2.drawContours(img_2, mask_sum[k][0], -1, (0, 255, 0), 1)
                                cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 255),2)
                            elif preds==2:
                                print('predicted success')
                        # if preds==2 and real_pre['label']==0:
                        #     print('dddddd')
                            # cv2.imwrite('/cptjack/totem/barrylee/cut_small_cell/1-3' + '/' + str(flag_path) + str(k) + '.png', s)
                            # iii = cv2.imread('/cptjack/totem/barrylee/cut_small_cell/1-3' + '/' + str(flag_path) + str(k) + '.png')
                            # print(iii.shape)
                            # if iii.shape[0] != own_slide or iii.shape[1] != own_slide:
                            #     print('/cptjack/totem/barrylee/cut_small_cell/1-3' + '/' + str(flag_path) + str(k) + '.png')
                            #     os.remove('/cptjack/totem/barrylee/cut_small_cell/1-3' + '/' + str(flag_path) + str(k) + '.png')
                            # cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 0))
                            # if w_ == h_ ==48:
                            #     cv2.rectangle(img_2, (x1,y1),(w_,h_),(0,0,255),1)
                        # elif preds==2 and real_pre['label']==1:
                        #     print('dddddd')
                            # cv2.imwrite(
                            #     '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(k) + '.png', s)
                            # iii = cv2.imread(
                            #     '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(k) + '.png')
                            # if iii.shape[0] != own_slide or iii.shape[1] != own_slide:
                            #     print(
                            #         '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(k) + '.png')
                            #     os.remove(
                            #         '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(k) + '.png')
                            # cv2.circle(img_2, (cent_x, cent_y), 1, (0,255,255))
                            # if w_ == h_ == 48:
                            #     cv2.rectangle(img_2, (x1,y1),(w_,h_), (0, 255, 255), 1)
                        # elif preds==2 and real_pre['label']==0:
                        #     cv2.imwrite(
                        #         '/cptjack/totem/barrylee/cut_small_cell/-3' + '/' + str(flag_path) + str(k) + '.png',
                        #         s)
                        #     iii = cv2.imread(
                        #         '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(k) + '.png')
                        #     if iii.shape[0] != own_slide or iii.shape[1] != own_slide:
                        #         print(
                        #             '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(
                        #                 k) + '.png')
                        #         os.remove(
                        #             '/cptjack/totem/barrylee/cut_small_cell/2-3' + '/' + str(flag_path) + str(
                        #                 k) + '.png')
                        #     cv2.circle(img_2, (cent_x, cent_y), 1, (0, 255, 0))
                            # cv2.rectangle(img_2, (x1,y1),(w_,h_), (0, 255, 0), 1)
                        # print(preds)
                    else:
                        print('is not equal')
                else:
                    print('image scale is not ')
    if not os.path.exists(sum_dir):
        os.makedirs(sum_dir)
    print(sum_dir + '/' + str(flag_path) + '.png')
    cv2.imwrite(sum_dir + '/' + str(flag_path) + '.png', img_2)
    print(i)


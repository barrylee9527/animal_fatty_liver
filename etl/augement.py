# -*- coding: utf-8 -*-
"""
get the transformation image of 40x image(2048x1536):
                                        rotate it 90,180, 270 degrees.
                                        flip it up and down
                                        flip it left and right
increase one image to 6 images.
"""
# import build_dataset3 as b
from PIL import Image
import sys
import os
import random
sys.path.append('/cptjack/totem/barrylee/Image_Augementation-master')
from imutils import paths
from Image_Augementation import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
# 数据增强的代码
def get_transform(bach_data_0, save_dir):
    path = list(os.listdir(bach_data_0))
    #sess = tf.InteractiveSession()
    for file,i in zip(path,range(len(path))):
        # img = Image.open(bach_data_0 + '/'+file)
        image = cv2.imread(bach_data_0 + '/'+file)
        # print(image.shape)
        if image.shape[0]!=96 or image.shape[1]!=96:
            print(file, i)
        # image_copy = image.copy()
        # hsv = cv2.cvtColor(image_copy.copy(), cv2.COLOR_BGR2HSV)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB
        # 高斯模糊
        # gaussian = cv2.GaussianBlur(image,(3,3),0)
        # H, S, V = cv2.split(hsv)
        # H, S, V = (H - H * 0.1).astype(np.uint8), (S + S * 0.1).astype(np.uint8), (V).astype(np.uint8)
        # if H.any() > 180:
        #     H = 180
        # if S.any() > 255:
        #     S = 255
        # new_hsv = cv2.merge((H, S, V))
        # NEW_ = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
        # cv2.imwrite(save_dir + '/' + file.split('.')[0]+ str(i) + '_hs.png', NEW_)
        # 亮度调节
        # random_brightness = tf.image.adjust_brightness(image.copy(), delta=0.1)
        # random_brightness = cv2.cvtColor(random_brightness.eval(), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_dir + '/' + file.split('.')[0]+ str(i) + '_gaussian.png', gaussian)
        # 色度调节
        # random_hue = tf.image.adjust_hue(image.copy(), delta=-0.08)
        # random_hue = cv2.cvtColor(random_hue.eval(), cv2.COLOR_BGR2RGB)
        # cv2.imwrite('/cptjack/totem/barrylee/temp/114.png', random_hue)
        #cv2.imwrite(save_dir + '/' + file.split('.')[0]+ str(i) + '_adj_contrast.png', random_contrast)
        # random_hue = tf.image.adjust_hue(image.copy(), delta=-0.1)
        # random_hue = cv2.cvtColor(random_hue.eval(), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_dir + '/' + file.split('.')[0]+ str(i) + '_adj_hue.png', random_hue)
        # 调节对比度
        # random_satu = tf.image.adjust_saturation(image.copy(), saturation_factor=1.6)
        # random_satu = cv2.cvtColor(random_satu.eval(), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_dir + '/' + file.split('.')[0]+ str(i) + '_adj_saturation.png', random_satu)
        # ng2 = img.transpose(Image.FLIP_TOP_BOTTOM)
        # ng2.save(save_dir + '/' + file.split('.')[0]+ str(i) + '_ud.png')
        # ng3 = img.transpose(Image.FLIP_LEFT_RIGHT)
        # ng3.save(save_dir + '/'  + file.split('.')[0] + str(i) + '_lr.png')
        # ng4 = img.transpose(Image.ROTATE_90)
        # ng4.save(save_dir + '/'  + file.split('.')[0] + str(i) + '_90.png')
        # ng5 = img.transpose(Image.ROTATE_180)
        # ng5.save(save_dir + '/' + file.split('.')[0] + str(i) + '_180.png')
        # ng6 = cv2.GaussianBlur(img1,(5,5),1.2)
        # cv2.imwrite(save_dir + '/' + file.split('.')[0] + str(i) + '_gaussblur.png',ng6)
    return
bach_data_0 = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/96-val'
save_dir0 = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/333'
if not os.path.exists(save_dir0):
    os.makedirs(save_dir0)
for k in os.listdir(bach_data_0):
    bach_data = bach_data_0 + '/' + k
    save_dir = save_dir0 + '/'+k
    print(bach_data)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    get_transform(bach_data, save_dir)




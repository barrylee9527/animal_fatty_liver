#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 05:58:37 2019

@author: root
"""

import pandas as pd
import random
import os 
import shutil

# path = '/cptjack/totem/barrylee/cut_small_cell/immune_cells'
first_path = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/22'
train_path = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/96-train/'
val_path = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/96-val/'
# if not os.path.exists(first_path):
#     os.mkdir(first_path)
# if not os.path.exists(train_path):
#     os.mkdir(train_path)
# if not os.path.exists(val_path):
#     os.mkdir(val_path)
val_rate = 0.2
# filelist=os.listdir(path)
# total_num = len(filelist)
# random.shuffle(filelist)
# val_num = int(total_num*val_rate)
# train_num = total_num - val_num
# for i in range(train_num):
#     img_name = filelist[i]
#     shutil.copy(path+'/'+img_name,train_path+img_name)
#     print('%d is ok'% i)
# for j in range(val_num):
#     img_name = filelist[j+train_num]
#     shutil.copy(path+'/'+img_name,val_path+img_name)
#     print('%d is ok'% j)
for i in os.listdir(first_path):
    filelist = os.listdir(first_path +'/'+i)
    total_num = len(filelist)
    random.shuffle(filelist)
    val_num = int(total_num * val_rate)
    train_num = total_num - val_num
    if not os.path.exists(train_path+ i):
        os.makedirs(train_path + i)
    for j in range(train_num):
        img_name = filelist[j]
        shutil.copy(first_path + '/' + i + '/'+ img_name, train_path + i + '/'+img_name)
        print('%d is ok' % j)
    if not os.path.exists(val_path+ i):
        os.makedirs(val_path + i)
    for k in range(val_num):
        img_name = filelist[k + train_num]
        shutil.copy(first_path + '/' + i + '/'+ img_name,val_path + i + '/'+img_name)
        print('%d is ok'% k)


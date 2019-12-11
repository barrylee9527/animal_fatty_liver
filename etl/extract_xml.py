import os
import matplotlib.pyplot as plt
import sys
import shutil
import lxml.etree as ET
import openslide as opsl
from PIL import ImageDraw
import cv2

import numpy as np
xml_path = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/xml_file/#58.xml'
from item_utils.xml_utils import xml_to_region
region_list = xml_to_region(xml_path)
region_final = []
print(len(region_list))
ndpi_file = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/111/#58.ndpi'
ndpi_img = opsl.OpenSlide(ndpi_file)
for i in region_list:
    x_y_cod = []
    contour = []
    for j in i:
        x_y_cod.append((int(j['X']),int(j['Y'])))
    region_final.append(x_y_cod)
    print(x_y_cod)
    x_max = int(max(x_y_cod, key=lambda point: point[0])[0])
    x_min = int(min(x_y_cod, key=lambda point: point[0])[0])
    y_max = int(max(x_y_cod, key=lambda point: point[1])[1])
    y_min = int(min(x_y_cod, key=lambda point: point[1])[1])
    print(x_max,x_min,y_max,y_min)
    h = y_max - y_min
    w = x_max - x_min
    for s in i:
        contour.append((int(s['X'])-x_min+80,int(s['Y'])-y_min+80))
    mask_pre = np.full((y_max-y_min+160,x_max-x_min+160),0,dtype=np.uint8)
    x = np.array(ndpi_img.read_region((x_min-80,y_min-80),0,(x_max-x_min+160,y_max-y_min+160)))[:,:,:3]
    raw_img = x.copy()
    mask_pre = cv2.fillPoly(mask_pre,[np.array(contour)],255)
    print(mask_pre)
    #_, cxx, hx = cv2.findContours(mask_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x[mask_pre==0]=0
    plt.imsave('/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/renew_img/mask/#58-' + str(w) + '.png',x)
    plt.imsave('/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/renew_img/raw/#58-'+str(w)+'.png',raw_img)
    plt.imsave('/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/renew_img/mask_contours/#58-'+str(w) + '.png',mask_pre)
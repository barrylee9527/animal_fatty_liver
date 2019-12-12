#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 07:30:44 2018

@author: Bohrium Kwong
"""
import numpy as np
import cv2

def region_view(coordinate,im_input,nuclei_info,c):
    im_result = cv2.cvtColor(im_input,cv2.COLOR_RGB2BGR)
    nuclei_info[:,0] = nuclei_info[:,0] - coordinate[0]
    nuclei_info[:,1] = nuclei_info[:,1] - coordinate[1]
    nuclei_label = np.argmax(nuclei_info[:,4:],axis=1)
    for l in range(len(nuclei_info)):
        draw_flag = 0
        nuclei_flag = nuclei_info[l,2] > 10 and  nuclei_info[l,3] <0.1 
        if (nuclei_label[l] == 0 and nuclei_flag ==1) or nuclei_info[l,2] > 350: #epithelial : cancel
            draw_flag = 1
            color_flag = (36,28,237)
        elif nuclei_label[l] == 1 and nuclei_flag ==1 and nuclei_info[l,2] <= 350:#fibroblast
            draw_flag = 1
            color_flag = (0,215,255)
        elif nuclei_label[l] == 2 and nuclei_flag ==1 and nuclei_info[l,2] <= 350:#inflammatory
            draw_flag = 1
            color_flag = (204,72,63)
        elif nuclei_label[l] == 3 and nuclei_flag ==1:#miscellaneous
            draw_flag = 1
            color_flag = (76,177,34)  
        
        if draw_flag == 1:
            im_result = cv2.rectangle(im_result, (int(nuclei_info[l,0]),int(nuclei_info[l,1])),
                                      (int(nuclei_info[l,0]+1),int(nuclei_info[l,1]+1)),color_flag,3)
            
    return cv2.cvtColor(im_result,cv2.COLOR_BGR2RGB)



    
def nuclei_statistics(nuclei_info):
    """
    nuclei_statistics of each input nuclei_info array
    :return:the the sum of all cells, cancel_cell_count,fibroblast_cell_count,inflammatory_cell_count,miscellaneous_cell_count
    """
    nuclei_label = np.argmax(nuclei_info[:,4:],axis=1)
    nuclei_flag = (nuclei_info[:,2] > 10) & (nuclei_info[:,3] <0.1)
    cancel_cell_count = np.sum(((nuclei_label == 0) | (nuclei_info[:,2]  > 350)) & (nuclei_info[:,2] > 10) & (nuclei_info[:,3] <0.1))
#    cancel_positive_count = np.sum(((nuclei_info[:,5]  > class_proba[0]) | (nuclei_info[:,3]  > 350)) 
#                                   & (nuclei_info[:,2] > positive_rate_threshold[0]) & (nuclei_info[:,3] > 10) & (nuclei_info[:,4] <0.1))
    

    fibroblast_cell_count = np.sum((nuclei_label == 1 ) & (nuclei_info[:,2] > 10) & (nuclei_info[:,3] <0.1) & (nuclei_info[:,2]  <= 350))
#    lymphocyte_positive_count = np.sum((nuclei_info[:,6]  > class_proba[1]) & (nuclei_info[:,2] > positive_rate_threshold[1])
#                                   & (nuclei_info[:,3] > 10) & (nuclei_info[:,4] <0.1) & (nuclei_info[:,3]  < 350))    
 
    inflammatory_cell_count = np.sum((nuclei_label == 2 ) & (nuclei_info[:,2] > 10) & (nuclei_info[:,3] <0.1) & (nuclei_info[:,2]  <= 350))
    
    miscellaneous_cell_count = np.sum((nuclei_label == 3 ) & (nuclei_info[:,2] > 10) & (nuclei_info[:,3] <0.1) & (nuclei_info[:,2]  <= 350))
    return np.sum(nuclei_flag),cancel_cell_count,fibroblast_cell_count,inflammatory_cell_count,miscellaneous_cell_count

#positive_rate_threshold = [0.05,0.1]
#class_proba = [0.6,0.85]
#cell_sum,cancel_cell_count,cancel_positive_count,lymphocyte_cell_count,lymphocyte_positive_count = post_treatment.nuclei_statistics(result_list[0],positive_rate_threshold,class_proba)


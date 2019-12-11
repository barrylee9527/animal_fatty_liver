# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide as opsl
import sys
import os
patch_c = 800
patch_r = 800
step = 800
class Openslide1:
    def __init__(self,patch_c,patch_r,step):
        self.patch_c = patch_c
        self.patch_r = patch_r
        self.step = step
    def chul(self,slide,flag):
        # print(slide.level_dimensions[0])
        w_count = int(slide.level_dimensions[0][0]//self.step)
        h_count = int(slide.level_dimensions[0][1]//self.step)
        count = 0    
        for x in range(1,int(w_count)-1):
            for y in range(int(h_count)):
                print(x,y)             
                if count<=50:
                    print(count)
                    out_path = '/cptjack/totem/barrylee/cut_small_cell/all-pre/' + flag.split('.')[0] +'-'+ str(x) +str(y) +'.png'
                    slide_region = np.array(slide.read_region((x*self.step,y*self.step),0,(self.patch_c,self.patch_r)))[:,:,0:3]
                    if(np.mean(slide_region)<200):
                        plt.imsave(out_path,slide_region)
                        count = count + 1
                else:
                    continue
def main():        
    for file in os.listdir(r'/cptjack/totem/barrylee/NASH-ndpi/NASH-test'):
        filename = '/cptjack/totem/barrylee/NASH-ndpi/NASH-test/' + file
        print(file)
#        if(filename!='/cptjack/totem/Liver0729/LIV005001-CDA Liver-G2-4 VF-4.ndpi'):
        flag = file
        slide = opsl.OpenSlide(filename)
        s = Openslide1(patch_c, patch_r, step)
        s.chul(slide, flag)
        slide.close()
if __name__ =='__main__':
    main()

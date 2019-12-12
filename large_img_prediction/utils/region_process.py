# @Time    : 2019.08.28
# @Author  : Bohrium Kwong
# @Licence : bio-totem

import os
import gc
from PIL import Image
import cv2
import numpy as np
#from keras import optimizers
#from keras.models import load_model
import configparser
import sys
sys.path.append('../')
#from utils.log_utils import get_logger
#from utils.openslide_utils import Slide
from utils.tissue_utils import get_tissue
from skimage import exposure
from cell_classifition.cell_classifition import cancer_cell_caculating
#from cell_classifition.post_treatment import nuclei_statistics
from output_process.process_script import matrix_resize

current_path = os.path.dirname(__file__)
LOGS_DIR = os.path.join(current_path, "..", "logs")
conf = configparser.ConfigParser()
conf.read(os.path.join(current_path, "..", "sys.ini"))
GPU_USE = conf.get("DEFAULT", "GPU_USE")
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(GPU_USE))
Image.MAX_IMAGE_PIXELS=500000000

def maskrcnn_detection(model, image):
    """
    Mask-RCNN model detect image
    :param model: trained model
    :param image: the image wanted to be detected
    :return: Mask-RCNN detection mask
    """

    results = model.detect(image, verbose=1)
    mask_allsingle_list = []
    for result in results:
        mask_allsingle = result['masks'].copy()
        mask_allsingle_list.append(mask_allsingle)

    return mask_allsingle_list

def rgb_similarity(im_arr, similarity, threshold):
    """
    calculate image RGB channels similarity
    :param im_arr: image numpy.array
    :param similarity: image rgb three channels similarity
    :param threshold: the threshold that rgb three channels value need to be set 250
    :return:
    """

    r_channel = im_arr[:, :, 0].astype(np.int16)
    g_channel = im_arr[:, :, 1].astype(np.int16)
    b_channel = im_arr[:, :, 2].astype(np.int16)

    thres_0 = r_channel > threshold
    thres_1 = g_channel > threshold
    thres_2 = b_channel > threshold

    sim_0 = abs(r_channel - g_channel) < similarity
    sim_1 = abs(r_channel - b_channel) < similarity
    sim_2 = abs(g_channel - b_channel) < similarity

    temp_0 = np.logical_and(sim_0, sim_1)
    temp_0 = np.logical_and(temp_0, sim_2)
    temp_1 = np.logical_and(thres_0, thres_1)
    temp_1 = np.logical_and(temp_1, thres_2)
    temp = np.logical_and(temp_0, temp_1)

    return np.dstack((temp, temp, temp))

def svs_to_probmat(slide,patch_R, patch_C, seg_model, cell_class_model1,cell_class_model2,patch_size ,predict = True):
    """
    convert svs file to probability matrix
    :param svs: svs file path
    :param patch_R: patch row
    :param patch_C: patch column
    :param seg_model: mask-rcnn segmentation model
    :param class_model: nucleus classification model
    :param patch_size: patch size
    :param predict: let the region_raw_tensor be predicted by the cell classify model or not
    :return:
    """
#    slide = Slide(svs)
    slide_width, slide_height = slide.get_level_dimension(0)
    N = patch_R // patch_C 
    # N should be no more than 2
    W_ps_NI = slide_width // patch_C
    H_ps_NI = slide_height // patch_R
    widen = patch_size // 2 + 2  
    
    level_downsamples = slide.get_level_downsample(level=2)
#    svs_im = slide.read_region((0, 0), level=2, size= slide.get_level_dimension(level=2))
    svs_im = np.asarray(slide.get_thumb())[:,:,:3]
    svs_regin_mask,_ = get_tissue(svs_im,contour_area_threshold=2000)
    cell_model_output_shape = cell_class_model1.output.shape.as_list()[1] + cell_class_model2.output.shape.as_list()[1]
    cell_cls_prediction_info = np.zeros((1,4 + cell_model_output_shape))
    svs_W_H_info = np.zeros((1, 4), dtype=np.uint32)

    # opencv augmentation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(N * 4, 4))

    
    for w in range(W_ps_NI):
        for h in range(H_ps_NI):
            bound_C = 1                             # bound column flag
            bound_R = 1                             # bound row flag
            widen_patch_C = patch_C + widen

            if (w+1) * patch_C + widen > slide_width:
                widen_patch_C = patch_C
                bound_C = 0
            if (h+1) * patch_R + widen > slide_height:
                bound_R = 0
                
            cc_widen_subHIC_list = []
            ul_region_point = []
            bound_list = [(bound_C, bound_R)] * N
            region_raw_tensor = np.zeros((1,patch_C,patch_C,3))
            region_flag = 0
            for g in range(N):
                bottom = int((h * patch_R + g * patch_C) / level_downsamples)
                top = bottom + int(patch_C / level_downsamples) -1
                left = int(w * patch_C / level_downsamples)
                right = left + int(patch_C / level_downsamples) -1
                if np.sum(svs_regin_mask[bottom : top,left : right ] > 0) > 0.75 * (patch_C / level_downsamples)**2:                       
                    widen_subHIC = np.array(slide.read_region((w * patch_C, h * patch_R + g * patch_C), 0, (widen_patch_C, widen_patch_C)))
                    rgb_s = (abs(widen_subHIC[:,:,0] -122) >= 93) & (abs(widen_subHIC[:,:,1] -122) >= 93) & (abs(widen_subHIC[:,:,2] -122) >= 93)
                    if np.sum(rgb_s)<= widen_patch_C**2 * 0.85:
                    # 当且仅当白色/黑色像素点加起来不超过read_region截图范围的85%时，才进行后续处理
                        region_point = (w * patch_C, h * patch_R + g * patch_C)
                        ul_region_point.append(region_point)
                        cc_widen_subHIC_list.append(widen_subHIC[:, :, :3].copy())
                        
                        widen_subHIC = np.where(rgb_similarity(widen_subHIC[:, :, :3], 15, 195), 250, widen_subHIC[:, :, :3])
                    # adjust equalization histogram and adjust brightness
                        for k in range(widen_subHIC.shape[2]):
                            widen_subHIC[:, :, k] = clahe.apply(widen_subHIC[:, :, k])
                        widen_subHIC = exposure.adjust_gamma(widen_subHIC, gamma=1.5)
                        # 尝试进行直方图均衡化以及RGB调整以使得图像增强，提高细胞核的分割召回
                        if region_flag == 0 :
                            region_raw_tensor[0,:,:,:] = widen_subHIC[:patch_C, :patch_C, :3]
                            region_flag = region_flag + 1
                        else:
                            region_raw_tensor = np.row_stack((region_raw_tensor,
                                                                  np.expand_dims(widen_subHIC[:patch_C, :patch_C, :3], axis=0)))
            
            if len(ul_region_point) > 0:
                # ul_region_point长度大于0代表有真正需要送到Mask-Rcnn进行细胞核分割的图像
                image_tensor_len = region_raw_tensor.shape[0]
                input_data = region_raw_tensor.reshape(image_tensor_len*patch_C,patch_C,3) 
                imagesMask = maskrcnn_detection(seg_model, [input_data])
                imagesMask = imagesMask[0]
                    
                imagesMask_list = [imagesMask[l * patch_C : patch_C * (l + 1) , :patch_C, :] for l in range(len(ul_region_point))]
                
                image_result_list, is_success_flag_list = cancer_cell_caculating((patch_C,patch_C),
                                            cc_widen_subHIC_list, imagesMask_list,
                                            ul_region_point, bound_list, cell_class_model1,cell_class_model2,
                                            patch_size,predict)
                
                del imagesMask, widen_subHIC,imagesMask_list,cc_widen_subHIC_list
                gc.collect()
                    
                valit_flag = 0
                
                for each in range(N):   
                    if each - valit_flag < len(image_result_list):  
                        if 1 == is_success_flag_list[each - valit_flag] and each - valit_flag >=0:
                            # storage cell classification prediction information
                            ul_w = w * patch_C
                            ul_h = h * patch_R + each * patch_C
                            temp_svs_W_H_info = np.asarray([[ul_w,
                                                             ul_h,
                                                             (cell_cls_prediction_info.shape[0] - 1),
                                                             (cell_cls_prediction_info.shape[0] - 1 + image_result_list[each - valit_flag].shape[0])]])
                            svs_W_H_info = np.row_stack((svs_W_H_info, temp_svs_W_H_info))
#                            print(image_result_list[each - valit_flag].shape)
                            cell_cls_prediction_info = np.row_stack((cell_cls_prediction_info, image_result_list[each - valit_flag]))
                        else:
                            valit_flag = valit_flag + 1    
                            
            # save cell classification prediction information
    cell_cls_prediction_info = np.delete(cell_cls_prediction_info, 0, axis=0)
    svs_W_H_info = np.delete(svs_W_H_info, 0, axis=0)

    gc.collect()
    return cell_cls_prediction_info, svs_W_H_info

from utils.opencv_utils import OpenCV                    
def cell_predict_add(slide,nuclei_info,class_model,patch_R,patch_C,patch_size):
    """
    :param class_model: nucleus classification model
    :param patch_size: patch size
    """
      
    slide_width, slide_height = slide.get_level_dimension(0)
    model_input_shape = class_model.input.shape.as_list()[1]
#    N = patch_R // patch_C 
    W_ps_NI = slide_width // patch_C
    H_ps_NI = slide_height // patch_R
    
    
#    for index in range(svs_W_H_info.shape[0]):
#        widen = patch_size // 2 + 2
#        
#        w_coordinate = svs_W_H_info[index,0]
#        h_coordinate = svs_W_H_info[index,1]
#        nuclei_info_select = nuclei_info[svs_W_H_info[index,2]:svs_W_H_info[index,3],:].copy()
#        nuclei_info_select[:,0] = nuclei_info_select[:,0] - w_coordinate
#        nuclei_info_select[:,1] = nuclei_info_select[:,1] - h_coordinate
#        widen_patch_C = patch_C + widen
#        if (w_coordinate + patch_C + widen >= slide_width) or (h_coordinate + patch_C + widen >= slide_height):
#            widen_patch_C = patch_C
#
#        widen_subHIC = np.array(slide.read_region((w_coordinate, h_coordinate), 0, (widen_patch_C, widen_patch_C)))[:,:,:3]
#        
#        cell_sum = 0
#        for i in range(nuclei_info_select.shape[0]):
#            
#            x1 = max(patch_size/2,nuclei_info_select[i,1] - patch_size/2)
#            y1 = max(patch_size/2,nuclei_info_select[i,0] - patch_size/2)
#            region = widen_subHIC[int(x1 - patch_size/2):int(x1 - patch_size/2) + patch_size,
#                                                     int(y1 - patch_size/2):int(y1 - patch_size/2) + patch_size,:]
#            region = OpenCV(region).resize(model_input_shape,model_input_shape)
#            if cell_sum ==0:
#                region_raw_tensor = np.expand_dims(region/255, axis=0)
#                cell_sum = cell_sum + 1
#            else:
#                region_raw_tensor = np.row_stack((region_raw_tensor,np.expand_dims(region/255, axis=0)))
#        
#        preds_c_l = class_model.predict_proba(region_raw_tensor)
#        nuclei_info[svs_W_H_info[index,2]:svs_W_H_info[index,3],:] = preds_c_l

        
    for w in range(W_ps_NI):
        for h in range(H_ps_NI):
            widen = patch_size // 2 + 2
            w_coordinate = w * patch_C
            h_coordinate = h * patch_R
            if (w_coordinate + patch_C + widen >= slide_width) or (h_coordinate + patch_R + widen >= slide_height):
                widen = 0
    
            widen_subHIC = np.array(slide.read_region((w_coordinate, h_coordinate), 0, (patch_C + widen, patch_R + widen)))[:,:,:3]
            select_flag = (nuclei_info[:,0]>=w_coordinate) & (nuclei_info[:,1]>=h_coordinate)\
                                             & (nuclei_info[:,0] < w_coordinate + patch_C -1)\
                                             & (nuclei_info[:,1] < h_coordinate + patch_R -1)
            nuclei_info_select = nuclei_info[select_flag].copy()
            nuclei_info_select[:,0] = nuclei_info_select[:,0] - w_coordinate
            nuclei_info_select[:,1] = nuclei_info_select[:,1] - h_coordinate  
            if nuclei_info_select.shape[0] > 0:
                cell_sum = 0
                for i in range(nuclei_info_select.shape[0]):
                    y = max(patch_size/2,nuclei_info_select[i,1] - patch_size/2)
                    x = max(patch_size/2,nuclei_info_select[i,0] - patch_size/2)
                    region = widen_subHIC[int(y - patch_size/2):int(y - patch_size/2) + patch_size,
                                                             int(x - patch_size/2):int(x - patch_size/2) + patch_size,:]
                    region = OpenCV(region).resize(model_input_shape,model_input_shape)
                    if cell_sum ==0:
                        region_raw_tensor = np.expand_dims(region/255, axis=0)
                        cell_sum = cell_sum + 1
                    else:
                        region_raw_tensor = np.row_stack((region_raw_tensor,np.expand_dims(region/255, axis=0)))
                
                preds_c_l = class_model.predict_proba(region_raw_tensor)
                nuclei_info[select_flag,4:] = preds_c_l       
        
        
        
        
    return nuclei_info
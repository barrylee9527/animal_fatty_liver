# @Time    : 2019.01.28
# @Author  : Bohrium Kwong
# @Licence : bio-totem

import os
import gc
from PIL import Image
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
import configparser
import sys
import cv2
sys.path.append('../')
from utils.log_utils import get_logger
from utils.openslide_utils import Slide
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.nucleus import nucleus_train
from cell_classifition.cell_classifition import cancer_cell_caculating
from cell_classifition.post_treatment import nuclei_statistics
from skimage import io
import numpy as np

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

def openslide_region_predict(slide,model,datagen,patch_size,y_nd=2):
    """
    region_classification_model predict image
    :svs_file: read by method of openslide.OpenSlide
    :model :trained model
    :datagen : the ImageDataGenerator of above-model,is used to fit image before perdicting
    :patch_size: the value of above-model 's input shape (patch_size,patch_size,3)
    :y_nd : the number of the batch size in the method of model's predict,default 2
    :return: region label result mat
    """  
    slide_width, slide_height = slide.get_level_dimension(0)
    # patch_size = 299
    w_count = int(slide_width // patch_size)
    h_count = int(slide_height // patch_size)
    out_img = np.zeros([h_count,w_count])
    y_nd_list = [3,4,5,7,11]
    for t in range(len(y_nd_list)):
        if h_count%y_nd_list[t]==0:
            y_nd = y_nd_list[t]
            break 
        elif h_count%y_nd_list[t]==1:
            if y_nd < y_nd_list[t]:
                y_nd = y_nd_list[t]
    #The final value of y_nd will be caculated by above-process:max(y_nd){h_count%y_nd<=1}          
    for x in range (1,w_count - 1):
        for y in range (int(h_count//y_nd)):
            slide_region = np.array(slide.read_region((x * patch_size , y * patch_size * y_nd) , 0 , (patch_size ,patch_size * y_nd)))
            slide_img = slide_region[:,:,:3]
            rgb_s = (abs(slide_img[:,:,0] -107) >= 93) & (abs(slide_img[:,:,1] -107) >= 93) & (abs(slide_img[:,:,2] -107) >= 93)
            if np.sum(rgb_s)<=(patch_size * patch_size * y_nd) * 0.4:
                slide_img2 = slide_img.reshape(y_nd,patch_size,patch_size,3)
                datagen.fit(slide_img2)
                prob = model.predict_proba(slide_img2,batch_size=y_nd)
                for g in range(y_nd):
                    if prob[g][0] > 0.5:
                        out_img[y * y_nd + g,x] = 1   
                        
            del slide_region,slide_img,rgb_s
            gc.collect()                                         
    return out_img



def svs_region_to_probmat(slide, region_result,patch_R, patch_C, seg_model, class_model,patch_size):
    """
    region_classification_model predict image
    :svs_file: read by method of openslide.OpenSlide
    :param patch_R: patch row
    :param patch_C: patch column
    :param gpu_count: the nums of parallel GPUS that you want to use,each GPU could only afford an patch_R * patch_C image 
    :param seg_model: mask-rcnn segmentation model
    :param class_model: nucleus classification model
    :param patch_size: patch size
    :return:cell classification prediction information
    """
    slide_width, slide_height = slide.get_level_dimension(0)
    N = patch_R // patch_C 
    # N should be no more than 2
    W_ps_NI = slide_width // patch_C
    H_ps_NI = slide_height // patch_R
    widen = patch_size // 2 + 2           # widen bounding
    # Change the value of widen from original（patch_size // 2） to （patch_size // 2 + 2）
    # In order to reduce bugs when calls the method of cancer_cell_caculating
    # by Bohrium Kwong 2019.01.21
    CancerProb_arr = np.zeros((slide_height, slide_width), dtype=np.float16)

    cell_cls_prediction_info = np.zeros((1,8))
    svs_W_H_info = np.zeros((1, 4), dtype=np.uint32)
    for w in range(W_ps_NI):
        for h in range(H_ps_NI):
            bound_C = 1                             # bound column flag
            bound_R = 1                             # bound row flag
            widen_patch_C = patch_C + widen
#            widen_patch_R = patch_R + widen
#            step_patch_R = widen_patch_C
            if (w+1) * patch_C + widen > slide_width:
                widen_patch_C = patch_C
                bound_C = 0
            if (h+1) * patch_R + widen > slide_height:
#                widen_patch_R = patch_R
#                step_patch_R = patch_C
                bound_R = 0
                
            cc_widen_subHIC_list = []
            ul_region_point = []
            bound_list = [(bound_C, bound_R)] * N
            region_raw_tensor = np.zeros((1,patch_C,patch_C,3))
            region_flag = 0
            for g in range(N):
                if region_result[h * N + g,w] > 0:
                    widen_subHIC = np.array(slide.read_region((w * patch_C, h * patch_R + g * patch_C), 0, (widen_patch_C, widen_patch_C)))
                    region_point = (w * patch_C, h * patch_R + g * patch_C)
                    ul_region_point.append(region_point)
                    cc_widen_subHIC_list.append(widen_subHIC[:, :, :3])
                    if region_flag == 0 :
                        region_raw_tensor[0,:,:,:] = widen_subHIC[:patch_C, :patch_C, :3]
                        region_flag = region_flag + 1
                    else:
                        region_raw_tensor = np.row_stack((region_raw_tensor,
                                                              np.expand_dims(widen_subHIC[:patch_C, :patch_C, :3], axis=0)))
            
            if len(ul_region_point) > 0:
                image_tensor_len = region_raw_tensor.shape[0]
                input_data = region_raw_tensor.reshape(image_tensor_len*patch_C,patch_C,3) 
                imagesMask = maskrcnn_detection(seg_model, [input_data])
                imagesMask = imagesMask[0]
                #If there are more than one region to be segmented,reshape it into an iamge(patch_C * N,patch_C),is better than
                # cutting 2 patches(make sure the image's maskrcnn_detection memory cost is under the upper limit of graphic card), 
                # because the processing of keeping one big size image is more efficient than two patches of image in mask rcnn model detection. 
                # base on the value of region_flag(=1 or 2),the size of input image will turn to (region_flag *patch_C,patch_R)
                # the above method could save about 10% detection cost time 
                # by Bohrium Kwong 2019.01.23                
                imagesMask_list = [imagesMask[l * patch_C : patch_C * (l + 1) , :patch_C, :] for l in range(len(ul_region_point))]
                
                image_result_list, is_success_flag_list = cancer_cell_caculating(
                                            patch_C, cc_widen_subHIC_list, imagesMask_list,
                                            ul_region_point, bound_list, class_model,
                                            patch_size)
                
                del imagesMask, widen_subHIC,imagesMask_list,cc_widen_subHIC_list
                gc.collect()
                    
                valit_flag = 0
                
                for each in range(N):
                    if region_result[h * N + each ,w] > 0 and each - valit_flag < len(image_result_list):  
                        if 1 == is_success_flag_list[each - valit_flag] and each - valit_flag >=0:
                            # storage cell classification prediction information
                            ul_w = w * patch_C
                            ul_h = h * patch_R + each * patch_C
                            temp_svs_W_H_info = np.asarray([[ul_w,
                                                             ul_h,
                                                             (cell_cls_prediction_info.shape[0] - 1),
                                                             (cell_cls_prediction_info.shape[0] - 1 + image_result_list[each - valit_flag].shape[0])]])
                            svs_W_H_info = np.row_stack((svs_W_H_info, temp_svs_W_H_info))
                            cell_cls_prediction_info = np.row_stack((cell_cls_prediction_info, image_result_list[each - valit_flag]))
                            # patch cancer probability
                            h_upper = h * patch_R + each * patch_C
                            h_bottom = h * patch_R + (each+1) * patch_C
                            w_left = w * patch_C
                            w_right = (w+1) * patch_C
                            cell_sum,cancel_cell_count,_,_,_  = nuclei_statistics(image_result_list[each - valit_flag])
                            # this method return return cell_sum,cancel_cell_count,fibroblast_cell_count,inflammatory_cell_count,miscellaneous_cell_count
                            try:
                                cancer_prob = cancel_cell_count / cell_sum
                            except Exception as e:
                                error_logger = get_logger(level="error")
                                error_logger.error('y: '+str(h * N + each)+ ' x: '+str(w) + ' something wrong with the value of cell_sum.', exc_info=True)
                                cancer_prob = 0
                            #cancer_prob = np.sum(image_result_list[each][:, -2] > cc_prob_threshold) / len(image_result_list[each])
                            # Change the method of cancer_prob 's caculating from original codes to call the 'nuclei_statistics' method ,
                            # becareful sometimes it will return a division by zero value,that is the reason that I add a try Exception module.
                            # and so on,the parameter cc_prob_threshold is no longer in use,I have modified the involved definition in the methods are related
                            # by Bohrium Kwong 2019.01.21                          
                            CancerProb_arr[h_upper:h_bottom, w_left:w_right] = cancer_prob
                        else:
                            valit_flag = valit_flag + 1    
                            
            # save cell classification prediction information
    cell_cls_prediction_info = np.delete(cell_cls_prediction_info, 0, axis=0)
    svs_W_H_info = np.delete(svs_W_H_info, 0, axis=0)
#    pkl_result = (cell_cls_prediction_info, svs_W_H_info)
#    pkl_thread = threading.Thread(target=cell_cls_prediction_to_pickle, args=(slide.get_basename(), pkl_result,))
#    pkl_thread.start()

    return  CancerProb_arr,cell_cls_prediction_info, svs_W_H_info
                    
                    
def svs_region_to_probmat_save_img_mask(file_name, seg_model, mask_save_dir, file):
    """
    region_classification_model predict image
    :svs_file: read by method of openslide.OpenSlide
    :file_name: the base name of svs_file
    :param patch_R: patch row
    :param patch_C: patch column
    :param gpu_count: the nums of parallel GPUS that you want to use,each GPU could only afford an patch_R * patch_C image 
    :param seg_model: mask-rcnn segmentation model
    :param patch_size: patch size
    :img_save_dir:the path that saves images
    :mask_save_dir:the path that saves the mask of images
    :return:cell classification prediction information
    """
    input_data = cv2.imread(file_name)
    w,h = input_data.shape[:2]
    input_data = cv2.resize(input_data,(w*2,h*2))
    imagesMask = maskrcnn_detection(seg_model, [input_data])
    imagesMask = imagesMask[0]
    r_mask = np.zeros((round(w),round(h),imagesMask.shape[2]),dtype=np.bool)
    for i in range(imagesMask.shape[2]):
        r_mask[:,:,i] = transform.resize(imagesMask[:,:,i],(round(w),
                            round(h)),order=1,mode="constant",preserve_range=True)
    imagesMask = r_mask
    # mask_whole = []
    # for c in range(imagesMask.shape[2]):
    #     tmp_arr = imagesMask[:,:,c]
    #     if c == 0:
    #         mask_whole = tmp_arr
    #         mask_whole = np.expand_dims(mask_whole, axis=0)
    #     else:
    #        tmp_whole = np.sum(mask_whole,axis = 0)
    #        tmp_whole[tmp_whole>0]=1
    #        if np.sum(tmp_whole * tmp_arr) <= np.sum(tmp_arr)*0.25:
    #           mask_whole[0,:,:] =  mask_whole[0,:,:] + tmp_arr
    #        else:
    #           #print(mask_whole)
    #           mask_whole = np.row_stack((mask_whole,np.expand_dims(tmp_arr, axis=0)))
    #
    # mask_whole = np.transpose(mask_whole)
    # print(mask_whole.shape)
    # imagesMask = np.sum(imagesMask)
    # imagesMask = cv2.resize(imagesMask,(w/2,h/2))
    # print(imagesMask)
    out_mask = mask_save_dir + '/' + file + '.npy'
    np.save(out_mask, imagesMask)

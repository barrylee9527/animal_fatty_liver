# -*- coding: utf-8 -*-

# @Time    : 2018.10.25
# @Author  : kawa Yeung
# @Licence : bio-totem

######################
# Modified History:
# 1 Add the function of loading and running REGION_CLASSIFICATION_MODEL before mask rcnn 's segmentation
# 2 Change the size of input image(cancel cutting patches in patch_C * patch_C) in mask rcnn 's detection to speed up the whole processing
# 3 fix bugs in post_treatment, and make this scripts adapttde in colorectal project
# @Time    : 2019.01.19
# @Author  : Bohrium Kwong
######################

import os
import gc
import time
import glob
import pickle
import threading
import configparser

import cv2
import numpy as np
from skimage import exposure, io
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from utils.log_utils import get_logger
from utils.openslide_utils import Slide
from utils.region_process import openslide_region_predict
from utils.region_process import svs_region_to_probmat_save_img_mask
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.nucleus import nucleus_train
#from cell_classifition.cell_classifition import cancer_cell_caculating
#from cell_classifition.post_treatment import nuclei_statistics
from utils.binmat_conhull_utils import concave_hull_matrix


current_path = os.path.dirname(__file__)
#print(current_path)
LOGS_DIR = os.path.join(current_path, "..", "logs")
conf = configparser.ConfigParser()
conf.read(os.path.join(current_path, "..", "sys.ini"))
MASK_RCNN_SEG_MODEL = conf.get("DEFAULT", "MASK_RCNN_SEG_MODEL")
CELL_CLASSIFICATION_MODEL = conf.get("DEFAULT", "CELL_CLASSIFICATION_MODEL")
REGION_CLASSIFICATION_MODEL = conf.get("DEFAULT", "REGION_CLASSIFICATION_MODEL")
INPUT_IMAGE_DIR = conf.get("DEFAULT", "INPUT_IMAGE_DIR")
#xml_dir = os.path.join(current_path, "..", "output", "output_xml")
#if not os.path.isdir(xml_dir): os.makedirs(xml_dir)
cell_cls_prediction_dir = os.path.join(current_path, "..", "output", "output_cc_pickle")
if not os.path.isdir(cell_cls_prediction_dir): os.makedirs(cell_cls_prediction_dir)

GPU_COUNT = int(conf.get("DEFAULT", "GPU_COUNT"))
GPU_USE = conf.get("DEFAULT", "GPU_USE")
#DEVICE = '\"/gpu:' + GPU_USE + '\"'
print('\"/gpu:' + GPU_USE + '\"')
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(GPU_USE))
print('\"CUDA_VISIBLE_DEVICES\"'+ ' = ' '\"'+ GPU_USE + '\"')
# The graphic cards that assign to this task base on the configure file('sys.ini')
# by Bohrium Kwong 2019.01.24

def load_maskrcnn_model():
    """
    load maskrcnn nucleus model, and config the settings
    :return:
    """

    print("define the mask RCNN configuration!")
    config = nucleus_train.NucleusInferenceConfig()
    config.BACKBONE = "resnet101"
    config.DETECTION_MAX_INSTANCES = 2000
    config.POST_NMS_ROIS_INFERENCE = 6000
    config.RPN_NMS_THRESHOLD = 0.7
    config.DETECTION_NMS_THRESHOLD = 0.3
    config.BATCH_SIZE = 1 * GPU_COUNT #2  # the para indicate that you want to prediction image numbers everytime
    config.IMAGES_PER_GPU = 1  # 2   #every GPU can process how many image
    config.GPU_COUNT = GPU_COUNT  # The batch_size was calculated by GPU_COUNT * IMAGES_PER_GPU
    config.display()

    # DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    # TEST_MODE = "inference"
    print("load mask RCNN model and network weight!")
    print("Loading weights from ", MASK_RCNN_SEG_MODEL)
    # with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)
    model.load_weights(MASK_RCNN_SEG_MODEL, by_name=True)
    print("mask RCNN load complete!")

    return model


def load_cell_classification_model():
    """
    load cell classification trained model
    :return:
    """

    print("loading cell class model......")
    print("Loading weights from ", CELL_CLASSIFICATION_MODEL)
    model = load_model(CELL_CLASSIFICATION_MODEL)
    sgd = optimizers.SGD(lr=0.05, momentum=0.7, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    print("load cell class model complete!")

    return model


def load_region_classification_model():
    """
    load region classification trained model
    :return:
    """

    print("loading region class model......")
    print("Loading weights from ", REGION_CLASSIFICATION_MODEL)
    model = load_model(REGION_CLASSIFICATION_MODEL)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-1, decay=0.001,momentum=0.7,
                            nesterov=False,clipvalue=0.7,clipnorm=1),
              metrics=['accuracy'])
    datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=False)
    print("load REGION class model complete!")

    return model,datagen


def region_classification(model,image,sample_nums,patch_size):
    """
    region_classification_model predict image
    :param model: trained model
    :param image: the image(sample_nums*patch_size,patch_size,3) wanted to be predicted before reshape into 4D tensor
    :param sample_nums: the value of N in svs_to_probmat,it should be patch_R // patch_C
    :param patch_size: the value of patch_C
    :return: proba result list
    """  
#    image.reshape(sample_nums, patch_size,patch_size , 3)
    for l in range(sample_nums):
        slide_img = image[l * patch_size : patch_size * (l + 1) , : patch_size , :3]
        slide_img2 = slide_img - np.mean(slide_img,keepdims=True)
        if l ==0:
            input_data = np.expand_dims(slide_img2, axis=0)
        else:
            input_data = np.row_stack((input_data,np.expand_dims(slide_img2, axis=0)))
    prob = model.predict_proba(input_data,batch_size = sample_nums)
    #region_input_image will train into 4D tensor N*patch_C*patch_C×3,
    # make it predicted by region_classification_model in batch = sample_nums 
    # by Bohrium Kwong 2019.01.21
    return prob
    

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


def cell_cls_prediction_to_pickle(basename, cell_cls_prediction_info):
    """
    save cell classification information to pickle file
    :param cell_cls_prediction_info:
    :return:
    """

    with open(os.path.join(cell_cls_prediction_dir, basename+'.pkl'), 'wb') as fp:
        pickle.dump(cell_cls_prediction_info, fp)


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


def svs_to_probmat(svs, patch_R, patch_C, seg_model, class_model, region_model,patch_size):
    """
    convert svs file to probability matrix
    :param svs: svs file path
    :param patch_R: patch row
    :param patch_C: patch column
    :param seg_model: mask-rcnn segmentation model
    :param class_model: nucleus classification model
    :param patch_size: patch size
    :return:
    """
#the parameter cc_prob_threshold is no longer in use,I have modified the involved definition in the methods are related

    slide = Slide(svs)
    slide_width, slide_height = slide.get_level_dimension(0)

    N = patch_R // patch_C
    widen = patch_size // 2 + 2           # widen bounding
    # Change the value of widen from original（patch_size // 2） to （patch_size // 2 + 2）
    # In order to reduce bugs when calls the method of cancer_cell_caculating
    # by Bohrium Kwong 2019.01.21

    W_ps_NI = slide_width // patch_C   # 31782 // 299  = 106
    H_ps_NI = slide_height // patch_R  # 24529 // 598 = 41
    
    CancerProb_arr = np.zeros((slide_height, slide_width), dtype=np.float16)

    cell_ratio = 0.55   # the threshold that decide the patch is background or not

    cell_cls_prediction_info = np.zeros((1,8))
    svs_W_H_info = np.zeros((1, 4), dtype=np.uint32)

    # left-up data
    for w in range(1 , W_ps_NI - 1):
        for h in range(H_ps_NI):
            bound_C = 1                             # bound column flag
            bound_R = 1                             # bound row flag
            widen_patch_C = patch_C + widen
            widen_patch_R = patch_R + widen
            step_patch_R = widen_patch_C
            if (w+1) * patch_C + widen > slide_width:
                widen_patch_C = patch_C
                bound_C = 0
            if (h+1) * patch_R + widen > slide_height:
                widen_patch_R = patch_R
                step_patch_R = patch_C
                bound_R = 0

            widen_subHIC = np.array(slide.read_region((w * patch_C, h * patch_R), 0, (widen_patch_C, widen_patch_R)))
            widen_subHIC = widen_subHIC[:, :, :3]  # exclude alpha
            cc_widen_subHIC = widen_subHIC.copy()
           
            # print("widen_subHIC.shape: ", widen_subHIC.shape)
            # print("subHIC.shape: ", subHIC.shape)

            # rgb three channels value that >200 and  <40 are ignored segment
            rgb_s = (abs(widen_subHIC[:, :, 0] - 120) >= 80) & (abs(widen_subHIC[:, :, 1] - 120) >= 80) & (
                    abs(widen_subHIC[:, :, 2] - 120) >= 80)  # >200  <40
#            widen_subHIC_list = []
            cc_widen_subHIC_list = []
            bound_list = [(bound_C, bound_R)] * N
            if np.sum(rgb_s) <= (widen_patch_R * widen_patch_C) * cell_ratio:
                
                region_input_image = widen_subHIC[:patch_R,:patch_C,:]
            #region_input_image.reshape(N,patch_C,patch_C , 3)
                try:
                    region_prob = region_classification(region_model,region_input_image,N,patch_C)
                except Exception as e:
                    error_logger = get_logger(level="error")
                    error_logger.error('y: '+str(h * N)+ ' x: '+str(w) + ' something wrong in region classifying.', exc_info=True)
                    region_prob = np.zeros((N,2))
               #before the processing of nuclei segmentation in mask rcnn, we should screen the regions to be detected,
               # base on the result of region_classification,while a region is predicted as a cancer region,then start next step
               # by Bohrium Kwong 2019.01.21 
                ul_region_point = []
                region_flag = 0
                region_raw_tensor = np.zeros((1,patch_C,patch_C,3))
                for i in range(N):
                    if region_prob[i,0] >= 0.5:
                        region_point = (w * patch_C, h * patch_R + i * patch_C)
                        ul_region_point.append(region_point)
                        cc_widen_subHIC_list.append(cc_widen_subHIC[i * patch_C : i * patch_C + step_patch_R, :, :])
                        # get the region upper left point
                        if region_flag == 0 :
                            region_raw_tensor[0,:,:,:] = widen_subHIC[i*patch_C : (i + 1) * patch_C, :patch_C, :3]
                            region_flag = region_flag + 1
                        else:
                            region_raw_tensor = np.row_stack((region_raw_tensor,
                                                              np.expand_dims(widen_subHIC[i*patch_C : (i + 1) * patch_C, :patch_C, :3], axis=0)))
                       
                if len(ul_region_point) > 0:
#                if len(widen_subHIC_list) > 0:
                    image_tensor_len = region_raw_tensor.shape[0]
                    input_data = region_raw_tensor.reshape(image_tensor_len*patch_C,patch_C,3) 
                    imagesMask = maskrcnn_detection(seg_model, [input_data])
                    imagesMask = imagesMask[0]
                    #the original weight of mask rcnn input image is patch_C + step_patch_R,which is not necessary,I put a limit ':patch_C' on it.
                    # And if there are more than one region to be segmented,reshape it into an iamge(patch_C * N,patch_C),is better than
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
                    
                    del imagesMask, widen_subHIC,imagesMask_list,cc_widen_subHIC_list,cc_widen_subHIC
                    gc.collect()
                    
                    valit_flag = 0
                    #valit_flag is used to flag  how many regions are not predicted as a cancel region base on the region_classification result region_prob, 
                    # we could let the loop's variable 'each' minus to the value of valit_flag to return to real index in region_input_image which could be
                    #  consistent one-to-one match image_result_list
                    # by Bohrium Kwong 2019.01.23
                    for each in range(N):
                        if region_prob[each,0] >= 0.5 :    
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
    pkl_result = (cell_cls_prediction_info, svs_W_H_info)
    pkl_thread = threading.Thread(target=cell_cls_prediction_to_pickle, args=(slide.get_basename(), pkl_result,))
    pkl_thread.start()

    return slide, CancerProb_arr


def walk_file():
    """
    walk through the svs file and prediction them
    :param patch_R: patch row
    :param patch_C: patch column
    :param level: svs file level
    :param patch_size: patch size
    :return:
    """
    #the parameter cc_prob_threshold is no longer in use,I have modified the involved definition in all the methods are related
    # by Bohrium Kwong 2019.01.21   
    #probmat_dir = os.path.join(current_path, "..", "output", "output_probmat")
    #if not os.path.isdir(probmat_dir): os.makedirs(probmat_dir)
    #region_result_dir = os.path.join(current_path, "..", "output", "region_result")
    #if not os.path.isdir(region_result_dir): os.makedirs(region_result_dir)
    #add the method of saving region_result variable base on openslide_region_predict and openslide_region_predict
    # by Bohrium Kwong 2019.02.01

   # img_save_dir = os.path.join(current_path, "..", "output", "ori_image_save")
    #if not os.path.isdir(img_save_dir): os.makedirs(img_save_dir)
    # mask_save_dir = os.path.join(current_path, "..", "output", "ballooning")
    mask_save_dir = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/renew_img/mask_rcnn_mask'
    if not os.path.isdir(mask_save_dir): os.makedirs(mask_save_dir)
    
    info_logger = get_logger(level="info")
    error_logger = get_logger(level="error")

    seg_model = load_maskrcnn_model()
#    cls_model = load_cell_classification_model()
#    region_model,datagen = load_region_classification_model()
    start_time = time.time()
    input_path = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/renew_img/mask'
    file_name = os.listdir(input_path)
    print(file_name)
    svs_file = sorted(file_name)
#    file_svs_to_flag = int(len(svs_file) // 2)
    for file in svs_file:
        input_name = input_path + '/' + file
        try:
            svs_region_to_probmat_save_img_mask(input_name,seg_model,mask_save_dir,file.split('.')[0])
            info_logger.info("Finished inference %s, needed %.2f sec" % (file, time.time() - start_time))
        except Exception as e:
            print(e)
            error_logger.error('Inference %s Error' % file, exc_info=True)

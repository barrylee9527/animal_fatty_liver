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
from utils.region_process import svs_to_probmat
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.nucleus import nucleus_train



current_path = os.path.dirname(__file__)
#print(current_path)
LOGS_DIR = os.path.join(current_path, "..", "logs")
conf = configparser.ConfigParser()
conf.read(os.path.join(current_path, "..", "sys.ini"))
MASK_RCNN_SEG_MODEL = conf.get("DEFAULT", "MASK_RCNN_SEG_MODEL")
CELL_CLASSIFICATION_MODEL1 = conf.get("DEFAULT", "CELL_CLASSIFICATION_MODEL1")
CELL_CLASSIFICATION_MODEL2 = conf.get("DEFAULT", "CELL_CLASSIFICATION_MODEL2")
INPUT_IMAGE_DIR = conf.get("DEFAULT", "INPUT_IMAGE_DIR")
#xml_dir = os.path.join(current_path, "..", "output", "output_xml")
#if not os.path.isdir(xml_dir): os.makedirs(xml_dir)

cell_predict = True
#完成细胞核分割后是否进行预测的设置变量，如果设置为否，保存的路径是output_cc_pickle_non_cell_pridict
if cell_predict:
    cell_cls_prediction_dir = os.path.join(current_path, "..", "output", "output_cc_pickle")
else:
    cell_cls_prediction_dir = os.path.join(current_path, "..", "output", "output_cc_pickle_non_cell_pridict")
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
    print("Loading weights from ", CELL_CLASSIFICATION_MODEL1)
    model1 = load_model(CELL_CLASSIFICATION_MODEL1)
    print("Loading weights from ", CELL_CLASSIFICATION_MODEL2)
    model2 = load_model(CELL_CLASSIFICATION_MODEL2)
    sgd = optimizers.SGD(lr=0.05, momentum=0.7, nesterov=True)
    model1.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    model2.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    print("load cell class model complete!")

    return model1,model2

    

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




def walk_file(patch_R, patch_C, level, patch_size):
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
    probmat_dir = os.path.join(current_path, "..", "output", "output_probmat")
    if not os.path.isdir(probmat_dir): os.makedirs(probmat_dir)
    region_result_dir = os.path.join(current_path, "..", "output", "region_result")
    if not os.path.isdir(region_result_dir): os.makedirs(region_result_dir)
    #add the method of saving region_result variable base on openslide_region_predict and openslide_region_predict
    # by Bohrium Kwong 2019.02.01

    img_save_dir = os.path.join(current_path, "..", "output", "ori_image_save")
    if not os.path.isdir(img_save_dir): os.makedirs(img_save_dir)
    mask_save_dir = os.path.join(current_path, "..", "output", "mask_image_save")
    if not os.path.isdir(mask_save_dir): os.makedirs(mask_save_dir)
    
    info_logger = get_logger(level="info")
    error_logger = get_logger(level="error")

    seg_model = load_maskrcnn_model()
    cls_model1,cls_model2 = load_cell_classification_model()
#    region_model,datagen = load_region_classification_model()
    
    svs_file = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.ndpi"))
    svs_file = sorted(svs_file)

#    file_svs_to_flag = int(len(svs_file) // 2)
    for i in range(len(svs_file)):
        svs = svs_file[i]
#        svs_name = os.path.basename(svs).split('.')[0]
        try:
            start_time = time.time()
            info_logger.info("Starting inference %s..." % svs)
            slide = Slide(svs)
            cell_cls_prediction_info, svs_W_H_info = svs_to_probmat(slide,patch_R, patch_C, seg_model,cls_model1,cls_model2,patch_size,cell_predict)
            pkl_result = (cell_cls_prediction_info, svs_W_H_info)
            pkl_thread = threading.Thread(target=cell_cls_prediction_to_pickle, args=(slide.get_basename(), pkl_result,))
            pkl_thread.start()
                
    #            level_W, level_H = slide.get_level_dimension(level=level)
    #            level_prob_matrix = cv2.resize(result_prob_matrix.astype(np.float32), (level_W, level_H), interpolation=cv2.INTER_AREA)
    #            del result_prob_matrix
            del cell_cls_prediction_info, svs_W_H_info
            gc.collect()
            slide.close()
            info_logger.info("Finished inference %s, needed %.2f sec" % (svs, time.time() - start_time))
        except Exception:
            error_logger.error('Inference %s Error' % svs, exc_info=True)

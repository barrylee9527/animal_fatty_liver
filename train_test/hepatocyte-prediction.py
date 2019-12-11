# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:03:49 2018

@author: Biototem_1
"""
import pandas as pd
import numpy as np
import argparse
import datetime
#import GPUtil
import random
import keras
import glob
import time
import sys
import os
from PIL import Image
import shutil
from keras_applications import mobilenet_v2
from keras.models import *
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.xception import Xception
from keras.initializers import Orthogonal
from keras.utils import to_categorical
from keras.preprocessing import image
from generators import DataGenerator
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from skimage.transform import resize
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
import cv2
import numpy as np
import time
from MAUCpy import a_value
from MAUCpy import MAUC
from sklearn.metrics import f1_score
from keras.utils.generic_utils import CustomObjectScope
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# img_name = 'LIV999004 STZ+HFD+CHOL G4-24'
size1 = (72,72)
# with CustomObjectScope({'relu6': mobilenet_v2.relu6}):
Xception_model=load_model('/cptjack/totem/barrylee/codes/re-anno-hepatocyte-54/xception-hepatocyte.h5')
val = '/cptjack/totem/barrylee/cut_small_cell/hepat-tri-classification/new-train/val'
test_folders = [val+'/ballooning/',val+'/normal/',val+'/steatosis/']
print("\nImages for Testing")
print("=" * 30)
Xception_model.summary()
test_images = []
test_labels = []
test_img = []
files = []
for index, folder in enumerate(test_folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    test_img = glob.glob(folder + "*.png")
    images = [np.array(Image.fromarray(image).resize(size1)) for image in images]
    labels = [index] * len(images)
    test_images = test_images + images
    test_labels = test_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))
print("\n")
test_images = np.stack(test_images)
def imagenet_processing(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image[:,:,i] -= mean[i]
        image[:,:,i] /= std[i]
    return image
test_images = (test_images/255).astype(np.float32) ### Standardise
# test_images = imagenet_processing(test_images)
test_labels = np.array(test_labels).astype(np.int32)
print(len(test_labels[test_labels==0]))
print(test_labels)
Y_test = to_categorical(test_labels, num_classes=np.unique(test_labels).shape[0])
print(Y_test)
start_time= time.time()
x_posteriors = Xception_model.predict(test_images, batch_size=64)
print(x_posteriors)
predictions = np.argmax(x_posteriors, axis=1)
print(predictions)
cost_time=time.time()-start_time
print(test_img[0])

# for k in range(len(predictions)):
#     if test_labels[k] == 0 and predictions[k]==1:
#         shutil.copy(files[k],'/cptjack/totem/barrylee/cut_small_cell/wrong_he_to_imm'+'/'+files[k].split('/')[-2]+files[k].split('/')[-1])

from sklearn.metrics import accuracy_score
acc=accuracy_score(test_labels,predictions)
data=[]
for i in range(len(test_labels)):
    data.append((test_labels[i],x_posteriors[i]))

from MAUCpy import a_value
from MAUCpy import MAUC
auc = MAUC(data, 3)

from sklearn.metrics import f1_score
f1=f1_score(test_labels,predictions,average='weighted')

print("cost_time:",cost_time)
print("acc",acc)
print("F1:",f1)
# print("AUC:",auc)
cr = classification_report(test_labels, predictions, target_names = ["ballooning","normal","steatosis"], digits = 3)
print(cr, "\n")
print("Confusion Matrix")
print("=" * 30, "\n")
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
def draw_confusion_matrix_classes():
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["ballooning","normal","steatosis"], rotation=45, size=10)
    plt.yticks(tick_marks, ["ballooning","normal","steatosis"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    a = [0, 0,0]
    for i in range(len(a)):
        for j in range(len(a)):
            a[i] = cm[i][j] + a[i]

    for x in range(width):
        for y in range(height):
            plt.annotate(str(np.round(cm[x][y], 3)), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center')
    plt.savefig('./'+'64cell---.tif', bbox_inches='tight')
def draw_confusion_matrix():
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["ballooning","normal","steatosis"], rotation=45, size=10)
    plt.yticks(tick_marks, ["ballooning","normal","steatosis"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    a = [0, 0, 0]
    for i in range(len(a)):
        for j in range(len(a)):
            a[i] = cm[i][j] + a[i]

    for x in range(width):
        for y in range(height):
            plt.annotate(str(np.round(cm[x][y] / a[x], 3)), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center')

    plt.savefig('./'+'64cell.tif', bbox_inches='tight')
draw_confusion_matrix_classes()
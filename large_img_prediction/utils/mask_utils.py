# @Time    : 2018.12.01
# @Author  : kawa Yeung
# @Licence : bio-totem


import os
import glob
import configparser

import cv2
import tensorflow as tf
from skimage import io
from skimage import exposure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.log_utils import get_logger
from utils.probmat_utils import maskrcnn_detection, load_maskrcnn_model, rgb_similarity

current_path = os.path.dirname(__file__)


def image_mask(image, patch_R, patch_C, seg_model):
    """
    Create mask for image by using mask rcnn
    :param image: image been cut from svs file
    :param patch_R: patch row
    :param patch_C: patch column
    :param seg_model: mask-rcnn segmentation model
    :return:
    """

    im = Image.open(image)
    im_name = os.path.basename(image).split('.')[0]
    im_width, im_height = im.width, im.height

    N = patch_R // patch_C

    W_ps_NI = im_width // patch_C  # 31782 // 256  = 124
    # W_ps_NR = slide_width % patch_C    # 31782 %  256  = 38
    H_ps_NI = im_height // patch_R  # 24529 // 1024 = 23
    # H_ps_NR = slide_height % patch_R   # 24529 %  1024 = 977

    cell_ratio = 0.85  # the threshold that decide the patch is background or not

    output_dir = os.path.join(current_path, "..", "output", "output_mask")
    if not os.path.isdir(output_dir): os.makedirs(output_dir)

    np_im = np.array(im)[:, :, 0:3]  # exclude alpha
    for w in range(W_ps_NI):
        for h in range(H_ps_NI):
            subHIC = np_im[h * patch_R: (h+1) * patch_R, w * patch_C:(w+1) * patch_C, :]

            # rgb three channels value that >200 and  <40 are ignored segment
            rgb_s = (abs(subHIC[:, :, 0] - 120) >= 80) & (abs(subHIC[:, :, 1] - 120) >= 80) & (
                    abs(subHIC[:, :, 2] - 120) >= 80)  # >200  <40

            if np.sum(rgb_s) <= (patch_R * patch_C) * cell_ratio:
                # segment
                subHIC = np.where(rgb_similarity(subHIC, 15, 195), 250, subHIC)
                # adjust equalization histogram and adjust brightness
                for k in range(subHIC.shape[2]):
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(N * 4, 4))
                    subHIC[:, :, k] = clahe.apply(subHIC[:, :, k])
                subHIC = exposure.adjust_gamma(subHIC, gamma=1.5)
                subHIC = subHIC.reshape(N, patch_C, patch_C, 3)

                subHIC = subHIC.reshape(N, patch_C, patch_C, 3)
                allmask_prob_list = maskrcnn_detection(seg_model, subHIC)

                for i in range(len(allmask_prob_list)):
                    for layer in range(allmask_prob_list[i].shape[2]):
                        image, cnts, hierarchy = cv2.findContours(allmask_prob_list[i][:, :, layer],
                                                                  cv2.RETR_EXTERNAL,
                                                                  cv2.CHAIN_APPROX_SIMPLE)
                        np_im[h * patch_R + i * patch_C: h * patch_R + (i + 1) * patch_C, w * patch_C:(w + 1) * patch_C,
                        :] = cv2.drawContours(np_im[h * patch_R + i*patch_C: h*patch_R+(i+1)*patch_C, w * patch_C:(w + 1) * patch_C, :],
                                              cnts, -1, (0, 255, 0), 1)

                    # np_im[h * patch_R + i*patch_C: h*patch_R+(i+1)*patch_C, w * patch_C:(w + 1) * patch_C, :] = subHIC[i]

                    # plt.savefig(os.path.join(output_dir, f"{im_name}w{w}h{h}N{i}.png"))

    io.imsave(os.path.join(output_dir, f"{im_name}.png"), np_im)


def batch_image_mask(patch_R, patch_C):
    """
    Batch Create mask for image by using mask rcnn
    :param patch_R: patch row
    :param patch_C: patch column
    :return:
    """

    conf = configparser.ConfigParser()
    conf.read(os.path.join(current_path, "..", "sys.ini"))
    image_dir = conf.get("UTILS_MASK", "IMAGE_DIR")
    images = glob.glob(os.path.join(image_dir, "*.png"))
    images = sorted(images)

    info_logger = get_logger(level="info")
    error_logger = get_logger(level="error")

    DEVICE = "/gpu:1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    with tf.device(DEVICE):
        seg_model = load_maskrcnn_model()
        for image in images:
            try:
                image_mask(image, patch_R, patch_C, seg_model)
                info_logger.info(f"Create mask {image} success")
            except Exception as e:
                error_logger.error(f"Create mask {image} error", exc_info=True)

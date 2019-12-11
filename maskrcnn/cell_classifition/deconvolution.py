# -*- coding:utf-8 -*-

import numpy as np
import cv2
import math
from skimage import io, exposure, img_as_float
import warnings
warnings.filterwarnings("ignore")



# stainColorMap = {
#     'hematoxylin': [0.63100845,  0.63630474,    0.44378445],
#     'dab':         [0.40348947,  0.59615797,    0.6941123],
#     'res':         [0.6625893,   0.48960388,    0.5668011 ]
# }


def saparatestains(input_image,stainColorMap):
    he = np.array(stainColorMap['hematoxylin'])
    dab = np.array(stainColorMap['dab'])
    res = np.array(stainColorMap['res'])

    HDABtoRGB = np.array([he/np.linalg.norm(he, axis=0, keepdims=True),
                      dab/np.linalg.norm(dab, axis=0, keepdims=True),
                     res/np.linalg.norm(res, axis=0, keepdims=True)])
    RGBtoHDAB = np.linalg.inv(HDABtoRGB)


    input_image2 = input_image.astype(float) + 2
    input_image2 = -np.log(input_image2)
    imageOut = np.dot(input_image2.reshape((-1, 3)), RGBtoHDAB)
    imageOut = imageOut.reshape(input_image.shape)

    for c in range(imageOut.shape[2]):
        imageOut[:, :, c] = (imageOut[:, :, c] - imageOut[:, :, c].min()) \
                            / (imageOut[:, :, c].max()-imageOut[:, :, c].min())
        imageOut[:, :, c] = 1 - imageOut[:, :, c]
        v_min, v_max = np.percentile(imageOut[:, :, c], (0.1, 98.991))
        imageOut[:, :, c] = exposure.rescale_intensity(imageOut[:, :, c], in_range=(v_min, v_max))

    return imageOut


def ihc_sep(input_image,stainColorMap):
    he = np.array(stainColorMap['hematoxylin'])
    dab = np.array(stainColorMap['dab'])
    res = np.array(stainColorMap['res'])

    Io = 255
    input_image2 = input_image.astype(float).copy()

    for c in range(input_image2.shape[2]):
        a = input_image2[:, :, c]
        if c == 0:
            rgb = a.reshape((-1), order='F')
        else:
            rgb = np.row_stack((rgb, a.reshape((-1), order='F')))

    od_m = -np.log((rgb+1)/255)

    ref_vec = np.array([he, dab, res]).T

    d = np.dot(np.linalg.inv(ref_vec), od_m)

    h_t = np.array([-ref_vec[0, 0]*d[0, :],
                -ref_vec[1, 0]*d[0, :],
                -ref_vec[2, 0]*d[0, :]])

    H = 255 * np.exp(h_t)
    H = H.T.reshape(input_image2.shape, order='F')
    H = H*(H <= 255)*(H >= 0) + (H > 255)*255

    e_t = np.array([-ref_vec[0, 1]*d[1, :],
                -ref_vec[1, 1]*d[1, :],
                -ref_vec[2, 1]*d[1, :]])

    E = 255 * np.exp(e_t)
    E = E.T.reshape(input_image2.shape, order='F')
    E = E*(E <= 255)*(E >= 0) + (E > 255)*255

    return H, E


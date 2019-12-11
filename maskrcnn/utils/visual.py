# @Time    : 2018.11.23
# @Author  : kawa Yeung
# @Licence : bio-totem


import pickle

from PIL import ImageDraw

from utils.openslide_utils import Slide


def svs_region(svs_file, pkl_file, location, size):
    """
    svs read region, and put the inference information to it
    :param svs_file: svs file
    :param pkl_file: pickle file, storage inference information
    :param location: upper left point
    :param size: region size to read
    :return:
    """

    slide = Slide(svs_file)
    region_im = slide.read_region(location, 0, size)
    with open(pkl_file, "rb") as fp:
        cell_cls_prediction_info, svs_W_H_info = pickle.load(fp)

    w_start, h_start = location
    w_end, h_end = w_start + size[0], h_start + size[1]

    condition = (cell_cls_prediction_info[:, 0] >= w_start) & (cell_cls_prediction_info[:, 0] <= w_end) \
                & (cell_cls_prediction_info[:, 1] >= h_start) & (cell_cls_prediction_info[:, 1] <= h_end)

    nuclei_info = cell_cls_prediction_info[condition]

    return region_im, location, nuclei_info


def region_view(region_im, location, nuclei_info, class_proba):
    """
    view the cell region
    :param region_im: region image, PIL Image object
    :param location: region upper left point
    :param nuclei_info: model inference information
    :param class_proba: tuple, (cancer_cell_prob, lymphocyte_cell_prob)
    :return:
    """

    dr = ImageDraw.Draw(region_im)
    nuclei_info[:, 0] = nuclei_info[:, 0] - location[0]
    nuclei_info[:, 1] = nuclei_info[:, 1] - location[1]

    for l, info in enumerate(nuclei_info):
        draw_flag = 0
        nuclei_flag = nuclei_info[l, 2] > 10 and nuclei_info[l, 3] < 0.1
        if nuclei_info[l, 4] * nuclei_flag > class_proba[0] or nuclei_info[l, 2] > 350:
            draw_flag = 1
            color = (237, 28, 36)
        elif nuclei_info[l, 5] * nuclei_flag > class_proba[1]:
            draw_flag = 1
            color = (255, 215, 0)

        if draw_flag == 1:
            dr.rectangle((nuclei_info[l, 0], nuclei_info[l, 1], nuclei_info[l, 0]+5, nuclei_info[l, 1]+5), fill=color)

    return region_im

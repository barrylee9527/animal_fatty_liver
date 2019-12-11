# @Time    : 2019.01.28
# @Author  : kawa Yeung
# @Licence : bio-totem


import cv2
import numpy as np
from interval import Interval
from shapely.geometry import Point
from shapely.geometry import JOIN_STYLE

from utils.opencv_utils import OpenCV
from utils.concave_hull_utils import alpha_shape


def concave_hull_matrix(binary_matrix, concave_alpha):
    """
    Faster way to convert discrete binary matrix to concave hull binary matrix
    :param binary_matrix: numpy matrix
    :return:
    """

    # binary_matrix = np.uint8(binary_matrix)
    _, cnts, _, = cv2.findContours(np.uint8(binary_matrix), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # new_binary_matrix = binary_matrix * 255
    # binary = cv2.threshold(new_binary_matrix.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)[1]
    # opencv = OpenCV(binary)
    # cnts = opencv.find_contours(binary_matrix)
    np_cnts = np.zeros((1, 2), dtype=np.uint32)
    for cnt in cnts:
        temp_cnt = np.squeeze(np.asarray(cnt))
        np_cnts = np.row_stack((np_cnts, temp_cnt))

    np_cnts = np.delete(np_cnts, 0, axis=0)
    concave_hull, edge_points = alpha_shape(np_cnts, concave_alpha)

    concave_hull_x_min, concave_hull_y_min, concave_hull_x_max, concave_hull_y_max = concave_hull.bounds
    shape_interal = []
    polygon_list = []
    if concave_hull.geom_type == 'MultiPolygon':
        for polygon in concave_hull:
            x_min, y_min, x_max, y_max = polygon.bounds
            shape_interal.append((Interval(x_min, x_max), Interval(y_min, y_max)))
            # Can't fill the inner hole
            # mypolygon = polygon.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
            # polygon_list.append(mypolygon)

    else:
        shape_interal.append((Interval(concave_hull_x_min, concave_hull_x_max), Interval(concave_hull_y_min, concave_hull_y_max)))
        # mypolygon = concave_hull.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)
        # polygon_list.append(mypolygon)

    condition = np.zeros(binary_matrix.shape, dtype=np.bool)
    condition[int(concave_hull_y_min):int(concave_hull_y_max), int(concave_hull_x_min):int(concave_hull_x_max)] = True
    binmat_zero_indexs = np.argwhere(np.logical_and(binary_matrix == 0, condition))
    for each in binmat_zero_indexs:
        each = np.array([each[1], each[0]])
        point = Point(each)
        # for polygon in polygon_list:
        #     if polygon.covers(point):
        #         binary_matrix[int(each[1])][int(each[0])] = 1
        #         break
        if concave_hull.covers(point):
            binary_matrix[int(each[1])][int(each[0])] = 1

    return binary_matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # im = cv2.imread("/Users/kawa/Desktop/52800/52800_prob.png", 0)
    # im[im>0] = 1

    im = np.ones((100, 100))
    im[:50, :50] = 0
    binary_matrix = concave_hull_matrix(im, concave_alpha=0.000002)
    plt.imshow(binary_matrix)
    plt.show()

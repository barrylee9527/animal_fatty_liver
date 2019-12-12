import math
from scipy import signal
import numpy as np
import cv2
import csv
import sys
import glob

sys.path.append('../')
import time
# from output_process.nuclei_statistics import nuclei_amount
from utils.log_utils import get_logger
from utils.openslide_utils import Slide
import configparser
import os
import pickle
from scipy.spatial import Delaunay

from utils.tissue_utils import OpenCV
from output_process.process_script import remove_discrete_point, get_mask, fill_contours,get_centerpoint,matrix_resize

# """
# 获取组织区域MASK（0为非组织区域，1为组织区域）
# bgimg_rgb: RGB三通道的图片矩阵
#
# """
# 移到新文件

# def get_mask(bgimg_rgb):
#     mask, _ = get_tissue(bgimg_rgb, contour_area_threshold=10000)
#     mask = remove_discrete_point(mask, 50)
#     # 仅获取外部轮廓
#     # contours = get_contours(mask,cv2.RETR_EXTERNAL)
#     contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')
#     mask = fill_contours(mask.shape, contours)
#     return mask


"""
获取轮廓，并将轮廓列表顺序按轮廓质心坐标从左到右，从上到下排序
matrix：二值化矩阵
"""
def get_contours_by_sort(matrix):
    # contours = get_contours(matrix,cv2.RETR_EXTERNAL)
    cnt_number_lst = []
    # for cnt in contours:
    #     M = cv2.moments(cnt)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     cnt_number_lst.append([cnt, cX + cY * matrix.shape[0]])
    # cnt_number_lst.sort(key=lambda x: x[1])
    #
    # result_contours = [cnt_number[0] for cnt_number in cnt_number_lst]


    # 获取轮廓
    contours = OpenCV(matrix).find_contours(is_binary=True, mode='RETR_EXTERNAL')
    # 中心点
    centerpoint = list(map(get_centerpoint,contours))
    # 将中心点、轮廓对应排序
    cnt_number_lst = [[centerpoint[i][0],centerpoint[i][1],contours[i]]for i in range(len(contours)) ]
    cnt_number_lst.sort(key = lambda x: (x[0],x[1]))
    # 提取出来轮廓去掉中心点信息
    result_contours = [cnt_number[2] for cnt_number in cnt_number_lst]

    return result_contours

# 移到新文件
# """
# 去除矩阵中的离散点
# matrix: Numpy矩阵，浮点类型
# size: OPENCV开操作卷积核大小
# """
# def remove_discrete_point(matrix, size):
#     # 尽可能保证matrix类型为uint8类型
#     # 因为在当前项目的测试下，该函数处理float类型耗费的时间比uint8类型高出100倍
#     # float类型实测为17秒，uint8类型实测为0.1秒
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size));
#     matrix = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel);
#
#     return matrix
#
# 废弃方法
# """
# 获取二值化矩阵的轮廓
# matrix_0：二值化矩阵
# Contour_Retrieval_Mode：默认cv2.RETR_TREE
# """
# def get_contours(matrix_0,Contour_Retrieval_Mode=cv2.RETR_TREE):
#
# 	# matrix=np.copy(matrix_0)
# 	#
# 	# #将矩阵的边框处置0，靠近边框的轮廓也能顺利提取
# 	# matrix=cv2.copyMakeBorder(matrix,0,0,0,0,cv2.BORDER_CONSTANT,value=0)
# 	#
# 	# #提取轮廓
# 	# contours, hierarchy = cv2.findContours(matrix.astype(np.uint8),Contour_Retrieval_Mode,cv2.CHAIN_APPROX_SIMPLE)
# 	contours = OpenCV(matrix_0).find_contours(is_binary=True,mode='RETR_TREE')
# 	contours = 1
# 	return contours
#
# 移到新文件
# '''
# 填充轮廓到二维矩阵中，轮廓区域为1，非轮廓区域为0
# shape：二维矩阵的shape
# cnt_list：轮廓列表
# '''
#
#
# def fill_contours(shape, cnt_list):
#     result_matrix = np.full((shape[0], shape[1]), 0, dtype=np.uint8)
#     cv2.drawContours(result_matrix, cnt_list, -1, 1, cv2.FILLED)
#     return result_matrix
#


'''
过滤轮廓面积占最大轮廓面积比率小于指定过滤值的轮廓区域
matrix_0: 二值化矩阵
filter_rate: 过滤值
Contour_Retrieval_Mode：默认cv2.RETR_TREE
'''


def filter_contour_region(matrix_0, filter_rate):
    matrix = np.copy(matrix_0)

    # 获取matrix矩阵轮廓
    # contours = get_contours(matrix,Contour_Retrieval_Mode)

    # 获取轮廓面积及轮廓列表，并将列表按照轮廓面积从小到大进行排序
    # area_cnt_list = []
    # for cnt in contours:
    #     area_cnt_list.append([cv2.contourArea(cnt), cnt])
    # area_cnt_list.sort(key=lambda x: x[0])

    # 获取轮廓面积占最大轮廓面积的比率大于等于指定过滤值的轮廓列表
    # cnt_list = []
    # for area_cnt in area_cnt_list:
    #     if area_cnt[0] / area_cnt_list[-1][0] >= filter_rate:
    #         cnt_list.append(area_cnt[1])


    # 改用列表推到式
    # 获取matrix矩阵轮廓
    contours = OpenCV(matrix).find_contours(is_binary=True)
    # 获取轮廓面积及轮廓列表，并将列表按照轮廓面积从小到大进行排序
    area_cnt_list = [[cv2.contourArea(cnt), cnt] for cnt in contours]
    area_cnt_list.sort(key=lambda x: x[0])
    cnt_list =[area_cnt[1] for area_cnt in area_cnt_list if area_cnt[0] / area_cnt_list[-1][0] >= filter_rate ]
    # 填充轮廓
    filter_matrix = fill_contours(matrix.shape, cnt_list)

    return filter_matrix


'''
过滤轮廓面积占矩阵面积比率小于指定过滤值的轮廓区域
matrix_0：二值化矩阵
filter_rate：过滤值
Contour_Retrieval_Mode：默认cv2.RETR_TREE
'''
def filter_smresult_contour_region(matrix_0, filter_rate):
    matrix = np.copy(matrix_0)

    # 获取轮廓
    # contours=get_contours(matrix,Contour_Retrieval_Mode)

    # 初始化整体区域
    # result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=matrix.dtype)、

    # # # 遍历所有轮廓区域
    # for cnt in contours:
    #
    #     # 获取当前区域面积
    #     area = cv2.contourArea(cnt)
    #
    #     # 获取当前区域占矩阵区域的面积比率
    #     rate_area = area / (matrix.shape[0] * matrix.shape[1])
    #
    #     # 过滤当前区域占矩阵区域的面积比率小于等于filter_rate的区域
    #     if rate_area > filter_rate:
    #         # 将当前区域过滤matrix，并合并到整体区域中
    #         result_matrix += fill_contours(matrix.shape, [cnt])

    # 获取轮廓
    contours = OpenCV(matrix).find_contours(is_binary=True)
    # 初始化整体区域
    result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=matrix.dtype)
    # 获取当前区域面积
    area_cnt_list = list(map(cv2.contourArea, contours))
    # 获取当前区域占矩阵区域的面积比率
    rate_area = [area/ matrix.size for area in area_cnt_list]
    # 过滤当前区域占矩阵区域的面积比率小于等于filter_rate的区域
    rate_filter_area = [ fill_contours(matrix.shape, [contours[i]])
                        for i in range(len(contours)) if rate_area[i] >filter_rate ]
    # 将当前区域过滤matrix，并合并到整体区域中
    for i in rate_filter_area:
        result_matrix += i
    return result_matrix


'''
将矩阵经过Delaunay三角剖分算法处理，并将三角部分进行填充
matrix_0: 二值化矩阵
triangle_maxlen: 三角剖分算法中三角形最大边长值
discrete_distance：离散间距，将矩阵进行离散化处理,这样可以大大加快三角剖分算法和三角填充的处理速度，当前默认为10
'''
def get_triangle_matrix(matrix_0, triangle_maxlen, discrete_distance):

    matrix = np.copy(matrix_0)

    triangle_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.uint8)

    # 将矩阵进行离散化处理,这样可以大大加快三角剖分算法和三角填充的处理速度
    # 加速的主要原理是减少输入三角剖分算法的点坐标数量

    # 1.如果用于其他项目或者改变了矩阵下采样率，离散间距则需要进行调节，否则会造成三角剖分算法结果面积偏小。
    # 2.一般来说，若离散间距偏大，三角剖分面积便会偏小。这主要是因为过滤的点数偏多，影响三角剖分结果的轮廓边缘部分。
    # 3.当矩阵下采样率受到变化，亦或者是其他项目使用此函数时，应当尝试对比离散化处理和没离散化处理的三角剖分结果。
    # 4.以下使用10作为离散间距仅适用于当前结直肠癌项目九分类矩阵的下采样率。

    point_filter = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.int)
    point_filter[::discrete_distance, ::discrete_distance] = 1
    matrix = point_filter * matrix

    if np.max(matrix) != 0:

        # 获取所有需要三角剖分算法处理的点坐标
        npy = np.argwhere(matrix != 0)
        if npy.size > 0:
            # 坐标位置调换，为了适应三角剖分算法的坐标位置
            npy[:, [0, 1]] = npy[:, [1, 0]]

            # 三角剖分算法处理
            tri = Delaunay(npy)

            # 获取三角形的边
            eadge_npy = np.array([npy[tri.vertices[:, 0]], npy[tri.vertices[:, 1]], npy[tri.vertices[:, 2]]])

            # 获取三角形边长
            hypot_npy = np.array([np.sqrt(
                np.power(eadge_npy[0, :, 0] - eadge_npy[1, :, 0], 2) + np.power(eadge_npy[0, :, 1] - eadge_npy[1, :, 1],
                                                                                2)),
                np.sqrt(np.power(eadge_npy[0, :, 0] - eadge_npy[2, :, 0], 2) + np.power(
                    eadge_npy[0, :, 1] - eadge_npy[2, :, 1], 2)),
                np.sqrt(np.power(eadge_npy[1, :, 0] - eadge_npy[2, :, 0], 2) + np.power(
                    eadge_npy[1, :, 1] - eadge_npy[2, :, 1], 2))])

            # 轴互换
            eadge_npy = np.swapaxes(eadge_npy, 1, 0)
            hypot_npy = np.swapaxes(hypot_npy, 1, 0)

            # 获取三角形三条边长均小于等于最大三角形边长的三角形边
            eadge_npy = eadge_npy[(hypot_npy[:, 0] <= triangle_maxlen) &
                                  (hypot_npy[:, 1] <= triangle_maxlen) &
                                  (hypot_npy[:, 2] <= triangle_maxlen)]

            # 绘制三角形
            cv2.fillPoly(triangle_matrix, eadge_npy, 1)

    return triangle_matrix


'''
获取Norm占组织区域比率小于等于norm_rate，并且Mus, Str, Lym之和占组织区域比率大于等于mus_str_lym_rate的组织区域
matrix：九分类矩阵
mask: MASK矩阵（组织区域为1，非组织区域为0）
norm_value: 在matrix中Norm的值 
norm_rate: Norm占组织区域比率的过滤值
mus_value：在matrix中Mus的值
str_value：在matrix中Str的值
lym_value：在matrix中Lym的值
mus_str_lym_rate：Mus, Str, Lym占组织区域比率的过滤值
'''

def filter_tissue_matrix(matrix_0, mask, norm_value, norm_rate,
                         mus_value, str_value, lym_value, mus_str_lym_rate):
    matrix = np.copy(matrix_0)

    # MASK矩阵面积
    image_area = mask.shape[0] * mask.shape[1]

    # 获取matrix矩阵轮廓
    # contours=get_contours(mask)
    # contours = get_contours(mask, cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=matrix.dtype)



    # 遍历获取符合条件的轮廓
    for cnt in contours:
        # 计算轮廓面积占MASK矩阵面积的比率
        area = cv2.contourArea(cnt)
        area_rate = area / image_area

        # 只处理合适面积的轮廓，面积比率小于0.01的轮廓过于零散，大于0.95会获取到非组织区域的轮廓
        if 0.01 < area_rate < 0.95:
            # 填充轮廓
            filter_matrix = fill_contours(matrix.shape, [cnt])

            # 过滤matrix矩阵，非填充轮廓区域为0，填充轮廓区域为原值
            filter_matrix = filter_matrix * matrix

            # 分别获取填充轮廓区域norm，mus，str以及lym的元素个数和
            sum_norm = np.sum(filter_matrix == norm_value)
            sum_mus = np.sum(filter_matrix == mus_value)
            sum_str = np.sum(filter_matrix == str_value)
            sum_lym = np.sum(filter_matrix == lym_value)

            # 获取Norm占组织区域比率小于等于norm_rate，并且Mus, Str, Lym元素个数和占组织区域比率大于等于mus_str_lym_rate的组织区域
            if sum_norm / area <= norm_rate and (sum_mus + sum_str + sum_lym) / area >= mus_str_lym_rate:
                result_matrix = result_matrix + filter_matrix

    return result_matrix


'''
获取肿瘤区域矩阵
matrix：九分类矩阵
mask：组织区域为1，非组织区域为0
cancer_index：九分类矩阵中肿瘤区域的值
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
filter_rate：过滤肿瘤区域占矩阵区域的比率小于等于filter_rate的肿瘤区域
'''


def get_cancer_region(matrix, mask, cancer_index, open_kernel_size, triangle_maxlen, filter_rate):
    # 二值化处理matrix，非0为癌症细胞
    cancer_only_matrix = np.copy(matrix)
    cancer_only_matrix[cancer_only_matrix != cancer_index] = 0

    # 去除矩阵中的零散点，避免三角剖分算法得到的癌症区域因为零散点而拓展
    cancer_only_matrix = remove_discrete_point(cancer_only_matrix.astype(np.uint8), open_kernel_size).astype(
        cancer_only_matrix.dtype)

    # 获取MASK轮廓外部
    # contours=get_contours(mask)
    # contours = get_contours(mask, cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 初始化整体肿瘤区域
    result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.int)

    # 遍历MASK轮廓分别获取不同组织的肿瘤区域
    # for cnt in contours:
    #     # 获取当前组织区域的九分类肿瘤区域矩阵
    #     filter_matrix = fill_contours(matrix.shape, [cnt])

    #
    #     part_cancer_only_matrix = filter_matrix * cancer_only_matrix
    #
    #     # 利用三角剖分算法获取癌症区域矩阵
    #     cancer_matrix = get_triangle_matrix(part_cancer_only_matrix, triangle_maxlen, 10)
    #
    #     # 过滤肿瘤区域面积占矩阵面积比率小于指定过滤值的肿瘤区域，cv2.RETR_EXTERNAL是为了填补肿瘤区域内的空洞
    #     result_matrix += filter_smresult_contour_region(cancer_matrix, filter_rate)
    # #
    #
    contours_len = len(contours)
    #
    # # 获取当前组织区域的九分类肿瘤区域矩阵
    #
    filter_matrix_list = map(fill_contours,[matrix.shape]*contours_len,[contours])
    #
    part_cancer_only_matrix_list = map(lambda x:x* cancer_only_matrix,filter_matrix_list)
    # 利用三角剖分算法获取癌症区域矩阵
    cancer_matrix_list =map(get_triangle_matrix,part_cancer_only_matrix_list, [triangle_maxlen]*contours_len , [10]*contours_len)
    # 过滤肿瘤区域面积占矩阵面积比率小于指定过滤值的肿瘤区域，cv2.RETR_EXTERNAL是为了填补肿瘤区域内的空洞
    matrix_list = list(map(filter_smresult_contour_region,cancer_matrix_list, [filter_rate]*contours_len))
    for i in matrix_list:
        result_matrix += i

    return result_matrix


'''
获取肿瘤区域的浸润区域
cancer_matrix：肿瘤区域矩阵
mask：组织区域为1，非组织区域为0
exclude_matrix：肿瘤区域，间质区域，以及浸润区域需要去除区域的过滤矩阵
invasive_thickness：浸润区域的厚度
filter_rate：过滤浸润区域面积占最大浸润区域面积的比率小于指定过滤值的浸润区域
'''


def get_cancer_invasive_region(cancer_matrix, mask, exclude_matrix, invasive_thickness, filter_rate):
    # 将肿瘤区域矩阵按倍数缩小，这一步是为了快速得到浸润区域矩阵
    scale_multiple = 0.1

    # cancer_resize_matrix=cv2.resize(cancer_matrix.astype(np.float),
    # 								(int(cancer_matrix.shape[1]*scale_multiple),
    # 								 int(cancer_matrix.shape[0]*scale_multiple))).astype(np.int)

    cancer_resize_matrix = OpenCV(cancer_matrix.astype(np.float)) \
        .resize(int(cancer_matrix.shape[1] * scale_multiple), int(cancer_matrix.shape[0] * scale_multiple),
                interpolation='INTER_LINEAR') \
        .astype(np.int)

    # 通过卷积获取浸润区域矩阵
    kernel_length = 2 * int(invasive_thickness * scale_multiple) + 1
    kernel = np.full((kernel_length, kernel_length), 1, dtype=np.int)
    invasive_margin_matrix = signal.convolve2d(cancer_resize_matrix, kernel, boundary='fill', mode='same')
    invasive_margin_matrix[invasive_margin_matrix > 0] = 1

    # 将癌症浸润区域矩阵还原成matrix矩阵的尺寸
    # invasive_margin_matrix=cv2.resize(invasive_margin_matrix.astype(np.float),
    # 								(cancer_matrix.shape[1],cancer_matrix.shape[0])).astype(np.int)
    invasive_margin_matrix = OpenCV(invasive_margin_matrix.astype(np.float)) \
        .resize(cancer_matrix.shape[1], cancer_matrix.shape[0], interpolation='INTER_LINEAR') \
        .astype(np.int)

    # 平滑处理浸润区域边缘的锯齿
    invasive_margin_matrix = remove_discrete_point(
        invasive_margin_matrix.astype(np.uint8), 30).astype(np.int)

    # 过滤浸润区域的exclude_matrix区域，肿瘤区域，以及MASK区域
    invasive_margin_matrix[(exclude_matrix == 1) | (cancer_matrix == 1) | (mask == 0)] = 0

    # 获取轮廓面积占最大轮廓面积比率大于指定过滤值的轮廓，此处是为了过滤零散浸润区域
    invasive_margin_matrix = filter_contour_region(invasive_margin_matrix, filter_rate)

    # 过滤浸润区域的肿瘤区域
    invasive_margin_matrix[cancer_matrix == 1] = 0

    return invasive_margin_matrix


'''
获取浸润区域
cancer_matrix：肿瘤区域矩阵
mask：组织区域为1，非组织区域为0
exclude_matrix：肿瘤区域，间质区域，以及浸润区域需要去除区域的过滤矩阵
invasive_thickness：浸润区域的厚度
filter_rate：过滤浸润区域面积占最大浸润区域面积的比率小于指定过滤值的浸润区域
'''


def get_invasive_region(cancer_matrix, mask, exclude_matrix, invasive_thickness, filter_rate):
    # 获取MASK轮廓外部
    # contours=get_contours(mask)
    # contours = get_contours(mask, cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 整体浸润区域
    result_matrix = np.full((cancer_matrix.shape[0], cancer_matrix.shape[1]), 0, dtype=np.int)

    # 遍历MASK轮廓
    for cnt in contours:

        # 获取当前组织区域的肿瘤区域
        tissue_matrix = fill_contours(mask.shape, [cnt])
        tissue_cancer_matrix = tissue_matrix * cancer_matrix

        # 获取当前组织区域的肿瘤区域的所有轮廓
        # cancer_contours=get_contours(tissue_cancer_matrix)
        cancer_contours = OpenCV(tissue_cancer_matrix).find_contours(is_binary=True)




        # 遍历当前组织区域的肿瘤区域的所有轮廓
        for cancer_cnt in cancer_contours:

            # 获取当前肿瘤区域
            part_cancer_matrix = fill_contours(cancer_matrix.shape, [cancer_cnt])

            # 获取当前肿瘤区域的浸润区域
            invasive_margin_matrix = get_cancer_invasive_region(part_cancer_matrix, mask, exclude_matrix,
                                                                invasive_thickness, filter_rate)

            # 将当前组织区域过滤当前肿瘤区域的浸润区域，防止当前组织区域的浸润区域跨越到其他组织区域当中
            invasive_margin_matrix = invasive_margin_matrix * tissue_matrix

            # 合并当前浸润区域到整体浸润区域当中
            result_matrix += invasive_margin_matrix

    return result_matrix


'''
获取九分类矩阵中index_list指定细胞类型值的二值化矩阵
matrix：九分类矩阵
index_list：九分类矩阵指定细胞类型的值列表
'''


def get_index_list_region(matrix, index_list):
    result_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.int)

    # 获取九分类矩阵中index_list指定细胞类型值的二值化矩阵
    for index in index_list:
        result_matrix[:, :][matrix[:, :] == index] = 1

    return result_matrix


'''
获取肿瘤，浸润，间质区域矩阵（癌症区域为1，浸润区域为3，间质区域为2）
svs_img：svs图片路径
matrix_file: 九分类矩阵路径
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 浸润区域的厚度
'''


def get_cancer_invasive_matrix(svs_file, matrix_file, open_kernel_size, triangle_maxlen, invasive_thickness):
    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取2级下采样的SVS图片尺寸
    matrix_size = slide.get_level_dimension(level=2)


    # 获取2级下采样的SVS图片
    svs_img = slide.read_region((0, 0), 2, matrix_size)


    # 加载九分类矩阵
    matrix = np.load(matrix_file)

    matrix = matrix_resize(slide,matrix,patch_size=224)
    slide.close()

    # 底图尺寸以及格式处理
    svs_img = svs_img.resize((matrix.shape[1], matrix.shape[0]))
    svs_img_rgb = svs_img.convert('RGB')

    # 获取MASK，并将其过滤matrix矩阵中的非组织区域
    mask = get_mask(np.array(svs_img_rgb).astype(np.uint8))

    # 过滤MASK区域面积占矩阵面积比率小于等于0.01的MASK区域
    mask = filter_smresult_contour_region(mask, 0.01)

    # 将MASK过滤九分类矩阵
    matrix = matrix.astype(int)
    matrix = matrix * mask

    # 过滤matrix矩阵中非细胞类型的值
    matrix[:, :][(matrix < 0) | (matrix > 8) | (matrix <= 1)] = 9

    # 获取Norm占组织区域比率小于等于0.4，并且Mus, Str, Lym之和占组织区域比率大于等于0.1的组织区域
    # 并用该组织区域去过滤九分类矩阵
    matrix = filter_tissue_matrix(matrix, mask, 6, 0.4, 5, 7, 3, 0.1)

    # 获取九分类矩阵中只包含为2和6的矩阵
    exclude_matrix = get_index_list_region(matrix, [2, 6])

    # 获取肿瘤区域，0.001是过滤肿瘤区域占矩阵区域的比率小于等于0.001的肿瘤区域
    cancer_matrix = get_cancer_region(matrix, mask, 8, open_kernel_size, triangle_maxlen, 0.001)

    # 获取肿瘤浸润区域
    invasive_margin_matrix = get_invasive_region(cancer_matrix, mask, exclude_matrix, invasive_thickness, 1)

    # 过滤肿瘤区域的非组织部分以及九分类矩阵中只包含为2和6的矩阵
    cancer_matrix[(mask == 0) | (exclude_matrix == 1)] = 0

    # 获取九分类矩阵中只包含为3，4，5，7的矩阵，并获取肿瘤间质区域
    stroma_matrix = get_index_list_region(matrix, [3, 4, 5, 7])
    stroma_matrix = stroma_matrix * cancer_matrix

    # 获取癌症，浸润，间质区域矩阵
    cancer_invasive_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.float)
    cancer_invasive_matrix[cancer_matrix == 1] = 1.
    cancer_invasive_matrix[stroma_matrix == 1] = 2.
    cancer_invasive_matrix[invasive_margin_matrix == 1] = 3.

    return cancer_invasive_matrix, mask


'''
获取指定步长区域的细胞数量矩阵
nuclei_info：对应细胞的PKL细胞核信息
svs_width：SVS图片0级下采样宽度
svs_height：SVS图片0级下采样高度
step：SVS图片0级下采样的区域步长
'''


def get_nuclei_amount_matrix(nuclei_info, svs_width, svs_height, step):
    # 获取指定步长的细胞数量矩阵尺寸
    width_step_num = math.ceil(svs_width / step)
    height_step_num = math.ceil(svs_height / step)
    # 获取初始为0的细胞数量矩阵
    nuclei_amount_matrix = np.zeros((height_step_num, width_step_num), dtype=np.float32)
    # 将细胞核信息中每个细胞的坐标转换为指定步长的细胞数量矩阵映射坐标
    region_coord = np.array([nuclei_info[:, 1] // step, nuclei_info[:, 0] // step])
    print(region_coord.shape)
    # 将映射坐标转换为复数，Numpy.unique方法便能够根据映射坐标统计映射坐标的重复数量（细胞数量）
    region_coord_plural = region_coord[0, :] + region_coord[1, :] * 1j
    # 获取复数映射坐标及其重复数量（细胞数量）
    # Numpy.unique是向量化运算，因此能够快速统计区域细胞数量
    region_coord_plural_count = np.unique(region_coord_plural, return_counts=True)
    print(region_coord_plural_count)
    # 将复数映射坐标及其重复数量（细胞数量）映射到细胞数量矩阵
    # for i in range(region_coord_plural_count[0].shape[0]):
    #     x = np.real(region_coord_plural_count[0][i]).astype(np.int)
    #     y = np.imag(region_coord_plural_count[0][i]).astype(np.int)
    #     nuclei_amount_matrix[x, y] = region_coord_plural_count[1][i]
    loop = range(region_coord_plural_count[0].shape[0])
    x_list = list(map(lambda i: np.real(region_coord_plural_count[0][i]).astype(np.int),loop))
    y_list = list(map(lambda i:np.imag(region_coord_plural_count[0][i]).astype(np.int),loop))
    for i in loop:
        nuclei_amount_matrix[x_list[i], y_list[i]] = region_coord_plural_count[1][i]
    return nuclei_amount_matrix


'''
获取所有指定步长区域的细胞数量分布矩阵
pkl_file: PKL文件路径
svs_file: SVS文件路径
step: SVS图片0级下采样的区域步长
'''


def get_nuclei_amount_region_matrix(pkl_file, svs_file, step):
    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取0级下采样的图片尺寸，如果修改级数，步长需要乘去相应级数的缩放倍数
    svs_width, svs_height = slide.get_level_dimension(level=0)

    slide.close()

    # 加载PKL文件
    with open(pkl_file, "rb") as fp:
        nuclei_info, _ = pickle.load(fp)

    # 将细胞信息数组中的后四项取其最大值的相对索引
    nuclei_label1 = np.argmax(nuclei_info[:, 4:7], axis=1)
    nuclei_label2 = np.argmax(nuclei_info[:, 7:10], axis=1)
    # 并行化
    # 获取上皮(肿瘤)细胞、 淋巴细胞、其他细胞信息数组
    # target_cell_nuclei_info  = [epithelial_nuclei_info, lymphocyte_nuclei_info, other_nuclei_info]
    # target_cell_nuclei_info = list(map(lambda x: nuclei_info[(nuclei_label == x) & (nuclei_info[:, 3] < 0.1)], [0, 1, 2]))
    result_nuclei_matrix = [ nuclei_info[(nuclei_label1==1)],
                          nuclei_info[(nuclei_label1==2)],
                          nuclei_info[(nuclei_label1==0) & (nuclei_label2==0)],
                          nuclei_info[(nuclei_label1==0) & (nuclei_label2==1)],
                          nuclei_info[(nuclei_label1==0) & (nuclei_label2==2)]]
    length = len(result_nuclei_matrix)
    print(result_nuclei_matrix)
    immune_cells_nuclei_matrix, other_cells_nuclei_matrix, bresultooning_nuclei_matrix, normal_nuclei_matrix,steatosis_nuclei_matrix = list(map(get_nuclei_amount_matrix, result_nuclei_matrix, [svs_width] * length, [svs_height] * length, [step] * length))
    # 使用map方法将[target_cell_nuclei_info, [svs_width] * 4, [svs_height] * 4, [step] * 4]作为传参调用get_nuclei_amount_matrix
    return immune_cells_nuclei_matrix,other_cells_nuclei_matrix,\
           bresultooning_nuclei_matrix,normal_nuclei_matrix,steatosis_nuclei_matrix
'''
'''
def get_nuclei_classification_result_info(nuclei_info):
    np.set_printoptions(suppress=True)
    count = nuclei_info.shape[0]
    result = np.array([None]*count)
    x = np.array([None]*count)
    y = np.array([None]*count)
    print(nuclei_info)
    nuclei_label1 = np.argmax(nuclei_info[:, 4:7],axis=1)
    nuclei_label2 = np.argmax(nuclei_info[:, 7:10],axis=1)
    index1 = np.argwhere((nuclei_label1 == 2) & (nuclei_info[:, 3] < 0.1))
    index2 = np.argwhere((nuclei_label1 == 1) & (nuclei_info[:, 3] < 0.1))
    index3 = np.argwhere((nuclei_label1==0) & (nuclei_label2==0) & (nuclei_info[:, 3] < 0.1))
    index4 = np.argwhere((nuclei_label1==0) & (nuclei_label2==1) & (nuclei_info[:, 3] < 0.1))
    index5 = np.argwhere((nuclei_label1==0) & (nuclei_label2==2) & (nuclei_info[:, 3] < 0.1))
    result[index1] = ['other_cells']
    result[index2] = ['immune_cells']
    result[index3] = ['ballooning_cells']
    result[index4] = ['normal_cells']
    result[index5] = ['steatosis_cells']
    print(result)
    return result





'''
获取上皮(肿瘤)细胞，淋巴细胞，其他细胞的细胞数量分布矩阵
以及肿瘤，浸润，间质区域矩阵和组织区域矩阵
pkl_file: PKL文件路径
svs_file: SVS文件路径
step: SVS图片0级下采样的区域步长
matrix_file: 坏死VS其他(上皮、间质)的二分类矩阵路径
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 浸润区域的厚度
region_filter：若为1，细胞数量分布需要经过肿瘤区域，间质区域，以及浸润区域矩阵过滤，为0则不过滤
'''


def get_nuclei_invasive_region_matrix(pkl_file, svs_file, step, matrix_file, open_kernel_size, triangle_maxlen,
                                      invasive_thickness, region_filter):
    # 加载dead_other二分类矩阵
    # 未使用变量
    # region_dead_other_output_matrix = np.load(matrix_file)

    # 分别获取上皮细胞，淋巴细胞，其他细胞的细胞数量分布矩阵
    Amount_epithelial, Amount_lymphocyte, Amount_other = get_nuclei_amount_region_matrix(pkl_file, svs_file, step)

    # 获取肿瘤，浸润，间质区域矩阵
    cancer_invasive_matrix, mask = get_cancer_invasive_matrix(svs_file, matrix_file, open_kernel_size, triangle_maxlen,
                                                              invasive_thickness)

    # 将肿瘤，浸润，间质区域矩阵缩放到细胞数量分布矩阵尺寸
    # cancer_invasive_matrix=cv2.resize(cancer_invasive_matrix,(AmountT.shape[1],AmountT.shape[0]),interpolation=cv2.INTER_NEAREST)
    cancer_invasive_matrix = OpenCV(cancer_invasive_matrix) \
        .resize(Amount_epithelial.shape[1], Amount_epithelial.shape[0], interpolation='INTER_NEAREST')
    # 缩放MASK区域矩阵
    # AreaM=cv2.resize(mask,(AmountT.shape[1],AmountT.shape[0]),cv2.INTER_NEAREST)
    AreaM = OpenCV(mask).resize(Amount_epithelial.shape[1], Amount_epithelial.shape[0], interpolation='INTER_NEAREST')
    # 如果细胞数量分布需要肿瘤区域，间质区域，以及浸润区域矩阵过滤
    if region_filter:
        # 将肿瘤区域，间质区域，以及浸润区域矩阵分别过滤所有类型细胞数量分布矩阵
        Amount_epithelial[cancer_invasive_matrix == 0] = 0
        Amount_lymphocyte[cancer_invasive_matrix == 0] = 0
        Amount_other[cancer_invasive_matrix == 0] = 0


    # 将区域矩阵分别分割为肿瘤区域，间质区域，以及浸润区域
    AreaT = np.where(cancer_invasive_matrix != 1, 0, cancer_invasive_matrix)
    AreaS = np.where(cancer_invasive_matrix != 2, 0, cancer_invasive_matrix)
    AreaI = np.where(cancer_invasive_matrix != 3, 0, cancer_invasive_matrix)

    # 将区域矩阵二值化处理
    AreaT[AreaT != 0] = 1
    AreaS[AreaS != 0] = 1
    AreaI[AreaI != 0] = 1

    return Amount_epithelial, Amount_lymphocyte, Amount_other, AreaT, AreaS, AreaI, AreaM


'''
获取热点矩阵
nuclei_amount_matrix：细胞数量矩阵
nr：热点邻域阶数
'''


def get_hotpot_matrix(nuclei_amount_matrix, nr):
    # 将邻域阶数转化为邻域尺寸
    nr_size = 2 * (nr + 1) - 1

    npy = np.copy(nuclei_amount_matrix)

    # 获取非0矩阵
    nonzero_npy = npy[npy != 0]

    n = nonzero_npy.size
    sumC = np.sum(nonzero_npy)
    sumCj2 = np.sum(nonzero_npy ** 2)

    # 求得C均值
    meanC = sumC / n

    # 求得S值
    S = math.sqrt(sumCj2 / n - pow(meanC, 2))

    # 设定卷积核，尺寸为领域域尺寸
    scharr = np.full((nr_size, nr_size), 1, dtype=np.float)

    # i区域为非邻域
    scharr[nr][nr] = 0

    # 获取非0则密度为1的矩阵
    onenpy = np.copy(npy)
    onenpy[onenpy != 0] = 1

    # 卷积邻域非0区域数量
    sumWij_npy = signal.convolve2d(onenpy, scharr, boundary='fill', mode='same')

    # 卷积邻域非0密度
    sumWijCj_npy = signal.convolve2d(npy, scharr, boundary='fill', mode='same')

    # 求得U矩阵
    U_npy = np.sqrt(np.floor(n * sumWij_npy - pow(sumWij_npy, 2)) / (n - 1))

    # 防止公式除0错误
    U_npy[U_npy == 0] = 1

    # 求得Zi矩阵
    Zi_npy = (sumWijCj_npy - meanC * sumWij_npy) / (S * U_npy)

    # 过滤密度为0区域
    Zi_npy *= onenpy

    # 二值化
    Zi_npy[Zi_npy <= 1.96] = 0
    Zi_npy[Zi_npy > 1.96] = 1

    return Zi_npy


'''
根据组织区域获取热点矩阵
mask：组织区域为1，非组织区域为0
nuclei_amount_matrix：细胞数量矩阵
nr：热点邻域阶数
'''


def get_hotpot_matrix_with_mask(mask, nuclei_amount_matrix, nr):
    # 过滤MASK区域面积占矩阵面积比率小于等于0.01的MASK区域
    mask = filter_smresult_contour_region(mask, 0.01)

    # 缩放MASK尺寸为nuclei_amount_matrix的尺寸
    # mask=cv2.resize(mask,(nuclei_amount_matrix.shape[1],nuclei_amount_matrix.shape[0]),cv2.INTER_NEAREST)
    mask = OpenCV(mask).resize(nuclei_amount_matrix.shape[1], nuclei_amount_matrix.shape[0],
                               interpolation='INTER_NEAREST')
    # 获取MASK轮廓外部
    # contours=get_contours(mask)
    # contours = get_contours(mask, cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 初始化所有组织区域热点矩阵
    result_matrix = np.full((nuclei_amount_matrix.shape[0], nuclei_amount_matrix.shape[1]), 0, dtype=np.float)
    # 遍历MASK轮廓，不同组织区域分别计算热点
    for cnt in contours:

        # 获取当前组织区域的细胞数量分布矩阵
        filter_matrix = fill_contours(mask.shape, [cnt])
        part_nuclei_amount_matrix = nuclei_amount_matrix * filter_matrix

        if len(np.unique(part_nuclei_amount_matrix)) > 2:
            # 获取当前组织区域的热点矩阵
            result_matrix += get_hotpot_matrix(part_nuclei_amount_matrix, nr)

    return result_matrix


'''
获取区域面积平均值及标准差
matrix：二值化矩阵
step：SVS图片0级下采样的区域步长
min_area：过滤区域最小面积
real_sample：实际尺寸缩放倍数
'''


def get_region_area_mean_std(matrix, step, min_area, real_sample):
    # 获取矩阵轮廓
    # contours=get_contours(matrix)
    # contours = get_contours(matrix,cv2.RETR_EXTERNAL)
    contours = OpenCV(matrix).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 过滤轮廓面积，并添加轮廓绝对面积到面积列表
    # area_filter_list = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > min_area:
    #         area_filter_list.append(area * real_sample ** 2)

    # 过滤轮廓面积，并添加轮廓绝对面积到面积列表
    area_list = map(cv2.contourArea,contours)
    area_filter_list = [area * real_sample ** 2 for area in area_list if area > min_area]
    # 计算所有面积平均值及标准差
    if len(area_filter_list) > 0:
        mean = np.mean(area_filter_list)
        std = np.std(area_filter_list)
    else:
        mean = np.nan
        std = np.nan

    return mean, std

'''
获取轮廓之间边的最短距离和最短边
contour1：轮廓1
contour2：轮廓2
'''


def get_contours_min_edge(contour_1, contour_2):
    contour_1 = contour_1[:, 0, :]
    contour_2 = contour_2[:, 0, :]

    # 获取轮廓坐标的笛卡儿积（交叉坐标对）
    contour_1_repeat = np.repeat(contour_1, contour_2.shape[0], axis=0)
    contour_2_tile = np.tile(contour_2, (contour_1.shape[0], 1))
    contour_direct = np.stack((contour_1_repeat, contour_2_tile), axis=1)

    # 计算坐标对的距离
    distances = np.sqrt(np.power(contour_direct[:, 0, 0] - contour_direct[:, 1, 0], 2) + np.power(
        contour_direct[:, 0, 1] - contour_direct[:, 1, 1], 2))

    # 获取最短距离
    min_disance = np.min(distances)

    # 获取最短距离索引
    min_index = np.argmin(distances)

    # 获取最短边的两个坐标点
    min_edge = [contour_direct[min_index, 0, :], contour_direct[min_index, 1, :]]

    return [min_disance, min_edge]


'''
获取热点区域与其相邻区域的最短距离和最短边
matrix：二值化矩阵
min_area：过滤区域最小面积
n：相邻区域最大数量
'''


def get_region_neighbor_edge(matrix, min_area, n):
    result_neighbor_edge_list = []

    # 提取轮廓
    # contours=get_contours(matrix,cv2.RETR_EXTERNAL)
    contours = OpenCV(matrix).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 过滤轮廓最小面积
    cnt_list = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cnt_list.append(cnt)
    # 获取所有区域的邻居最短距离和最短边
    for i in range(len(cnt_list)):
        neighbor_edge_list = []
        for j in range(len(cnt_list)):

            # 相同区域不处理
            if i == j:
                continue

            # 获取当前i轮廓和j轮廓的最短距离和最短边
            neighbor_edge = get_contours_min_edge(cnt_list[i], cnt_list[j])
            neighbor_edge_list.append(neighbor_edge)

        # 把i轮廓所有邻居的最短距离和最短边列表按从小到大排序
        neighbor_edge_list.sort(key=lambda x: x[0])

        # 添加列表前n个距离到所有区域邻居最短距离和最短边列表中
        result_neighbor_edge_list.append(neighbor_edge_list[0:n])

    return result_neighbor_edge_list


'''
获取热点区域与其相邻区域的最短距离和最短边,不同组织分别处理
matrix：二值化矩阵
mask: 组织区域为1，非组织区域为0
min_area：过滤区域最小面积
n：相邻区域最大数量
'''


def get_region_neighbor_with_mask(matrix, mask, min_area, n):
    result_neighbor_edge_list = []

    # 获取MASK轮廓
    # contours = get_contours(mask, cv2.RETR_EXTERNAL)
    contours = OpenCV(mask).find_contours(is_binary=True, mode='RETR_EXTERNAL')

    # 不同组织分别获取其对应的相邻区域距离，防止出现区域和相邻区域出现跨组织现象
    # for cnt in contours:
    #     # 获取当前组织区域
    #     filter_matrix = fill_contours(matrix.shape, [cnt])
    #     tissue_matrix = filter_matrix * matrix
    #
    #     # 获取当前组织热点区域与其相邻区域的距离
    #     region_neighbor_edge_list = get_region_neighbor_edge(tissue_matrix, min_area, n)
    #     result_neighbor_edge_list.extend(region_neighbor_edge_list)
    contours_len = len(contours)


    # 不同组织分别获取其对应的相邻区域距离，防止出现区域和相邻区域出现跨组织现象
    # 获取当前组织区域
    filter_matrix = map(fill_contours,[matrix.shape]*contours_len,[contours])
    tissue_matrix = map(lambda x,y: x* y,filter_matrix,[matrix]*contours_len)
    # 获取当前组织热点区域与其相邻区域的距离
    region_neighbor_edge_list = list(map(get_region_neighbor_edge,tissue_matrix,[min_area]*contours_len,[n]*contours_len))
    for i in region_neighbor_edge_list:
        result_neighbor_edge_list += i
    return result_neighbor_edge_list


'''
获取热点区域与其相邻区域的距离的平均值及标准差
matrix：二值化矩阵
mask：组织区域为1，非组织区域为0
step：SVS图片0级下采样的区域步长
min_area：过滤区域最小面积
n：相邻区域最大数量
real_sample：实际尺寸缩放倍数
'''


def get_region_neighbor_distance_mean_std(matrix, mask, step, min_area, n, real_sample):
    # 获取热点区域与其相邻区域的最短距离和最短边,不同组织分别处理
    result_neighbor_edge_list = get_region_neighbor_with_mask(matrix, mask, min_area, n)

    # 计算每个区域的相邻区域距离平均值及标准差
    mean_list = []
    std_list = []
    for neighbor_edge_list in result_neighbor_edge_list:
        if len(neighbor_edge_list) > 0:
            # 获取相邻区域的绝对距离列表
            distance_list = [edge_list[0] * real_sample for edge_list in neighbor_edge_list]

            # 求当前区域与相邻区域的距离平均值及标准差
            mean_list.append(np.mean(distance_list))
            std_list.append(np.std(distance_list))

    # 计算所有热点区域与其相邻区域的距离的平均值的平均值以及标准差的标准差
    if len(mean_list) > 0:
        mean = np.mean(mean_list)
        std = np.std(std_list)
    else:
        mean = np.nan
        std = np.nan

    return mean, std


"""
关于肿瘤细胞、成纤维细胞、以及淋巴细胞的热点区域、肿瘤区域、间质区域、浸润区域和细胞数量的统计计算
AmountT: 肿瘤细胞数量分布矩阵
AmountF: 成纤维细胞数量分布矩阵
AmountL: 淋巴细胞数量分布矩阵
AmountM: 混合细胞数量分布矩阵
AreaT: 肿瘤区域矩阵
AreaS: 间质区域矩阵
AreaI: 浸润区域矩阵
AreaM: MASK区域矩阵
step: SVS图片0级下采样的区域步长
nr: 热点邻域阶数
min_area: 热点面积过滤最小值
n：相邻区域最大数量
sample：SVS最大分辨率下一个像素的缩放倍数
"""


def get_statistics(AmountT, AmountF, AmountL, AmountM, AreaT, AreaS, AreaI, AreaM, step, nr, min_area, n, sample):
    # 除0错误忽略
    np.seterr(invalid='ignore')
    # 20190723mpp
    # 实际尺寸缩放倍数
    real_sample = step * sample

    # 癌变区域
    AreaTS = AreaT + AreaS

    # 其他区域
    AreaO = AreaM - AreaTS - AreaI

    # 所有细胞数量矩阵
    AmountA = AmountT + AmountF + AmountL + AmountM

    # HotS热点区域，HotS_T是指肿瘤热点区域
    HotS_T = get_hotpot_matrix(AmountT, nr)
    HotS_F = get_hotpot_matrix(AmountF, nr)
    HotS_L = get_hotpot_matrix(AmountL, nr)
    HotS_TL = HotS_T * HotS_L
    HotS_TF = HotS_T * HotS_F
    HotS_LF = HotS_L * HotS_F

    # AH区域热点重叠区域，AH_TT是指肿瘤区域与肿瘤热点区域的重叠区域
    AH_TT = AreaT * HotS_T
    AH_TL = AreaT * HotS_L
    AH_TF = AreaT * HotS_F
    AH_TTL = AreaT * HotS_TL
    AH_TTF = AreaT * HotS_TF
    AH_TLF = AreaT * HotS_LF

    AH_ST = AreaS * HotS_T
    AH_SL = AreaS * HotS_L
    AH_SF = AreaS * HotS_F
    AH_STL = AreaS * HotS_TL
    AH_STF = AreaS * HotS_TF
    AH_SLF = AreaS * HotS_LF

    AH_TST = AreaTS * HotS_T
    AH_TSL = AreaTS * HotS_L
    AH_TSF = AreaTS * HotS_F
    AH_TSTL = AreaTS * HotS_TL
    AH_TSTF = AreaTS * HotS_TF
    AH_TSLF = AreaTS * HotS_LF

    AH_IT = AreaI * HotS_T
    AH_IL = AreaI * HotS_L
    AH_IF = AreaI * HotS_F
    AH_ITL = AreaI * HotS_TL
    AH_ITF = AreaI * HotS_TF
    AH_ILF = AreaI * HotS_LF

    # 区域细胞数量，amount_AreaT_T是指肿瘤区域中的肿瘤细胞数量
    amount_AreaT_T = np.sum(AmountT * AreaT)
    amount_AreaT_L = np.sum(AmountL * AreaT)
    amount_AreaT_F = np.sum(AmountF * AreaT)
    amount_AreaT_A = np.sum(AmountA * AreaT)

    amount_AreaS_T = np.sum(AmountT * AreaS)
    amount_AreaS_L = np.sum(AmountL * AreaS)
    amount_AreaS_F = np.sum(AmountF * AreaS)
    amount_AreaS_A = np.sum(AmountA * AreaS)

    amount_AreaTS_T = np.sum(AmountT * AreaTS)
    amount_AreaTS_L = np.sum(AmountL * AreaTS)
    amount_AreaTS_F = np.sum(AmountF * AreaTS)
    amount_AreaTS_A = np.sum(AmountA * AreaTS)

    amount_AreaI_T = np.sum(AmountT * AreaI)
    amount_AreaI_L = np.sum(AmountL * AreaI)
    amount_AreaI_F = np.sum(AmountF * AreaI)
    amount_AreaI_A = np.sum(AmountA * AreaI)

    # 无调用
    # amount_AreaO_T = np.sum(AmountT * AreaO)
    amount_AreaO_L = np.sum(AmountL * AreaO)
    amount_AreaO_F = np.sum(AmountF * AreaO)
    # 无调用
    # amount_AreaO_A = np.sum(AmountA * AreaO)

    # 无调用
    # amount_AreaM_T = np.sum(AmountT * AreaM)
    amount_AreaM_L = np.sum(AmountL * AreaM)
    amount_AreaM_F = np.sum(AmountF * AreaM)
    # 无调用
    # amount_AreaM_A = np.sum(AmountA * AreaM)

    # 区域面积，area_AreaT是指肿瘤区域面积
    area_AreaT = np.sum(AreaT) * real_sample ** 2
    area_AreaS = np.sum(AreaS) * real_sample ** 2
    area_AreaTS = np.sum(AreaTS) * real_sample ** 2
    area_AreaI = np.sum(AreaI) * real_sample ** 2
    area_AreaM = np.sum(AreaM) * real_sample ** 2
    # 无调用
    # area_AreaO = np.sum(AreaO) * real_sample ** 2

    # 热点面积，area_HotS_T是指肿瘤热点区域面积
    area_HotS_T = np.sum(HotS_T) * real_sample ** 2
    area_HotS_F = np.sum(HotS_F) * real_sample ** 2
    area_HotS_L = np.sum(HotS_L) * real_sample ** 2
    area_HotS_TL = np.sum(HotS_TL) * real_sample ** 2
    area_HotS_TF = np.sum(HotS_TF) * real_sample ** 2
    area_HotS_LF = np.sum(HotS_LF) * real_sample ** 2

    # 区域热点重叠面积，area_AH_TT是指肿瘤区域与肿瘤热点区域的重叠区域面积
    area_AH_TT = np.sum(AH_TT) * real_sample ** 2
    area_AH_TL = np.sum(AH_TL) * real_sample ** 2
    area_AH_TF = np.sum(AH_TF) * real_sample ** 2
    area_AH_TTL = np.sum(AH_TTL) * real_sample ** 2
    area_AH_TTF = np.sum(AH_TTF) * real_sample ** 2
    area_AH_TLF = np.sum(AH_TLF) * real_sample ** 2

    area_AH_ST = np.sum(AH_ST) * real_sample ** 2
    area_AH_SL = np.sum(AH_SL) * real_sample ** 2
    area_AH_SF = np.sum(AH_SF) * real_sample ** 2
    area_AH_STL = np.sum(AH_STL) * real_sample ** 2
    area_AH_STF = np.sum(AH_STF) * real_sample ** 2
    area_AH_SLF = np.sum(AH_SLF) * real_sample ** 2

    area_AH_TST = np.sum(AH_TST) * real_sample ** 2
    area_AH_TSL = np.sum(AH_TSL) * real_sample ** 2
    area_AH_TSF = np.sum(AH_TSF) * real_sample ** 2
    area_AH_TSTL = np.sum(AH_TSTL) * real_sample ** 2
    area_AH_TSTF = np.sum(AH_TSTF) * real_sample ** 2
    area_AH_TSLF = np.sum(AH_TSLF) * real_sample ** 2

    area_AH_IT = np.sum(AH_IT) * real_sample ** 2
    area_AH_IL = np.sum(AH_IL) * real_sample ** 2
    area_AH_IF = np.sum(AH_IF) * real_sample ** 2
    area_AH_ITL = np.sum(AH_ITL) * real_sample ** 2
    area_AH_ITF = np.sum(AH_ITF) * real_sample ** 2
    area_AH_ILF = np.sum(AH_ILF) * real_sample ** 2

    # 面积比率，rate_area_AreaT_AreaM是值AreaT面积占Mask面积比率
    rate_area_AreaT_AreaM = area_AreaT / area_AreaM
    rate_area_AreaS_AreaM = area_AreaS / area_AreaM
    rate_area_AreaTS_AreaM = area_AreaTS / area_AreaM
    rate_area_AreaI_AreaM = area_AreaI / area_AreaM
    rate_area_AreaT_AreaS = area_AreaT / area_AreaS
    rate_area_AreaS_AreaTS = area_AreaS / area_AreaTS

    rate_area_HotS_T_AreaM = area_HotS_T / area_AreaM
    rate_area_HotS_L_AreaM = area_HotS_L / area_AreaM
    rate_area_HotS_F_AreaM = area_HotS_F / area_AreaM
    rate_area_HotS_TL_AreaM = area_HotS_TL / area_AreaM
    rate_area_HotS_TF_AreaM = area_HotS_TF / area_AreaM
    rate_area_HotS_LF_AreaM = area_HotS_LF / area_AreaM

    rate_area_HotS_TL_HotS_T = area_HotS_TL / area_HotS_T
    rate_area_HotS_TL_HotS_L = area_HotS_TL / area_HotS_L
    rate_area_HotS_TF_HotS_T = area_HotS_TF / area_HotS_T
    rate_area_HotS_TF_HotS_F = area_HotS_TF / area_HotS_F
    rate_area_HotS_LF_HotS_L = area_HotS_LF / area_HotS_L
    rate_area_HotS_LF_HotS_F = area_HotS_LF / area_HotS_F

    rate_area_AH_TT_AreaT = area_AH_TT / area_AreaT
    rate_area_AH_TL_AreaT = area_AH_TL / area_AreaT
    rate_area_AH_TF_AreaT = area_AH_TF / area_AreaT
    rate_area_AH_TTL_AreaT = area_AH_TTL / area_AreaT
    rate_area_AH_TTF_AreaT = area_AH_TTF / area_AreaT
    rate_area_AH_TLF_AreaT = area_AH_TLF / area_AreaT

    rate_area_AH_ST_AreaS = area_AH_ST / area_AreaS
    rate_area_AH_SL_AreaS = area_AH_SL / area_AreaS
    rate_area_AH_SF_AreaS = area_AH_SF / area_AreaS
    rate_area_AH_STL_AreaS = area_AH_STL / area_AreaS
    rate_area_AH_STF_AreaS = area_AH_STF / area_AreaS
    rate_area_AH_SLF_AreaS = area_AH_SLF / area_AreaS

    rate_area_AH_TST_AreaTS = area_AH_TST / area_AreaTS
    rate_area_AH_TSL_AreaTS = area_AH_TSL / area_AreaTS
    rate_area_AH_TSF_AreaTS = area_AH_TSF / area_AreaTS
    rate_area_AH_TSTL_AreaTS = area_AH_TSTL / area_AreaTS
    rate_area_AH_TSTF_AreaTS = area_AH_TSTF / area_AreaTS
    rate_area_AH_TSLF_AreaTS = area_AH_TSLF / area_AreaTS

    rate_area_AH_IT_AreaI = area_AH_IT / area_AreaI
    rate_area_AH_IL_AreaI = area_AH_IL / area_AreaI
    rate_area_AH_IF_AreaI = area_AH_IF / area_AreaI
    rate_area_AH_ITL_AreaI = area_AH_ITL / area_AreaI
    rate_area_AH_ITF_AreaI = area_AH_ITF / area_AreaI
    rate_area_AH_ILF_AreaI = area_AH_ILF / area_AreaI

    rate_area_AH_TT_HotS_T = area_AH_TT / area_HotS_T
    rate_area_AH_TL_HotS_L = area_AH_TL / area_HotS_L
    rate_area_AH_TF_HotS_F = area_AH_TF / area_HotS_F
    rate_area_AH_TTL_HotS_TL = area_AH_TTL / area_HotS_TL
    rate_area_AH_TTF_HotS_TF = area_AH_TTF / area_HotS_TF
    rate_area_AH_TLF_HotS_LF = area_AH_TLF / area_HotS_LF

    rate_area_AH_ST_HotS_T = area_AH_ST / area_HotS_T
    rate_area_AH_SL_HotS_L = area_AH_SL / area_HotS_L
    rate_area_AH_SF_HotS_F = area_AH_SF / area_HotS_F
    rate_area_AH_STL_HotS_TL = area_AH_STL / area_HotS_TL
    rate_area_AH_STF_HotS_TF = area_AH_STF / area_HotS_TF
    rate_area_AH_SLF_HotS_LF = area_AH_SLF / area_HotS_LF

    rate_area_AH_TST_HotS_T = area_AH_TST / area_HotS_T
    rate_area_AH_TSL_HotS_L = area_AH_TSL / area_HotS_L
    rate_area_AH_TSF_HotS_F = area_AH_TSF / area_HotS_F
    rate_area_AH_TSTL_HotS_TL = area_AH_TSTL / area_HotS_TL
    rate_area_AH_TSTF_HotS_TF = area_AH_TSTF / area_HotS_TF
    rate_area_AH_TSLF_HotS_LF = area_AH_TSLF / area_HotS_LF

    rate_area_AH_IT_HotS_T = area_AH_IT / area_HotS_T
    rate_area_AH_IL_HotS_L = area_AH_IL / area_HotS_L
    rate_area_AH_IF_HotS_F = area_AH_IF / area_HotS_F
    rate_area_AH_ITL_HotS_TL = area_AH_ITL / area_HotS_TL
    rate_area_AH_ITF_HotS_TF = area_AH_ITF / area_HotS_TF
    rate_area_AH_ILF_HotS_LF = area_AH_ILF / area_HotS_LF

    # 细胞密度，density_amount_AreaT_T是指AreaT中的肿瘤细胞密度
    density_amount_AreaT_T = amount_AreaT_T / area_AreaT
    density_amount_AreaT_L = amount_AreaT_L / area_AreaT
    density_amount_AreaT_F = amount_AreaT_F / area_AreaT

    density_amount_AreaS_T = amount_AreaS_T / area_AreaS
    density_amount_AreaS_L = amount_AreaS_L / area_AreaS
    density_amount_AreaS_F = amount_AreaS_F / area_AreaS

    density_amount_AreaTS_T = amount_AreaTS_T / area_AreaTS
    density_amount_AreaTS_L = amount_AreaTS_L / area_AreaTS
    density_amount_AreaTS_F = amount_AreaTS_F / area_AreaTS

    density_amount_AreaI_T = amount_AreaI_T / area_AreaI
    density_amount_AreaI_L = amount_AreaI_L / area_AreaI
    density_amount_AreaI_F = amount_AreaI_F / area_AreaI

    # 细胞数目比率，rate_amount_AreaT_T_AreaT_A是指AreaT中肿瘤细胞数目占AreaT中所有细胞数目的比率
    rate_amount_AreaT_T_AreaT_A = amount_AreaT_T / amount_AreaT_A
    rate_amount_AreaT_L_AreaT_A = amount_AreaT_L / amount_AreaT_A
    rate_amount_AreaT_F_AreaT_A = amount_AreaT_F / amount_AreaT_A

    rate_amount_AreaS_T_AreaS_A = amount_AreaS_T / amount_AreaS_A
    rate_amount_AreaS_L_AreaS_A = amount_AreaS_L / amount_AreaS_A
    rate_amount_AreaS_F_AreaS_A = amount_AreaS_F / amount_AreaS_A

    rate_amount_AreaTS_T_AreaTS_A = amount_AreaTS_T / amount_AreaTS_A
    rate_amount_AreaTS_L_AreaTS_A = amount_AreaTS_L / amount_AreaTS_A
    rate_amount_AreaTS_F_AreaTS_A = amount_AreaTS_F / amount_AreaTS_A

    rate_amount_AreaI_T_AreaI_A = amount_AreaI_T / amount_AreaI_A
    rate_amount_AreaI_L_AreaI_A = amount_AreaI_L / amount_AreaI_A
    rate_amount_AreaI_F_AreaI_A = amount_AreaI_F / amount_AreaI_A

    rate_amount_AreaT_L_AreaT_T = amount_AreaT_L / amount_AreaT_T
    rate_amount_AreaT_F_AreaT_T = amount_AreaT_F / amount_AreaT_T
    rate_amount_AreaT_F_AreaT_L = amount_AreaT_F / amount_AreaT_L

    rate_amount_AreaS_L_AreaS_T = amount_AreaS_L / amount_AreaS_T
    rate_amount_AreaS_F_AreaS_T = amount_AreaS_F / amount_AreaS_T
    rate_amount_AreaS_F_AreaS_L = amount_AreaS_F / amount_AreaS_L

    rate_amount_AreaTS_L_AreaTS_T = amount_AreaTS_L / amount_AreaTS_T
    rate_amount_AreaTS_F_AreaTS_T = amount_AreaTS_F / amount_AreaTS_T
    rate_amount_AreaTS_F_AreaTS_L = amount_AreaTS_F / amount_AreaTS_L

    rate_amount_AreaI_L_AreaI_T = amount_AreaI_L / amount_AreaI_T
    rate_amount_AreaI_F_AreaI_T = amount_AreaI_F / amount_AreaI_T
    rate_amount_AreaI_F_AreaI_L = amount_AreaI_F / amount_AreaI_L

    rate_amount_AreaTS_L_AreaM_L = amount_AreaTS_L / amount_AreaM_L
    rate_amount_AreaTS_F_AreaM_F = amount_AreaTS_F / amount_AreaM_F

    rate_amount_AreaI_L_AreaM_L = amount_AreaI_L / amount_AreaM_L
    rate_amount_AreaI_F_AreaM_F = amount_AreaI_F / amount_AreaM_F

    rate_amount_AreaO_L_AreaM_L = amount_AreaO_L / amount_AreaM_L
    rate_amount_AreaO_F_AreaM_F = amount_AreaO_F / amount_AreaM_F

    # 求面积平均值及标准差，mean_area_AH_TT,std_area_AH_TT是指AreaT与HotS_T重叠部分的热点区域的面积的平均值和标准差
    mean_area_AH_TT, std_area_AH_TT = get_region_area_mean_std(AH_TT, step, min_area, real_sample)
    mean_area_AH_TL, std_area_AH_TL = get_region_area_mean_std(AH_TL, step, min_area, real_sample)
    mean_area_AH_TF, std_area_AH_TF = get_region_area_mean_std(AH_TF, step, min_area, real_sample)
    mean_area_AH_TTL, std_area_AH_TTL = get_region_area_mean_std(AH_TTL, step, min_area, real_sample)
    mean_area_AH_TTF, std_area_AH_TTF = get_region_area_mean_std(AH_TTF, step, min_area, real_sample)
    mean_area_AH_TLF, std_area_AH_TLF = get_region_area_mean_std(AH_TLF, step, min_area, real_sample)

    mean_area_AH_ST, std_area_AH_ST = get_region_area_mean_std(AH_ST, step, min_area, real_sample)
    mean_area_AH_SL, std_area_AH_SL = get_region_area_mean_std(AH_SL, step, min_area, real_sample)
    mean_area_AH_SF, std_area_AH_SF = get_region_area_mean_std(AH_SF, step, min_area, real_sample)
    mean_area_AH_STL, std_area_AH_STL = get_region_area_mean_std(AH_STL, step, min_area, real_sample)
    mean_area_AH_STF, std_area_AH_STF = get_region_area_mean_std(AH_STF, step, min_area, real_sample)
    mean_area_AH_SLF, std_area_AH_SLF = get_region_area_mean_std(AH_SLF, step, min_area, real_sample)

    mean_area_AH_TST, std_area_AH_TST = get_region_area_mean_std(AH_TST, step, min_area, real_sample)
    mean_area_AH_TSL, std_area_AH_TSL = get_region_area_mean_std(AH_TSL, step, min_area, real_sample)
    mean_area_AH_TSF, std_area_AH_TSF = get_region_area_mean_std(AH_TSF, step, min_area, real_sample)
    mean_area_AH_TSTL, std_area_AH_TSTL = get_region_area_mean_std(AH_TSTL, step, min_area, real_sample)
    mean_area_AH_TSTF, std_area_AH_TSTF = get_region_area_mean_std(AH_TSTF, step, min_area, real_sample)
    mean_area_AH_TSLF, std_area_AH_TSLF = get_region_area_mean_std(AH_TSLF, step, min_area, real_sample)

    mean_area_AH_IT, std_area_AH_IT = get_region_area_mean_std(AH_IT, step, min_area, real_sample)
    mean_area_AH_IL, std_area_AH_IL = get_region_area_mean_std(AH_IL, step, min_area, real_sample)
    mean_area_AH_IF, std_area_AH_IF = get_region_area_mean_std(AH_IF, step, min_area, real_sample)
    mean_area_AH_ITL, std_area_AH_ITL = get_region_area_mean_std(AH_ITL, step, min_area, real_sample)
    mean_area_AH_ITF, std_area_AH_ITF = get_region_area_mean_std(AH_ITF, step, min_area, real_sample)
    mean_area_AH_ILF, std_area_AH_ILF = get_region_area_mean_std(AH_ILF, step, min_area, real_sample)

    # 求最近邻域距离平均值及标准差，mean_distance_AH_TT,std_distance_AH_TT是指AreaT与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值和标准差
    mean_distance_AH_TT, std_distance_AH_TT = get_region_neighbor_distance_mean_std(AH_TT, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_TL, std_distance_AH_TL = get_region_neighbor_distance_mean_std(AH_TL, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_TF, std_distance_AH_TF = get_region_neighbor_distance_mean_std(AH_TF, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_TTL, std_distance_AH_TTL = get_region_neighbor_distance_mean_std(AH_TTL, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_TTF, std_distance_AH_TTF = get_region_neighbor_distance_mean_std(AH_TTF, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_TLF, std_distance_AH_TLF = get_region_neighbor_distance_mean_std(AH_TLF, AreaM, step, min_area, n,
                                                                                      real_sample)

    mean_distance_AH_ST, std_distance_AH_ST = get_region_neighbor_distance_mean_std(AH_ST, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_SL, std_distance_AH_SL = get_region_neighbor_distance_mean_std(AH_SL, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_SF, std_distance_AH_SF = get_region_neighbor_distance_mean_std(AH_SF, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_STL, std_distance_AH_STL = get_region_neighbor_distance_mean_std(AH_STL, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_STF, std_distance_AH_STF = get_region_neighbor_distance_mean_std(AH_STF, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_SLF, std_distance_AH_SLF = get_region_neighbor_distance_mean_std(AH_SLF, AreaM, step, min_area, n,
                                                                                      real_sample)

    mean_distance_AH_TST, std_distance_AH_TST = get_region_neighbor_distance_mean_std(AH_TST, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_TSL, std_distance_AH_TSL = get_region_neighbor_distance_mean_std(AH_TSL, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_TSF, std_distance_AH_TSF = get_region_neighbor_distance_mean_std(AH_TSF, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_TSTL, std_distance_AH_TSTL = get_region_neighbor_distance_mean_std(AH_TSTL, AreaM, step, min_area,
                                                                                        n, real_sample)
    mean_distance_AH_TSTF, std_distance_AH_TSTF = get_region_neighbor_distance_mean_std(AH_TSTF, AreaM, step, min_area,
                                                                                        n, real_sample)
    mean_distance_AH_TSLF, std_distance_AH_TSLF = get_region_neighbor_distance_mean_std(AH_TSLF, AreaM, step, min_area,
                                                                                        n, real_sample)

    mean_distance_AH_IT, std_distance_AH_IT = get_region_neighbor_distance_mean_std(AH_IT, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_IL, std_distance_AH_IL = get_region_neighbor_distance_mean_std(AH_IL, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_IF, std_distance_AH_IF = get_region_neighbor_distance_mean_std(AH_IF, AreaM, step, min_area, n,
                                                                                    real_sample)
    mean_distance_AH_ITL, std_distance_AH_ITL = get_region_neighbor_distance_mean_std(AH_ITL, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_ITF, std_distance_AH_ITF = get_region_neighbor_distance_mean_std(AH_ITF, AreaM, step, min_area, n,
                                                                                      real_sample)
    mean_distance_AH_ILF, std_distance_AH_ILF = get_region_neighbor_distance_mean_std(AH_ILF, AreaM, step, min_area, n,
                                                                                      real_sample)

    statistic_list = []
    statistic_list.append(rate_area_AreaT_AreaM)
    statistic_list.append(rate_area_AreaS_AreaM)
    statistic_list.append(rate_area_AreaTS_AreaM)
    statistic_list.append(rate_area_AreaI_AreaM)
    statistic_list.append(rate_area_AreaT_AreaS)
    statistic_list.append(rate_area_AreaS_AreaTS)
    statistic_list.append(density_amount_AreaT_T)
    statistic_list.append(density_amount_AreaT_L)
    statistic_list.append(density_amount_AreaT_F)
    statistic_list.append(density_amount_AreaS_T)
    statistic_list.append(density_amount_AreaS_L)
    statistic_list.append(density_amount_AreaS_F)
    statistic_list.append(density_amount_AreaTS_T)
    statistic_list.append(density_amount_AreaTS_L)
    statistic_list.append(density_amount_AreaTS_F)
    statistic_list.append(density_amount_AreaI_T)
    statistic_list.append(density_amount_AreaI_L)
    statistic_list.append(density_amount_AreaI_F)
    statistic_list.append(rate_amount_AreaT_T_AreaT_A)
    statistic_list.append(rate_amount_AreaS_T_AreaS_A)
    statistic_list.append(rate_amount_AreaTS_T_AreaTS_A)
    statistic_list.append(rate_amount_AreaI_T_AreaI_A)
    statistic_list.append(rate_amount_AreaT_L_AreaT_A)
    statistic_list.append(rate_amount_AreaS_L_AreaS_A)
    statistic_list.append(rate_amount_AreaTS_L_AreaTS_A)
    statistic_list.append(rate_amount_AreaI_L_AreaI_A)
    statistic_list.append(rate_amount_AreaT_F_AreaT_A)
    statistic_list.append(rate_amount_AreaS_F_AreaS_A)
    statistic_list.append(rate_amount_AreaTS_F_AreaTS_A)
    statistic_list.append(rate_amount_AreaI_F_AreaI_A)
    statistic_list.append(rate_amount_AreaT_L_AreaT_T)
    statistic_list.append(rate_amount_AreaT_F_AreaT_T)
    statistic_list.append(rate_amount_AreaT_F_AreaT_L)
    statistic_list.append(rate_amount_AreaS_L_AreaS_T)
    statistic_list.append(rate_amount_AreaS_F_AreaS_T)
    statistic_list.append(rate_amount_AreaS_F_AreaS_L)
    statistic_list.append(rate_amount_AreaTS_L_AreaTS_T)
    statistic_list.append(rate_amount_AreaTS_F_AreaTS_T)
    statistic_list.append(rate_amount_AreaTS_F_AreaTS_L)
    statistic_list.append(rate_amount_AreaI_L_AreaI_T)
    statistic_list.append(rate_amount_AreaI_F_AreaI_T)
    statistic_list.append(rate_amount_AreaI_F_AreaI_L)
    statistic_list.append(rate_amount_AreaTS_L_AreaM_L)
    statistic_list.append(rate_amount_AreaI_L_AreaM_L)
    statistic_list.append(rate_amount_AreaO_L_AreaM_L)
    statistic_list.append(rate_amount_AreaTS_F_AreaM_F)
    statistic_list.append(rate_amount_AreaI_F_AreaM_F)
    statistic_list.append(rate_amount_AreaO_F_AreaM_F)
    statistic_list.append(rate_area_HotS_T_AreaM)
    statistic_list.append(rate_area_HotS_L_AreaM)
    statistic_list.append(rate_area_HotS_F_AreaM)
    statistic_list.append(rate_area_HotS_TL_AreaM)
    statistic_list.append(rate_area_HotS_TF_AreaM)
    statistic_list.append(rate_area_HotS_LF_AreaM)
    statistic_list.append(rate_area_HotS_TL_HotS_T)
    statistic_list.append(rate_area_HotS_TL_HotS_L)
    statistic_list.append(rate_area_HotS_TF_HotS_T)
    statistic_list.append(rate_area_HotS_TF_HotS_F)
    statistic_list.append(rate_area_HotS_LF_HotS_L)
    statistic_list.append(rate_area_HotS_LF_HotS_F)
    statistic_list.append(rate_area_AH_TT_AreaT)
    statistic_list.append(rate_area_AH_TT_HotS_T)
    statistic_list.append(rate_area_AH_TL_AreaT)
    statistic_list.append(rate_area_AH_TL_HotS_L)
    statistic_list.append(rate_area_AH_TF_AreaT)
    statistic_list.append(rate_area_AH_TF_HotS_F)
    statistic_list.append(rate_area_AH_ST_AreaS)
    statistic_list.append(rate_area_AH_ST_HotS_T)
    statistic_list.append(rate_area_AH_SL_AreaS)
    statistic_list.append(rate_area_AH_SL_HotS_L)
    statistic_list.append(rate_area_AH_SF_AreaS)
    statistic_list.append(rate_area_AH_SF_HotS_F)
    statistic_list.append(rate_area_AH_TST_AreaTS)
    statistic_list.append(rate_area_AH_TST_HotS_T)
    statistic_list.append(rate_area_AH_TSL_AreaTS)
    statistic_list.append(rate_area_AH_TSL_HotS_L)
    statistic_list.append(rate_area_AH_TSF_AreaTS)
    statistic_list.append(rate_area_AH_TSF_HotS_F)
    statistic_list.append(rate_area_AH_IT_AreaI)
    statistic_list.append(rate_area_AH_IT_HotS_T)
    statistic_list.append(rate_area_AH_IL_AreaI)
    statistic_list.append(rate_area_AH_IL_HotS_L)
    statistic_list.append(rate_area_AH_IF_AreaI)
    statistic_list.append(rate_area_AH_IF_HotS_F)
    statistic_list.append(rate_area_AH_TTL_AreaT)
    statistic_list.append(rate_area_AH_TTL_HotS_TL)
    statistic_list.append(rate_area_AH_TTF_AreaT)
    statistic_list.append(rate_area_AH_TTF_HotS_TF)
    statistic_list.append(rate_area_AH_TLF_AreaT)
    statistic_list.append(rate_area_AH_TLF_HotS_LF)
    statistic_list.append(rate_area_AH_STL_AreaS)
    statistic_list.append(rate_area_AH_STL_HotS_TL)
    statistic_list.append(rate_area_AH_STF_AreaS)
    statistic_list.append(rate_area_AH_STF_HotS_TF)
    statistic_list.append(rate_area_AH_SLF_AreaS)
    statistic_list.append(rate_area_AH_SLF_HotS_LF)
    statistic_list.append(rate_area_AH_TSTL_AreaTS)
    statistic_list.append(rate_area_AH_TSTL_HotS_TL)
    statistic_list.append(rate_area_AH_TSTF_AreaTS)
    statistic_list.append(rate_area_AH_TSTF_HotS_TF)
    statistic_list.append(rate_area_AH_TSLF_AreaTS)
    statistic_list.append(rate_area_AH_TSLF_HotS_LF)
    statistic_list.append(rate_area_AH_ITL_AreaI)
    statistic_list.append(rate_area_AH_ITL_HotS_TL)
    statistic_list.append(rate_area_AH_ITF_AreaI)
    statistic_list.append(rate_area_AH_ITF_HotS_TF)
    statistic_list.append(rate_area_AH_ILF_AreaI)
    statistic_list.append(rate_area_AH_ILF_HotS_LF)
    statistic_list.append(mean_area_AH_TT)
    statistic_list.append(std_area_AH_TT)
    statistic_list.append(mean_area_AH_TL)
    statistic_list.append(std_area_AH_TL)
    statistic_list.append(mean_area_AH_TF)
    statistic_list.append(std_area_AH_TF)
    statistic_list.append(mean_area_AH_TTL)
    statistic_list.append(std_area_AH_TTL)
    statistic_list.append(mean_area_AH_TTF)
    statistic_list.append(std_area_AH_TTF)
    statistic_list.append(mean_area_AH_TLF)
    statistic_list.append(std_area_AH_TLF)
    statistic_list.append(mean_area_AH_ST)
    statistic_list.append(std_area_AH_ST)
    statistic_list.append(mean_area_AH_SL)
    statistic_list.append(std_area_AH_SL)
    statistic_list.append(mean_area_AH_SF)
    statistic_list.append(std_area_AH_SF)
    statistic_list.append(mean_area_AH_STL)
    statistic_list.append(std_area_AH_STL)
    statistic_list.append(mean_area_AH_STF)
    statistic_list.append(std_area_AH_STF)
    statistic_list.append(mean_area_AH_SLF)
    statistic_list.append(std_area_AH_SLF)
    statistic_list.append(mean_area_AH_TST)
    statistic_list.append(std_area_AH_TST)
    statistic_list.append(mean_area_AH_TSL)
    statistic_list.append(std_area_AH_TSL)
    statistic_list.append(mean_area_AH_TSF)
    statistic_list.append(std_area_AH_TSF)
    statistic_list.append(mean_area_AH_TSTL)
    statistic_list.append(std_area_AH_TSTL)
    statistic_list.append(mean_area_AH_TSTF)
    statistic_list.append(std_area_AH_TSTF)
    statistic_list.append(mean_area_AH_TSLF)
    statistic_list.append(std_area_AH_TSLF)
    statistic_list.append(mean_area_AH_IT)
    statistic_list.append(std_area_AH_IT)
    statistic_list.append(mean_area_AH_IL)
    statistic_list.append(std_area_AH_IL)
    statistic_list.append(mean_area_AH_IF)
    statistic_list.append(std_area_AH_IF)
    statistic_list.append(mean_area_AH_ITL)
    statistic_list.append(std_area_AH_ITL)
    statistic_list.append(mean_area_AH_ITF)
    statistic_list.append(std_area_AH_ITF)
    statistic_list.append(mean_area_AH_ILF)
    statistic_list.append(std_area_AH_ILF)
    statistic_list.append(mean_distance_AH_TT)
    statistic_list.append(std_distance_AH_TT)
    statistic_list.append(mean_distance_AH_TL)
    statistic_list.append(std_distance_AH_TL)
    statistic_list.append(mean_distance_AH_TF)
    statistic_list.append(std_distance_AH_TF)
    statistic_list.append(mean_distance_AH_TTL)
    statistic_list.append(std_distance_AH_TTL)
    statistic_list.append(mean_distance_AH_TTF)
    statistic_list.append(std_distance_AH_TTF)
    statistic_list.append(mean_distance_AH_TLF)
    statistic_list.append(std_distance_AH_TLF)
    statistic_list.append(mean_distance_AH_ST)
    statistic_list.append(std_distance_AH_ST)
    statistic_list.append(mean_distance_AH_SL)
    statistic_list.append(std_distance_AH_SL)
    statistic_list.append(mean_distance_AH_SF)
    statistic_list.append(std_distance_AH_SF)
    statistic_list.append(mean_distance_AH_STL)
    statistic_list.append(std_distance_AH_STL)
    statistic_list.append(mean_distance_AH_STF)
    statistic_list.append(std_distance_AH_STF)
    statistic_list.append(mean_distance_AH_SLF)
    statistic_list.append(std_distance_AH_SLF)
    statistic_list.append(mean_distance_AH_TST)
    statistic_list.append(std_distance_AH_TST)
    statistic_list.append(mean_distance_AH_TSL)
    statistic_list.append(std_distance_AH_TSL)
    statistic_list.append(mean_distance_AH_TSF)
    statistic_list.append(std_distance_AH_TSF)
    statistic_list.append(mean_distance_AH_TSTL)
    statistic_list.append(std_distance_AH_TSTL)
    statistic_list.append(mean_distance_AH_TSTF)
    statistic_list.append(std_distance_AH_TSTF)
    statistic_list.append(mean_distance_AH_TSLF)
    statistic_list.append(std_distance_AH_TSLF)
    statistic_list.append(mean_distance_AH_IT)
    statistic_list.append(std_distance_AH_IT)
    statistic_list.append(mean_distance_AH_IL)
    statistic_list.append(std_distance_AH_IL)
    statistic_list.append(mean_distance_AH_IF)
    statistic_list.append(std_distance_AH_IF)
    statistic_list.append(mean_distance_AH_ITL)
    statistic_list.append(std_distance_AH_ITL)
    statistic_list.append(mean_distance_AH_ITF)
    statistic_list.append(std_distance_AH_ITF)
    statistic_list.append(mean_distance_AH_ILF)
    statistic_list.append(std_distance_AH_ILF)

    return statistic_list


'''
获取统计结果
pkl_file: PKL文件路径
svs_file: SVS文件路径
matrix_file: 九分类矩阵文件路径
step: SVS图片0级下采样的区域步长
nr: 热点邻域阶数
min_area: 热点面积过滤最小值
n：相邻区域最大数量
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 癌症浸润区域的厚度
sample：SVS最大分辨率下一个像素的缩放倍数
region_filter：若为1，细胞数量分布需要经过肿瘤区域，间质区域，以及浸润区域矩阵过滤，为0则不过滤
'''

# 0715
def get_nuclei_statistics(pkl_file, svs_file, matrix_file, step, nr, min_area, n, open_kernel_size, triangle_maxlen,
                          invasive_thickness, sample, region_filter):
    # 获取肿瘤细胞，成纤维细胞，淋巴细胞，混合细胞的细胞数量分布矩阵   肿瘤， 浸润，间质、组织区域矩阵
    AmountT, AmountF, AmountL, AmountM, AreaT, AreaS, AreaI, AreaM = get_nuclei_invasive_region_matrix(pkl_file,
                                                                                                       svs_file, step, \
                                                                                                       matrix_file,
                                                                                                       open_kernel_size, \
                                                                                                       triangle_maxlen,
                                                                                                       invasive_thickness,
                                                                                                       region_filter)

    # 获取MASK轮廓，并将轮廓列表顺序按轮廓质心坐标从左到右，从上到下排序
    contours = get_contours_by_sort(AreaM)

    # 组织区域统计结果列表
    tissue_statistics_lst = []

    # 遍历组织区域轮廓
    for cnt in contours:

        # 获取当前组织区域
        tissue_region = fill_contours(AreaM.shape, [cnt])

        # 获取当前组织区域的细胞数量分布矩阵以及区域矩阵
        AmountT_t = AmountT * tissue_region
        AmountF_t = AmountF * tissue_region
        AmountL_t = AmountL * tissue_region
        AmountM_t = AmountM * tissue_region
        AreaT_t = AreaT * tissue_region
        AreaS_t = AreaS * tissue_region
        AreaI_t = AreaI * tissue_region
        AreaM_t = AreaM * tissue_region

        # 只有当组织区域中包含肿瘤区域才计算当前组织区域的统计结果
        if np.sum(AreaT_t) > 0:
            # 获取当前组织区域的统计结果
            statistics_lst = get_statistics(AmountT_t, AmountF_t, AmountL_t, AmountM_t, AreaT_t, AreaS_t, AreaI_t,
                                            AreaM_t, step, nr, min_area, n, sample)

            # 添加当前组织区域的统计结果到组织区域统计结果列表中
            tissue_statistics_lst.append(statistics_lst)

    return tissue_statistics_lst


'''
获取指标的列名
'''


def GetTitle():
    title_list = []
    title_list.append("AreaT面积占组织Mask面积比率")
    title_list.append("AreaS面积占组织Mask面积比率")
    title_list.append("AreaTS面积占组织Mask面积比率")
    title_list.append("AreaI面积占组织Mask面积比率")
    title_list.append("AreaT面积占AreaS面积比率")
    title_list.append("AreaS面积占AreaTS面积比率")
    title_list.append("AreaT中的肿瘤细胞密度")
    title_list.append("AreaT中的淋巴细胞密度")
    title_list.append("AreaT中的成纤维细胞密度")
    title_list.append("AreaS中的肿瘤细胞密度")
    title_list.append("AreaS中的淋巴细胞密度")
    title_list.append("AreaS中的成纤维细胞密度")
    title_list.append("AreaTS中的肿瘤细胞密度")
    title_list.append("AreaTS中的淋巴细胞密度")
    title_list.append("AreaTS中的成纤维细胞密度")
    title_list.append("AreaI中的肿瘤细胞密度")
    title_list.append("AreaI中的淋巴细胞密度")
    title_list.append("AreaI中的成纤维细胞密度")
    title_list.append("AreaT中肿瘤细胞数目占AreaT中所有细胞数目的比率")
    title_list.append("AreaS中肿瘤细胞数目占AreaS中所有细胞数目的比率")
    title_list.append("AreaTS中肿瘤细胞数目占AreaTS中所有细胞数目的比率")
    title_list.append("AreaI中肿瘤细胞数目占AreaI中所有细胞数目的比率")
    title_list.append("AreaT中淋巴细胞数目占AreaT中所有细胞数目的比率")
    title_list.append("AreaS中淋巴细胞数目占AreaS中所有细胞数目的比率")
    title_list.append("AreaTS中淋巴细胞数目占AreaTS中所有细胞数目的比率")
    title_list.append("AreaI中淋巴细胞数目占AreaI中所有细胞数目的比率")
    title_list.append("AreaT中成纤维细胞数目占AreaT中所有细胞数目的比率")
    title_list.append("AreaS中成纤维细胞数目占AreaS中所有细胞数目的比率")
    title_list.append("AreaTS中成纤维细胞数目占AreaTS中所有细胞数目的比率")
    title_list.append("AreaI中成纤维细胞数目占AreaI中所有细胞数目的比率")
    title_list.append("AreaT中淋巴细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaT中成纤维细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaT中成纤维细胞数目与淋巴细胞数目的比率")
    title_list.append("AreaS中淋巴细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaS中成纤维细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaS中成纤维细胞数目与淋巴细胞数目的比率")
    title_list.append("AreaTS中淋巴细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaTS中成纤维细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaTS中成纤维细胞数目与淋巴细胞数目的比率")
    title_list.append("AreaI中淋巴细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaI中成纤维细胞数目与肿瘤细胞数目的比率")
    title_list.append("AreaI中成纤维细胞数目与淋巴细胞数目的比率")
    title_list.append("AreaTS中淋巴细胞数目与组织Mask中淋巴细胞总数的比率")
    title_list.append("AreaI中淋巴细胞数目与组织Mask中淋巴细胞总数的比率")
    title_list.append("AreaO中淋巴细胞数目与组织Mask中淋巴细胞总数的比率")
    title_list.append("AreaTS中成纤维细胞数目与组织Mask中成纤维细胞总数的比率")
    title_list.append("AreaI中成纤维细胞数目与组织Mask中成纤维细胞总数的比率")
    title_list.append("AreaO中成纤维细胞数目与组织Mask中成纤维细胞总数的比率")
    title_list.append("HotS_T面积占组织Mask面积比率")
    title_list.append("HotS_L面积占组织Mask面积比率")
    title_list.append("HotS_F面积占组织Mask面积比率")
    title_list.append("HotS_TL面积占组织Mask面积比率")
    title_list.append("HotS_TF面积占组织Mask面积比率")
    title_list.append("HotS_LF面积占组织Mask面积比率")
    title_list.append("HotS_TL面积占HotS_T面积比率")
    title_list.append("HotS_TL面积占HotS_L面积比率")
    title_list.append("HotS_TF面积占HotS_T面积比率")
    title_list.append("HotS_TF面积占HotS_F面积比率")
    title_list.append("HotS_LF面积占HotS_L面积比率")
    title_list.append("HotS_LF面积占HotS_F面积比率")
    title_list.append("AreaT与HotS_T重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_T重叠部分面积占HotS_T的比率")
    title_list.append("AreaT与HotS_L重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_L重叠部分面积占HotS_L的比率")
    title_list.append("AreaT与HotS_F重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_F重叠部分面积占HotS_F的比率")
    title_list.append("AreaS与HotS_T重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_T重叠部分面积占HotS_T的比率")
    title_list.append("AreaS与HotS_L重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_L重叠部分面积占HotS_L的比率")
    title_list.append("AreaS与HotS_F重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_F重叠部分面积占HotS_F的比率")
    title_list.append("AreaTS与HotS_T重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_T重叠部分面积占HotS_T的比率")
    title_list.append("AreaTS与HotS_L重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_L重叠部分面积占HotS_L的比率")
    title_list.append("AreaTS与HotS_F重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_F重叠部分面积占HotS_F的比率")
    title_list.append("AreaI与HotS_T重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_T重叠部分面积占HotS_T的比率")
    title_list.append("AreaI与HotS_L重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_L重叠部分面积占HotS_L的比率")
    title_list.append("AreaI与HotS_F重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_F重叠部分面积占HotS_F的比率")
    title_list.append("AreaT与HotS_TL重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_TL重叠部分面积占HotS_TL的比率")
    title_list.append("AreaT与HotS_TF重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_TF重叠部分面积占HotS_TF的比率")
    title_list.append("AreaT与HotS_LF重叠部分面积占AreaT的比率")
    title_list.append("AreaT与HotS_LF重叠部分面积占HotS_LF的比率")
    title_list.append("AreaS与HotS_TL重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_TL重叠部分面积占HotS_TL的比率")
    title_list.append("AreaS与HotS_TF重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_TF重叠部分面积占HotS_TF的比率")
    title_list.append("AreaS与HotS_LF重叠部分面积占AreaS的比率")
    title_list.append("AreaS与HotS_LF重叠部分面积占HotS_LF的比率")
    title_list.append("AreaTS与HotS_TL重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_TL重叠部分面积占HotS_TL的比率")
    title_list.append("AreaTS与HotS_TF重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_TF重叠部分面积占HotS_TF的比率")
    title_list.append("AreaTS与HotS_LF重叠部分面积占AreaTS的比率")
    title_list.append("AreaTS与HotS_LF重叠部分面积占HotS_LF的比率")
    title_list.append("AreaI与HotS_TL重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_TL重叠部分面积占HotS_TL的比率")
    title_list.append("AreaI与HotS_TF重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_TF重叠部分面积占HotS_TF的比率")
    title_list.append("AreaI与HotS_LF重叠部分面积占AreaI的比率")
    title_list.append("AreaI与HotS_LF重叠部分面积占HotS_LF的比率")
    title_list.append("AreaT与HotS_T重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_T重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_L重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_L重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_F重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_F重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_TL重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_TL重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_TF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_TF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_LF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaT与HotS_LF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_T重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_T重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_L重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_L重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_F重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_F重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_TL重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_TL重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_TF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_TF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaS与HotS_LF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaS与HotS_LF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_T重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_T重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_L重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_L重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_F重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_F重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_TL重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_TL重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_TF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_TF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaTS与HotS_LF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaTS与HotS_LF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_T重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_T重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_L重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_L重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_F重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_F重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_TL重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_TL重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_TF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_TF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaI与HotS_LF重叠部分的热点区域的面积的平均值")
    title_list.append("AreaI与HotS_LF重叠部分的热点区域的面积的标准差")
    title_list.append("AreaT与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaT与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaT与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaT与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaT与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaT与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaT与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaS与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaS与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaTS与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaTS与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_T重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_L重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_F重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_TL重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_TF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")
    title_list.append("AreaI与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的平均值")
    title_list.append("AreaI与HotS_LF重叠部分的热点区域与其最近N个相邻热点区域的距离的标准差")

    return title_list


'''
批量输出统计结果到CSV文件中
pkl_dir：PKL文件目录
svs_dir：SVS文件目录
matrix_dir: 九分类矩阵目录
step：SVS图片0级下采样的区域步长
nr：热点邻域阶数
min_area: 热点面积过滤最小值
n：相邻区域最大数量
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 癌症浸润区域的厚度
sample：SVS最大分辨率下一个像素的缩放倍数
region_filter：若为1，细胞数量分布需要经过肿瘤区域，间质区域，以及浸润区域矩阵过滤，为0则不过滤
statistic_save_name：统计文件保存命名
'''

# 去掉smaple根据图片获取
# def batch_ouput_statistics(pkl_dir, svs_dir, matrix_dir, step, nr, min_area, n, open_kernel_size, triangle_maxlen,
#                            invasive_thickness, sample, region_filter, statistic_save_name):
def batch_ouput_statistics(pkl_dir, svs_dir, matrix_dir, step, nr, min_area, n, open_kernel_size, triangle_maxlen,
                           invasive_thickness, region_filter, statistic_save_name):
    pkls = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    pkls = sorted(pkls)

    error_logger = get_logger(level="error")
    info_logger = get_logger(level="info")
    statistic_dir = os.path.join(current_path, "..", "output", "hotpot_statistics")
    if not os.path.isdir(statistic_dir): os.makedirs(statistic_dir)
    csv_title = GetTitle()
    fp_0 = open(os.path.join(statistic_dir, statistic_save_name + '.csv'), "w", encoding='utf-8', newline='')
    writer_0 = csv.writer(fp_0)
    csv_title.insert(0, "case")
    writer_0.writerow(csv_title)
    for pkl in pkls:
        try:
            start_time = time.time()
            info_logger.info("Starting create hotpot_statistics %s..." % (pkl))
            pkl_name = os.path.basename(pkl).split('.')[0]
            pkl_file = os.path.join(pkl_dir, pkl_name + ".pkl")
            svs_file = os.path.join(svs_dir, pkl_name + ".svs")
            matrix_file = os.path.join(matrix_dir, pkl_name + "-region_9_class_output.npy")
            # 改为动态获取
            slide = Slide(svs_file)
            sample = slide.get_mpp()
            slide.close()

            statistic_list = get_nuclei_statistics(pkl_file, svs_file, matrix_file, step, nr, min_area, n,
                                                   open_kernel_size, triangle_maxlen, invasive_thickness, sample,
                                                   region_filter)
            for i in range(len(statistic_list)):
                statistic_0 = statistic_list[i]
                statistic_0.insert(0, f"{pkl_name}_{str(i)}")
                writer_0.writerow(statistic_0)
            info_logger.info("Finished create hotpot_statistics %s, needed %.2f sec" % (pkl, time.time() - start_time))
        except Exception as e:
            error_logger.error('Create hotpot_statistics %s Error' % pkl, exc_info=True)
    fp_0.close()


if __name__ == '__main__':

    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path, "..", "sys.ini"))

    pkl_dir = conf.get("UTILS_HEATMAP", "PKL_DIR")
    svs_dir = conf.get("UTILS_HEATMAP", "SVS_DIR")
    matrix_dir = conf.get("UTILS_HEATMAP", "MATRIX_DIR")

    step = int(conf.get("DEFAULT", "STEP"))
    nr = int(conf.get("DEFAULT", "NR"))

    min_area = int(conf.get("DEFAULT", "MIN_AREA"))
    n = int(conf.get("DEFAULT", "N"))

    open_kernel_size = int(conf.get("DEFAULT", "OPEN_KERNEL_SIZE"))
    triangle_maxlen = int(conf.get("DEFAULT", "TRIANGLE_MAXLEN"))
    invasive_thickness = int(conf.get("DEFAULT", "INVASIVE_THICKNESS"))
    # 去掉smaple根据图片获取
    # sample = float(conf.get("DEFAULT", "SAMPLE"))

    region_filter = int(conf.get("DEFAULT", "REGION_FILTER"))

    statistics_save_name = conf.get("DEFAULT", "STATISTICS_SAVE_NAME")
    # 去掉smaple根据图片获取
    # batch_ouput_statistics(pkl_dir, svs_dir, matrix_dir, step, nr, min_area, n, open_kernel_size, triangle_maxlen,
    #                        invasive_thickness, sample, region_filter, statistics_save_name)
    batch_ouput_statistics(pkl_dir, svs_dir, matrix_dir, step, nr, min_area, n, open_kernel_size, triangle_maxlen,
                           invasive_thickness, region_filter, statistics_save_name)

# @Time	: 2018.12.04
# @Author  : kawa Yeung
# @Licence : bio-totem


import os
import gc
# import ast
import glob
import time
import configparser

import numpy as np
# from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import sys

sys.path.append('../')
# 主要这个路径要根据实际情况配置,如果是工程内部的话可以使用相对路径
from utils.log_utils import get_logger
from utils.openslide_utils import Slide
from utils.xml_utils import xml_to_region,region_handler
from utils.opencv_utils import OpenCV
from utils.tissue_utils import get_tissue

from output_process.infiltrating_region_staistics import get_cancer_invasive_matrix, get_region_neighbor_with_mask
from output_process.infiltrating_region_staistics import get_nuclei_invasive_region_matrix, get_hotpot_matrix_with_mask
from output_process.infiltrating_region_staistics import get_nuclei_amount_region_matrix
from output_process.process_script import remove_discrete_point, get_mask,matrix_resize
from utils.opencv_utils import OpenCV

'''
创建密度图（主要是将细胞数量分布矩阵归一化后，再用colormap进行绘制）
svs_file: svs图片路径
nuclei_matrix_0: 细胞数量分布矩阵
title: 密度图标题
output_dir: 输出目录
'''


def create_heatmap(svs_file, nuclei_matrix_0, pkl_name,step, sort_name, output_dir,xml = False):
    """
    Create cancer heatmap
    :param svs_im_file:
    :param nuclei_matrix_0: cancer nuclei matrix
    :param pkl_name: base file name of svs file
    :param step: stride of the target WSI file 'sampling in 0 level
    :param sort_name: heatmap title, alway equals the name of nuclei's class
    :param output_dir: the heatmap figure ' saving dir
    :return:
    """

    if np.sum(nuclei_matrix_0 != 0) < 1:
        return
    # 加载SVS图片
    print(svs_file)
    slide = Slide(svs_file)
    tile = slide.get_thumb()
    #xml_file = os.path.join(INPUT_XML_DIR,pkl_name + '.xml')
    # 获取2级下采样的SVS图片
    # 获取2级下采样的SVS图片
    svs_im_npy = np.array(tile.convert('RGBA'))
    #if xml and os.path.exists(xml_file):
    #    region_list,region_class = xml_to_region(xml_file)
     #   svs_im_npy = region_handler(tile, region_list, region_class,slide.get_level_downsample())
     #   svs_im_npy = np.array(svs_im_npy.convert('RGBA'))
    #else:
     #   svs_im_npy = np.array(tile.convert('RGBA'))
    slide.close()
    plt_size = (tile.size[0] // 100, tile.size[1] // 100)
    # fig, ax = plt.subplots(figsize=plt_size, dpi=100)
    fig, ax = plt.subplots(figsize=plt_size)
    cancer_nuclei_matrix = nuclei_matrix_0.copy()
    # 对矩阵进行归一化处理
    min = np.min(cancer_nuclei_matrix)
    max = np.max(cancer_nuclei_matrix)
    #sigmma = np.std(cancer_nuclei_matrix)
    mean = np.average(cancer_nuclei_matrix)
    median = np.median(cancer_nuclei_matrix)
    #mnu = max - min

    print(mean,median,min,max)
    #cancer_nuclei_matrix = cancer_nuclei_matrix - mean
    #cancer_nuclei_matrix = cancer_nuclei_matrix / sigmma
    # cancer_nuclei_matrix *= 10
    # 矩阵过滤化处理及尺寸调整
    # cancer_nuclei_matrix = cv2.resize(cancer_nuclei_matrix,svs_im.size, interpolation=cv2.INTER_AREA)
    cancer_nuclei_matrix = OpenCV(cancer_nuclei_matrix).resize(tile.size[0], tile.size[1],                                                              interpolation='INTER_AREA')
    cancer_nuclei_matrix /= step ** 2 / 10
    print(cancer_nuclei_matrix)
    # 矩阵colormap处理
    cax = ax.imshow(cancer_nuclei_matrix, cmap=plt.cm.jet, alpha=0.45)
    # 背景透明化处理
    svs_im_npy[:, :][cancer_nuclei_matrix[:, :] > 0] = 0
    ax.imshow(svs_im_npy)
    max_matrix_value = cancer_nuclei_matrix.max()
    plt.colorbar(cax, ticks=np.linspace(0, max_matrix_value, 25, endpoint=True))
    sort_name = sort_name.title()
    ax.set_title(pkl_name + ' ' + sort_name +' Amount Distribution HeatMap', fontsize=40)
    plt.axis('off')
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    sort_name = sort_name.lower()
    plt.savefig(os.path.join(output_dir, pkl_name + '-' + sort_name + "-heatmap.png"))
    plt.close('all')
    del tile, cancer_nuclei_matrix,svs_im_npy
    gc.collect()


'''
创建热点图
svs_file：svs图片路径
nuclei_amount_matrix：细胞数量分布矩阵
nr:热点邻域阶数
pkl_name:PKL名称
sort_name:细胞类型
color:热点颜色
min_area:过滤热点区域面积最小值
n:热点区域的相邻区域的最大数量，当n大于0时，绘制热点区域的相邻区域连线图
output_dir:热点图输出目录
'''


def create_hotspot(svs_file, nuclei_amount_matrix, nr, pkl_name, sort_name, color, min_area, n, output_dir):
    if np.sum(nuclei_amount_matrix != 0) < 1:
        return

    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取2级下采样的SVS图片尺寸
    matrix_size = slide.get_level_dimension(level=2)

    # 获取2级下采样的SVS图片
    svs_im = slide.read_region((0, 0), 2, matrix_size)

    slide.close()

    # 获取MASK
    mask = get_mask(np.array(svs_im.convert('RGB')))

    # 获取热点矩阵
    hotspot_matrix = get_hotpot_matrix_with_mask(mask, nuclei_amount_matrix, nr)

    # 将svs图片转换为Nunpy矩阵，方便后面进行背景透明化处理
    svs_im_npy = np.array(svs_im.convert('RGBA'))

    # 将热点矩阵放大为svs图片的尺寸，热点矩阵尺寸过小
    # resize_hotspot_matrix = cv2.resize(hotspot_matrix,(svs_im_npy.shape[1],svs_im_npy.shape[0]), interpolation=cv2.INTER_AREA)
    resize_hotspot_matrix = OpenCV(hotspot_matrix).resize(svs_im_npy.shape[1], svs_im_npy.shape[0],
                                                          interpolation='INTER_AREA')
    # 根据热点矩阵的热点像素位置，将svs图片矩阵用指定颜色在对应像素位置进行标注
    svs_im_npy[:, :, :][resize_hotspot_matrix[:, :] != 0] = color

    # 如果相邻区域最大数量不为0，则绘制热点区域的相邻区域连线图
    if n > 0:
        edge_img = get_region_neighbor_edge_img(svs_im, hotspot_matrix, min_area, n)

        # 连线的颜色绘制为黑色
        svs_im_npy[:, :, :][edge_img[:, :, 3] != 0] = [0, 0, 0, 255]

    plt_size = (svs_im_npy.shape[1] // 200, svs_im_npy.shape[0] // 200)
    fig, ax = plt.subplots(figsize=plt_size, dpi=200)
    ax.imshow(svs_im_npy)
    sort_name = sort_name.title()
    ax.set_title(' NR = ' + str(nr) + ' ' + pkl_name + ' ' + sort_name + ' Hotspots', fontsize=20)
    plt.axis('off')
    sort_name = sort_name.lower()
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'NR-' + str(nr) + '-' + pkl_name + '-' + sort_name + '-hotspot.png'))
    plt.close('all')


'''
获取热点区域与相邻区域的连线图
svs_im: 二级下采样svs图片
matrix_0: 热点矩阵
min_area：过滤热点区域最小面积
n：热点区域的相邻区域的最大数量
'''


def get_region_neighbor_edge_img(svs_im, matrix_0, min_area, n):
    matrix = np.copy(matrix_0)

    svs_im_npy = np.array(svs_im)

    # 过滤矩阵中的最小面积轮廓，此步代替了获取热点区域与相邻区域的边的面积过滤
    # contours=get_contours(matrix)
    contours = OpenCV(matrix).find_contours(is_binary=True)
    cnt_list = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cnt_list.append(cnt)

    # 过滤矩阵最小面积轮廓
    filter_matrix = np.full((matrix.shape[0], matrix.shape[1]), 0, dtype=np.uint8)
    cv2.drawContours(filter_matrix, cnt_list, -1, 1, cv2.FILLED)
    matrix = filter_matrix * matrix

    # 获取组织MASK
    mask = get_mask(svs_im_npy)

    # 缩放矩阵为svs图片二级下采样的尺寸
    # matrix=cv2.resize(matrix,(svs_im_npy.shape[1],svs_im_npy.shape[0]),cv2.INTER_NEAREST)
    matrix = OpenCV(matrix).resize(svs_im_npy.shape[1], svs_im_npy.shape[0], interpolation='INTER_NEAREST')
    # 由于矩阵是以插值的方式放大，为了保证矩阵的二值化，需要将非0数值设为1
    matrix[matrix != 0] = 1

    # 获取所有热点区域与相邻区域的边列表，热点区域面积过滤最小值设置0是因为上面已经做了过滤
    all_neighbor_edge_list = get_region_neighbor_with_mask(matrix, mask, 0, n)

    # 绘制所有热点区域与相邻区域的边，颜色为黑色，厚度为5
    edge_img = np.zeros((svs_im_npy.shape[0], svs_im_npy.shape[1], 4), dtype=np.uint8)
    for neighbor_edge_list in all_neighbor_edge_list:
        if len(neighbor_edge_list) > 0:
            for edge_list in neighbor_edge_list:
                cv2.line(edge_img, tuple(edge_list[1][0]), tuple(edge_list[1][1]), [0, 0, 0, 255], 5)

    return edge_img


"""
通过索引1~1275获取红到蓝渐变色中的指定颜色
index: 索引1~1275对应颜色红到蓝
"""


def index_color(index):
    color = [0, 0, 0, 255]
    area = int(index / 255)
    move = index % 255
    baseloc = int(area / 2)
    exloc = int((area / 2 + area % 2) % 3)
    ready = int(2 - ((1 + area) % 3))
    sign = 1 - area % 2 * 2
    color[baseloc] = 255
    color[exloc] = 255;
    color[ready] += sign * move
    return color


"""
生成肿瘤区域，浸润区域，以及间质区域图片
svs_file: svs图片路径
matrix_file：九分类矩阵路径
title: 图片中需要展示的标题
output_dir: 输出目录
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 癌症浸润区域的厚度
"""


def cancer_invasive_show(svs_file, matrix_file, title, output_dir, open_kernel_size, triangle_maxlen,
                         invasive_thickness):
    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取2级下采样的SVS图片尺寸
    matrix_size = slide.get_level_dimension(level=2)

    # 获取2级下采样的SVS图片
    svs_img = slide.read_region((0, 0), 2, matrix_size)

    slide.close()

    # 获取肿瘤，浸润，间质区域矩阵（癌症区域为1，浸润区域为3，间质区域为2）
    cancer_invasive_matrix, mask = get_cancer_invasive_matrix(svs_file, matrix_file, open_kernel_size, triangle_maxlen,
                                                              invasive_thickness)

    # 加载标签以及对应颜色
    labels = ['tumor', 'stroma', 'invasive margin']
    colors = np.array([[0, 0, 255, 255], [0, 255, 0, 255], [0, 255, 255, 255]])

    # 将底图缩放为matrix矩阵的尺寸
    svs_img_bg = svs_img.resize((cancer_invasive_matrix.shape[1], cancer_invasive_matrix.shape[0]))

    # 设置Patches
    patches = [mpatches.Patch(color=colors[i] / 255, label="{:s}".format(labels[i])) for i in range(len(labels))]

    # 为癌症，浸润，症间区域分别上色
    cancer_invasive_image = np.full((cancer_invasive_matrix.shape[0], cancer_invasive_matrix.shape[1], 4), 0,
                                    dtype=np.uint8)
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 1] = colors[0]
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 2] = colors[1]
    cancer_invasive_image[:, :, :][cancer_invasive_matrix[:, :] == 3] = colors[2]

    # 画板配置
    plt_size = (cancer_invasive_matrix.shape[1] // 200, cancer_invasive_matrix.shape[0] // 200)
    fig, ax = plt.subplots(figsize=plt_size, dpi=200)

    # 绘制图片
    ax.imshow(svs_img_bg)
    ax.imshow(cancer_invasive_image)

    # 绘制图例
    ax.legend(handles=patches)

    ax.set_title(title, fontsize=20)
    plt.axis('off')
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))
    plt.close('all')


"""
生成九分类矩阵可视化图片
svs_file: svs图片路径
matrix_file: 二分类矩阵路径
title: 图片中需要展示的标题
output_dir: 输出目录
"""


def matrix_visual_show(svs_file, matrix_file, title, output_dir,xml = False):
    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取2级下采样的SVS图片
#    svs_img = slide.read_region((0, 0), 2, matrix_size)
    matrix = np.load(matrix_file)
    matrix = matrix_resize(slide,matrix,patch_size=224)

    svs_img =slide.get_thumb()
    xml_file = os.path.join(INPUT_XML_DIR,title + '.xml')
    # 获取2级下采样的SVS图片
    if xml and os.path.exists(xml_file):        
        region_list,region_class = xml_to_region(xml_file)
        svs_img = region_handler(svs_img, region_list, region_class,slide.get_level_downsample())
    slide.close()
    # 加载标签以及对应颜色
    labels = ['BACK', 'DEAD', 'OTHER']
    colors = np.full((len(labels), 4), 0, dtype=np.uint8)

    # 将底图缩放到matrix矩阵的尺寸，以及将其转换为RGB三通道
    svs_img = svs_img.resize((matrix.shape[1], matrix.shape[0]))
#    svs_img_rgb = svs_img.convert('RGB')

    # 获取MASK，并将其过滤matrix矩阵中的非组织区域
#    mask = get_mask(np.array(svs_img_rgb).astype(np.uint8))
#    matrix = matrix.astype(int)
#    matrix = matrix * mask

    # 过滤matrix矩阵中非细胞类型的值
    matrix[:, :][(matrix > 2) | ((matrix > 0) & (matrix < 1))] = 0

    # 去除矩阵中的离散点
    matrix = remove_discrete_point(matrix.astype(np.uint8), 15)

    # 获取matrix矩阵中的唯一值列表，唯一值列表会用在颜色以及标签的索引
    index_npy = np.unique(matrix)

    # 获取唯一值列表中属于细胞类型的值
    index_npy = index_npy[index_npy != 0]

    # 获取细胞种类数目
    index_count = np.size(index_npy)

    # 获取颜色倍数
    # 1020是最大颜色索引值
    color_irc = int(1020 / (index_count - 1)) if index_count >= 2 else 0

    # 根据唯一值列表获取颜色值列表
    for i in range(index_count):
        colors[index_npy[i]] = index_color(i * color_irc)

    # 根据matrix的值为图片矩阵上色
    img_matrix = np.full((matrix.shape[0], matrix.shape[1], 4), 0, dtype=np.uint8)
    img_matrix[:, :, :] = colors[matrix[:, :]]

    # 画板配置
    plt_size = (matrix.shape[1] // 200, matrix.shape[0] // 200)
    fig, ax = plt.subplots(figsize=plt_size, dpi=200)

    # 设置Patches
    patches = [mpatches.Patch(color=colors[i] / 255, label="{:s}".format(labels[i])) for i in index_npy]

    # 绘制图片
    ax.imshow(svs_img)
    ax.imshow(img_matrix)

    # 绘制图例
    ax.legend(handles=patches)

    ax.set_title(title, fontsize=20)
    plt.axis('off')
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))
    plt.close('all')

def cancel_dead_matrix_visual_show(svs_file, matrix_file, cancer_nuclei_matrix,step, cancel_threshold, title, output_dir,xml = True):
    # 加载SVS图片
    slide = Slide(svs_file)

    # 获取2级下采样的SVS图片
#    svs_img = slide.read_region((0, 0), 2, matrix_size)
    matrix = np.load(matrix_file)
    # 将矩阵放大到WSI图像2级下采样的大小
    matrix = matrix_resize(slide,matrix,patch_size=224)
    
    # 对癌细胞二值化矩阵进行相应处理,求出肿瘤细胞密集区域分布bool矩阵(会进行开闭操作)
    cancer_nuclei_matrix /= step**2/200
    cancer_nuclei_matrix = cancer_nuclei_matrix >= cancel_threshold    
    cancer_nuclei_matrix = OpenCV(np.uint8(cancer_nuclei_matrix)).resize(matrix.shape[1], matrix.shape[0])
#    kernel = np.ones((50,50),np.uint8) 
#    cancer_nuclei_matrix = cv2.morphologyEx(cancer_nuclei_matrix, cv2.MORPH_OPEN, kernel)
#    cancer_nuclei_matrix = cv2.morphologyEx(cancer_nuclei_matrix, cv2.MORPH_CLOSE, kernel)    
    cancer_nuclei_matrix = cancer_nuclei_matrix > 0  

    svs_img =slide.get_thumb()
    xml_file = os.path.join(INPUT_XML_DIR,title + '.xml')
    # 获取2级下采样的SVS图片
    if xml and os.path.exists(xml_file):        
        region_list,region_class = xml_to_region(xml_file)
        svs_img = region_handler(svs_img, region_list, region_class,slide.get_level_downsample())

    # 剔除白色区域
    mask, _ = get_tissue(np.array(svs_img)[:,:,:3], contour_area_threshold=1000)

    
    slide.close()
    # 定义组合标签以及对应颜色
    labels = ['BACK','DEAD', 'OTHER','CANCER']
    colors = np.full((len(labels) , 4), 0, dtype=np.uint8)

    # 将底图及mask缩放到matrix矩阵的尺寸，以及将其转换为RGB三通道
    svs_img = svs_img.resize((matrix.shape[1], matrix.shape[0]))
    mask = OpenCV(mask).resize(matrix.shape[1], matrix.shape[0])
    
    # 过滤matrix矩阵中非坏死、其他区域
    matrix[:, :][matrix < 1] = 0
    # 将白色区域填零
    
    matrix[(mask < 1) & (matrix >0)] = 0
    # 去除矩阵中的离散点
    matrix = remove_discrete_point(matrix.astype(np.uint8), 15)
    
    # 将肿瘤细胞密集区域分布bool矩阵和matrix矩阵融合
    matrix[cancer_nuclei_matrix] = 3
    # 获取matrix矩阵中的唯一值列表，唯一值列表会用在颜色以及标签的索引
    index_npy = np.unique(matrix)

    # 获取唯一值列表中属于细胞类型的值
    index_npy = index_npy[index_npy != 0]

    # 获取细胞种类数目
    index_count = np.size(index_npy)

    # 获取颜色倍数
    # 1020是最大颜色索引值
    color_irc = int(1020 / (index_count - 1)) if index_count >= 2 else 0

    # 根据唯一值列表获取颜色值列表
    for i in range(index_count):
        colors[index_npy[i]] = index_color(i * color_irc)

    # 根据matrix的值为图片矩阵上色
    img_matrix = np.full((matrix.shape[0], matrix.shape[1], 4), 0, dtype=np.uint8)
    img_matrix[:, :, :] = colors[matrix[:, :]]

    # 画板配置
    plt_size = (matrix.shape[1] // 200, matrix.shape[0] // 200)
    fig, ax = plt.subplots(figsize=plt_size, dpi=200)

    # 设置Patches
    patches = [mpatches.Patch(color=colors[i] / 255, label="{:s}".format(labels[i])) for i in index_npy]

    # 绘制图片
    ax.imshow(svs_img)
    ax.imshow(img_matrix)

    # 绘制图例
    ax.legend(handles=patches)

    ax.set_title(title, fontsize=20)
    plt.axis('off')
    output_dir = os.path.join(output_dir,str(cancel_threshold*100))
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, title + "_" + str(cancel_threshold*100) + "_percentage.png"))
    plt.close('all')

'''
分别生成九分类矩阵可视化图、癌症区域，间质区域，以及浸润区域图、肿瘤细胞，成纤维细胞，淋巴细胞的密度图、
肿瘤细胞，成纤维细胞，淋巴细胞的热点图，n大于0时则绘制热点区域的相邻区域连线图
pkl_dir: PKL目录路径
svs_dir：SVS目录路径
proba_matrix_dir: 九分类矩阵目录路径
output_dir：输出目录
step: SVS图片0级下采样的区域步长
nr: 热点邻域阶数
min_area: 过滤热点区域最小面积
n: 热点区域的相邻区域的最大数量
open_kernel_size: OPENCV开操作卷积核大小,开操作主要是去除矩阵中的离散点，使得生成的肿瘤区域更加合适
triangle_maxlen: 三角剖分算法中三角形最大边长值
invasive_thickness: 癌症浸润区域的厚度
region_filter：若为1，细胞数量分布需要经过肿瘤区域，间质区域，以及浸润区域矩阵过滤，为0则不过滤
'''


def batch_create_heatmap(pkl_dir, svs_dir, proba_matrix_dir, output_dir, step, cancel_threshold,nr, min_area, n, open_kernel_size,
                         triangle_maxlen, invasive_thickness, region_filter):
    """
    Batch create cancer heatmap
    :param pkl_dir:
    :param svs_dir:
    :param min_ration:
    :param proba_matrix_dir:
    :param singlewhole:
    :return:
    """
    # 针对癌细胞这一类,可以指定读取固定后缀格式的概率矩阵文件来实现区域画图，如"{pkl_name}-left-cnm.npy"
    #  如果是针对其他类别进行作图，可以直接修改下面读取文件的后缀格式来进行对应的画图，但比较建议以类别名字
    #  新建一个方法，如fibroblast就新建一个fibroblast.py文件(主要对应变量名字也要修改)
    #  如果想要单独执行这一类细胞的热力图，可直接执行对应的py文件则可。如果要进行癌细胞的统计，就直接执行本脚本
    svss = glob.glob(os.path.join(svs_dir, "*.svs"))
    svss = sorted(svss)

    error_logger = get_logger(level="error")
    info_logger = get_logger(level="info")
    for svs in svss:
        try:
            start_time = time.time()
            info_logger.info("Starting create cancer heatmap %s..." % svs)
            svs_name = os.path.splitext(os.path.basename(svs))[0]
            pkl_file = os.path.join(pkl_dir, svs_name + ".pkl")
            svs_file = os.path.join(svs_dir, svs_name + ".svs")
            matrix_file = os.path.join(proba_matrix_dir, f"{svs_name}-region_dead_other_output.npy")
            if os.path.exists(matrix_file) and os.path.exists(pkl_file) :
            # 生成二分类矩阵可视化图
#                matrix_visual_dir = os.path.join(output_dir, 'matrix_visual')
#                matrix_visual_show(svs_file, matrix_file, svs_name, matrix_visual_dir,xml=True)
#    
#                # 生成癌症区域，间质区域，以及浸润区域图
#                cancer_invasive_dir = os.path.join(output_dir, 'cancer_invasive')
#                cancer_invasive_show(svs_file, matrix_file, svs_name, cancer_invasive_dir, open_kernel_size,
#                                     triangle_maxlen, invasive_thickness)
    
##                 获取肿瘤细胞，成纤维细胞，淋巴细胞，混合细胞的细胞数量分布矩阵以及肿瘤，浸润，间质区域矩阵和组织区域矩阵
##                AmountT, AmountF, AmountL, AmountM, AreaT, AreaS, AreaI, AreaM = get_nuclei_invasive_region_matrix(pkl_file,
##                                                                                                                   svs_file,
##                                                                                                                   step, 
##                                                                                                                   matrix_file,
##                                                                                                                   open_kernel_size, 
##                                                                                                                   triangle_maxlen,
##                                                                                                                   invasive_thickness,
##                                                                                                                   region_filter)
                Amount_epithelial, Amount_lymphocyte, Amount_other = get_nuclei_amount_region_matrix(pkl_file, svs_file, step)
#                # 分别生成肿瘤细胞，成纤维细胞，淋巴细胞的密度图
                heatmap_dir = os.path.join(output_dir, 'heatmap')
                create_heatmap(svs_file, Amount_epithelial, svs_name, step, 'Cancer', heatmap_dir,xml=True)
                create_heatmap(svs_file, Amount_lymphocyte, svs_name, step,'Lymphocyte', heatmap_dir,xml=True)
                create_heatmap(svs_file, Amount_other, svs_name, step, 'Other', heatmap_dir,xml=True)
#                Amount_epithelial[(Amount_epithelial < Amount_lymphocyte) | (Amount_epithelial < Amount_other)]=0
                Amount_epithelial[Amount_epithelial < Amount_other * 0.5]=0
                # 只取肿瘤细胞矩阵中同等位置比其他两类细胞都多的元素
                cancel_dead_matrix_visual_dir = os.path.join(output_dir, 'cancel_dead_matrix_visual_dir')
                cancel_dead_matrix_visual_show(svs_file, matrix_file, Amount_epithelial,step, cancel_threshold, svs_name, cancel_dead_matrix_visual_dir)
#                # 分别生成肿瘤细胞，成纤维细胞，淋巴细胞的热点图，n大于0时则绘制热点区域的相邻区域连线图
#                hotspot_dir = os.path.join(output_dir, 'hotspot')
#                create_hotspot(svs_file, AmountT, nr, svs_name, 'cancer', [255, 0, 0, 255], min_area, n, hotspot_dir)
#                create_hotspot(svs_file, AmountF, nr, svs_name, 'fibroblast', [0, 255, 0, 255], min_area, n, hotspot_dir)
#                create_hotspot(svs_file, AmountL, nr, svs_name, 'inflammatory', [0, 0, 255, 255], min_area, n, hotspot_dir)
    
                info_logger.info("Finished create cancer heatmap %s, needed %.2f sec" % (svs, time.time() - start_time))
        except Exception:
            error_logger.error('Create cancer heatmap %s Error' % svs, exc_info=True)


if __name__ == "__main__":
    conf = configparser.ConfigParser()
    current_path = os.path.dirname(__file__)
    conf.read(os.path.join(current_path, "..", "sys.ini"))

    INPUT_XML_DIR = conf.get("DEFAULT", "INPUT_XML_DIR")
    svs_dir = conf.get("UTILS_HEATMAP", "SVS_DIR")
    pkl_dir = conf.get("UTILS_HEATMAP", "PKL_DIR")
    test_proba_matrix_dir = conf.get("UTILS_HEATMAP", "MATRIX_DIR")
    output_dir = os.path.join(current_path, "..", "output")

    step = int(conf.get("DEFAULT", "STEP"))
    cancel_threshold = float(conf.get("DEFAULT", "CACEL_CELL_COUNT_THRESHOLD"))
    nr = int(conf.get("DEFAULT", "NR"))

    min_area = int(conf.get("DEFAULT", "MIN_AREA"))
    n = int(conf.get("DEFAULT", "N"))

    open_kernel_size = int(conf.get("DEFAULT", "OPEN_KERNEL_SIZE"))
    triangle_maxlen = int(conf.get("DEFAULT", "TRIANGLE_MAXLEN"))
    invasive_thickness = int(conf.get("DEFAULT", "INVASIVE_THICKNESS"))

    region_filter = int(conf.get("DEFAULT", "REGION_FILTER"))

    batch_create_heatmap(pkl_dir, svs_dir, test_proba_matrix_dir, output_dir, step, cancel_threshold,nr, min_area, n, open_kernel_size,
                         triangle_maxlen, invasive_thickness, region_filter)

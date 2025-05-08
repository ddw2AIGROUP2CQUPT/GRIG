import argparse
import math
import os
import cv2
import numpy as np
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from collections import Counter
import networkx as nx
from collections import Counter
import scipy.sparse as sp
import itertools
import random

random.seed(0)


# 返回所有像素梯度信息
def get_grad(img, dir=None):
    """
    :param img:
    :return: map{(x,y):grad} and grad matric
    """
    # get grad list
    # 分别计算x、y方向：右减左，下减上
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 梯度矩阵 : same shape with the img

    return sobelxy2


def cal_bound(img, center, Rx, Ry):
    """
    :param img:  原图，需要获取原图的大小
    :param center: 中心点坐标
    :param Rx, Ry: 不同方向的半径
    :return: 上下左右边界
    """
    h, w = img.shape
    left = center[1] - Rx
    right = center[1] + Rx
    up = center[0] - Ry
    down = center[0] + Ry

    if left < 0:
        left = 0
    if right >= w:
        right = w - 1
    if up < 0:
        up = 0
    if down >= h:
        down = h - 1

    return int(left), int(right), int(up), int(down)


def center_select(img_grad, img_label):
    """
    :param img_grad:
    :param img_label:
    :return:
    """
    minPix_xy = np.where(img_grad == img_grad.min())  # 返回一个二维元组，前一个为所有梯度最小的点的横坐标，后一个为所有梯度最小的点的纵坐标

    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)  # 随机挑选一个中心点
        # print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos], minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos], minPix_xy[1][pos]]


def cal_Radius_1(img, center, purity, threshold, var_threshold):  # 半径计算1：半径初始化为0，纯度逐渐增加，适用于中心点梯度较小的情况
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径
    """
    Rx = 0  # 初始化半径
    Ry = 0

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0

    center_value = int(img[center[0], center[1]])

    while True:

        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2

        # print(item_count)

        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        # 计算切片边界
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)  # 当前总像素点

        if pixNum == temp_pixNum:
            return Rx, Ry, 1

        # print("当前中心为:", [center[0], center[1]], "当前半径为:", Rx, Ry ,"总像素点数为:", pixNum)

        # 计算异类点个数
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])

        # print("异类点个数为：", count)

        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum

        # print("当前纯度为：", temp_purity)
        if temp_purity > purity and var < var_threshold:
            if purity < 0.99:
                purity = purity * 1.005
            else:
                purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        # print(flag_x, flag_y)

        if flag_x == False and flag_y == False:
            return Rx, Ry, temp_purity


def calulate_weight(img, center_1, center_2):
    """
    :param img:
    :param center_1:
    :param center_2:
    :return: 两个粒矩相交像素点个数
    """
    left_1, right_1, up_1, down_1 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    left_2, right_2, up_2, down_2 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    x_list = [up_1, up_2, down_1, down_2]
    y_list = [left_1, left_2, right_1, right_2]
    x_list.sort()
    y_list.sort()

    res = (x_list[2] - x_list[1] + 1) * (y_list[2] - y_list[1] + 1)

    return res


def calulate_A_and_B(img, center_1, center_2):
    """
    :param center_1:
    :param center_2:
    :return: 两个粒矩的像素点数和
    """
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    return (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1)


def granular_balls_generate(img, purity_1=0.9, threshold=10, var_threshold_1=20):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值
    # print(grad_median)
    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label:  # 存在没有被划分的点
        temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点

        Rx, Ry, temp_purity = cal_Radius_1(img, temp_center, purity_1, threshold, var_threshold_1)

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)

        center.append((
            temp_center,
            Rx, Ry,
            RGB_img[..., 0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[..., 0][up:down + 1, left:right + 1]),
            RGB_img[..., 1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[..., 1][up:down + 1, left:right + 1]),
            RGB_img[..., 2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[..., 2][up:down + 1, left:right + 1]),
            img_grad[temp_center[0], temp_center[1]],
            img[up:down + 1, left:right + 1].max(), img[up:down + 1, left:right + 1].min()
        ))  # img[up:down + 1, left:right + 1].max
        #  temp_purity

        img_label[up:down + 1, left:right + 1] = 1

        img_grad[up:down + 1, left:right + 1] = max_Grad

        center_count += 1

    end = time.time()

    g = nx.Graph()

    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))

    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            # 两个中心点的 x,y 距离分别和 两个粒矩的 Rx,Ry 之和进行比较，判断两个粒矩之间的位置关系（相交，相接，相离）
            # if (abs(center_1[0][0] - center_2[0][0]) - 1) <= center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) <= center_1[1] + center_2[1]:  # 相接有边
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))

    # 3. 生成 GNN 需要的数据   
    a = nx.to_numpy_matrix(g)

    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 13))  # 节点属性 (需要在粒矩生成的时候将属性添加进去才能操作)
    edge_attr = np.zeros((len(adj[0]), 3))  # 边特征
    center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry

    # 边特征 edge_attr -> ①两个粒矩相交的像素点个数，②两个粒矩相交的像素点占两个粒矩总像素点的个数，③两个粒矩中心的距离
    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        center_dis = math.sqrt((center[adj[0][i]][0][0] - center[adj[1][i]][0][0]) ** 2 + (
                    center[adj[0][i]][0][1] - center[adj[1][i]][0][1]) ** 2)
        edge_attr[i] = [temp, temp / temp_iou, center_dis]
    # 粒矩相交像素点个数和粒矩中心距离归一化
    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()
    edge_attr[:, 2] = edge_attr[:, 2] / edge_attr[:, 2].max()

    # 生成节点属性和粒矩基础属性数组
    for id in range(len(center)):
        # 节点属性 center_array
        center_array[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3],
                            center[id][4],
                            center[id][5], center[id][6], center[id][7], center[id][8], center[id][9], center[id][10],
                            center[id][11]]
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    return center_array, adj, edge_attr, center_

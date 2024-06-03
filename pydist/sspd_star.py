#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: sspd_star.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2024/1/17 17:23
import numpy as np
from scipy.io import loadmat, savemat


def euclidean_dist(p1, p2):
    """
    :param p1:
    :param p2:
    :return:
    """
    # dist = np.linalg.norm(x - y)
    # dist = np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
    try:
        dist = np.sqrt((int(p2[0]) - int(p1[0])) ** 2 + (int(p2[1]) - int(p1[1])) ** 2)
    except:
        print(p1, p2)
    # dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def point_to_seg(p, s1, s2):
    """
    :param p:
    :param s1:
    :param s2:
    :return:
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x == p2x and p1y == p2y:
        dpl = euclidean_dist(p, s1)
        ix, iy = p1x, p1y
    else:
        segl = euclidean_dist(s1, s2)
        u1 = (((px - p1x) * (p2x - p1x)) + ((py - p1y) * (p2y - p1y)))

        u = u1 / (segl * segl)
        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = euclidean_dist(p, s1)
            iy = euclidean_dist(p, s2)

            if ix > iy:
                dpl = iy

            else:
                dpl = ix

        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * (p2x - p1x)
            iy = p1y + u * (p2y - p1y)

            dpl = euclidean_dist(p, np.array([ix, iy]))

    return dpl, [ix, iy]


def point_to_trajectory(p, t):
    if len(p) == 0 or len(t) < 2:
        dpt = 9e100
        return dpt

    dpt = 9e100
    dp_index = 0
    for i in range(0, len(t) - 1):
        s1 = t[i]
        s2 = t[i + 1]
        temp_dpt, _ = point_to_seg(p, s1, s2)
        if temp_dpt < dpt:
            dpt = temp_dpt
            dp_index = i
    return dpt, dp_index + 1


def directed_sspd_star(t1, t2):
    n1 = len(t1)
    n2 = len(t2)
    if n1 <= 2:
        return 0
    spd_list = []
    dp_index = 0
    for i in range(1, n1 - 1):

        if dp_index < (n2 - 1):
            spd, dp_index = point_to_trajectory(t1[i], t2)
        else:
            spd = euclidean_dist(t1[i], t2[-1])
            # print(n2, dp_index)
        spd_list.append(spd)
    return np.sum(spd_list) / (n1 - 2)


def sspd_star_dist(t1, t2, w1=0.1, w2=0.8, w3=0.1):
    """
    the implement of the SSPD* measure proposed by Zhang et al.(2021). SSPD* is improved based on the SSPD which considers
    the trajectory direction. The distance of endpoints and medium points are calculated separately and
    then their weighted average value is taken. It computes the distance of medium points in time series.
    :param t1: the first trajectory
    :param t2: the second trajectory
    :param w3:
    :param w2:
    :param w1:

    :return: the SSPD* distance
    """
    sspd_star_dist1 = directed_sspd_star(t1, t2)
    sspd_star_dist2 = directed_sspd_star(t2, t1)

    sspd_star_dist = w1 * euclidean_dist(t1[0], t2[0]) + w2 * ((sspd_star_dist1 + sspd_star_dist2) / 2) + \
                     w3 * euclidean_dist(t1[-1], t2[-1])
    return sspd_star_dist

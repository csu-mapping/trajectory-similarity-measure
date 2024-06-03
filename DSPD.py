#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np


def euclidean_dist(p1, p2):
    """
    :param p1:
    :param p2:
    :return:
    """
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


def point_to_trajectory(p, t, v):
    if len(p) == 0 or len(t) < 2:
        dpt = 9e100
        dpr = 9e100
        return dpt, dpr
    dpt = 9e100
    dpr = 0
    for i in range(0, len(t) - 1):
        s1 = t[i]
        s2 = t[i + 1]
        temp_dpt, _ = point_to_seg(p, s1, s2)
        if temp_dpt < dpt:
            dpt = temp_dpt
            dpr = i

    pt = t[dpr]
    pt_next = t[dpr + 1]
    temp = [pt_next[0] - pt[0], pt_next[1] - pt[1]]
    # v = 0
    dpr = 1 - (temp[0] * v[0] + temp[1] * v[1]) / np.sqrt(
        (temp[0] ** 2 + temp[1] ** 2) * (v[0] ** 2 + v[1] ** 2))
    if math.isnan(dpr):
        print(temp, v)
        print(t)
    return dpt, dpr


def calculate_direction_similarity(t_pts, t_pts_proj):

    dr = []
    for pt, pt_proj in zip(t_pts, t_pts_proj):
        numerator = pt[0] * pt_proj[0] + pt[1] * pt_proj[1]
        denominator = np.sqrt((pt[0] ** 2 + pt[1] ** 2) * (pt_proj[0] ** 2 + pt_proj[1] ** 2))

        dpr = 1 - numerator / denominator
        dpr = 1 - (pt[0] * pt_proj[0] + pt[1] * pt_proj[1]) / np.sqrt(
            (pt[0] ** 2 + pt[1] ** 2) * (pt_proj[0] ** 2 + pt_proj[1] ** 2))
        dr.append(dpr)
    return dr


def get_vector(s1, s2):
    return [s2[0] - s1[0], s2[1] - s1[1]]


def e_spd(t1, t2):
    n1 = len(t1)
    n2 = len(t2)
    if n1 == 0 or n2 == 0:
        ds = 9e100
        dr = 9e100
        return ds, dr
    else:
        ds = []
        dr = []
        direct_t1 = [[t1[i][0] - t1[i - 1][0], t1[i][1] - t1[i - 1][1]] for i in range(1, len(t1))]

        direct_t1.append(direct_t1[-1])

        for idx, d in enumerate(direct_t1):
            if d == [0, 0]:
                if idx > 0:
                    direct_t1[idx] = direct_t1[idx - 1]
                else:
                    direct_t1[idx] = direct_t1[idx + 1]

        for i in range(n1):
            ds_i, dr_i = point_to_trajectory(t1[i], t2, direct_t1[i])
            ds.append(ds_i)
            dr.append(dr_i)

    ds = np.mean(ds)
    dr = np.mean(dr)

    return ds, dr


def directed_spd(t1, t2):
    EPS = 1e-20

    ds1, dr1 = e_spd(t1, t2)
    ds2, dr2 = e_spd(t2, t1)

    d = 2 / (2 - min(dr1, dr2) + EPS) * max(min(ds1, ds2), + EPS)
    return d


def sure_t_type(t):
    t = [[float(pt[0]), float(pt[1])] for pt in t]
    return t


def fast_directed_spd(t1, t2, step=2):
    EPS = 1e-20
    t1 = sure_t_type(t1)
    t2 = sure_t_type(t2)
    ds1, dr1 = fast_e_spd(t1, t2, step=step)
    ds2, dr2 = fast_e_spd(t2, t1, step=step)

    d = 2 / (2 - min(dr1, dr2) + EPS) * max(min(ds1, ds2), + EPS)
    return d


def fast_e_spd(t1, t2, step=2):
    n1 = len(t1)
    n2 = len(t2)
    if n1 == 0 or n2 == 0:
        ds = 9e100
        dr = 9e100
        return ds, dr
    else:
        direct_t1 = [[t1[i][0] - t1[i - 1][0], t1[i][1] - t1[i - 1][1]] for i in range(1, len(t1))]
        direct_t1.append(direct_t1[-1])
        ds = []
        dr = []
        for i in range(n1):
            ds_i, dr_i = fast_point_to_trajectory(t1[i], t2, direct_t1[i], step=step)
            ds.append(ds_i)
            dr.append(dr_i)

    ds = np.mean(ds)
    dr = np.mean(dr)

    return ds, dr


def fast_point_to_trajectory(p, t, v, step=2):
    if len(p) == 0 or len(t) < 2:
        dpt = 9e100
        dpr = 9e100
        return dpt, dpr
    dpt = 9e100
    min_dist = 9e100
    min_idx = 0
    for i in range(len(t)):
        dist = euclidean_dist(p, t[i])
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    segs = [[t[idx], t[idx+1]] for idx in range(max(0, min_idx-step), min(min_idx+step, len(t)-1), 1)]
    for seg in segs:
        dist, _ = point_to_seg(p, seg[0], seg[1])

        if dist < dpt:
            dpt = dist
            temp = [seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]]
            dpr = 1 - (temp[0] * v[0] + temp[1] * v[1]) / np.sqrt(
                (temp[0] ** 2 + temp[1] ** 2) * (v[0] ** 2 + v[1] ** 2))

    return dpt, dpr


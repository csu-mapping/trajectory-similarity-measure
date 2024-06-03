import math

import numpy as np

from pydist.lcss import e_lcss


def directed_lcss(t1, t2):
    t1_d = calculate_t_direction(t1, t2)
    t2_d = calculate_t_direction(t2, t1)
    d_sim = 1 - 1 / (len(t1) + len(t2)) * (sum(t1_d) + sum(t2_d))
    return d_sim


def calculate_t_direction(t1, t2):
    t1_d = []
    accumulated_length, pts_length = calculate_accumulate_length(t1)

    for idx, pt in enumerate(t1):
        length_ratio = accumulated_length[idx]
        y1 = calculate_distance_function(length_ratio, t1)
        y2 = calculate_distance_function(length_ratio, t2)
        if idx == len(t1)-1:
            t1_d.append(t1_d[-1])
        else:
            t1_d.append(min(abs(y1 - y2), 360 - abs(y1 - y2)) / 180)

    return t1_d


def calculate_distance_function(x, t):
    accumulated_length, pts_length = calculate_accumulate_length(t)
    conditions = [x_i <= x < x_j for x_i, x_j in zip(accumulated_length[0: len(t) - 1], accumulated_length[1:])]
    functions = [calculate_bearing(p1, p2) for p1, p2 in zip(t[0: len(t) - 1], t[1:])]

    y = np.piecewise(x, conditions, functions)
    return y


def calculate_accumulate_length(t):
    accumulated_length = [0]
    pts_length = []
    for p1, p2 in zip(t[: len(t) - 1], t[1:]):
        pts_length.append(calculate_distance(p1, p2))
    total_length = sum(pts_length)
    for idx in range(len(pts_length)):
        length_ratio = accumulated_length[-1] + pts_length[idx] / total_length
        accumulated_length.append(length_ratio)
        idx += 1
    return accumulated_length, pts_length


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_bearing(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # 计算x和y坐标差值
    dx = int(x2) - int(x1)
    dy = int(y2) - int(y1)
    # 计算方位角
    bearing = math.atan2(dy, dx)
    # 将方位角转换为度数制
    bearing = math.degrees(bearing)
    if bearing >= 0:
        bearing = int(bearing + 0.5)
    else:
        # 这里角度取值范围是-180-180,你也可以改为0-360
        bearing = 180 - bearing
        bearing = int(bearing+0.5)

    return bearing


def e_lcst(t1, t2, eps):

    d_lcss = e_lcss(np.array(t1), np.array(t2), eps)
    t1 = np.array([[p[1], p[0]] for p in t1])
    t2 = np.array([[p[1], p[0]] for p in t2])
    d_sim = directed_lcss(t1, t2)

    return 1-(0.5 * (1-d_lcss) + 0.5 * d_sim)


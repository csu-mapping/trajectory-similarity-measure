#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: euclidean_distance.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2023/6/12 9:38

import numpy as np
from scipy.interpolate import interp1d


def align_trajectories(traj1, traj2):
    # 对齐轨迹长度，使其长度相同
    len1 = len(traj1)
    len2 = len(traj2)
    max_len = max(len1, len2)

    if len1 < max_len:
        traj1 = interpolate_trajectory(traj1, max_len)
    elif len2 < max_len:
        traj2 = interpolate_trajectory(traj2, max_len)

    return traj1, traj2


def interpolate_trajectory(trajectory, target_length):
    # 使用插值技术对轨迹进行插值，使其长度达到目标长度
    # 使用插值技术对轨迹进行插值，使其长度达到目标长度
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    indices = np.linspace(0, len(x) - 1, target_length)
    f = interp1d(np.arange(len(x)), x)
    interpolated_x = f(indices)
    f = interp1d(np.arange(len(y)), y)
    interpolated_y = f(indices)
    interpolated_trajectory = np.column_stack((interpolated_x, interpolated_y))
    return interpolated_trajectory


def euclidean_dist(traj1, traj2):
    # 计算两条轨迹的欧式距离
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    aligned_traj1, aligned_traj2 = align_trajectories(traj1, traj2)
    traj_length = len(aligned_traj1)
    if traj_length > 0:
        dist = np.mean([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) for p1, p2 in zip(aligned_traj1, aligned_traj2)])
    else:
        dist = 999999
    return dist


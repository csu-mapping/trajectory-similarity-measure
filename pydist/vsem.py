#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: vsem.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2024/1/29 13:52

import math
import numpy as np
from pydist.euclidean_distance import align_trajectories


def VSEM_dist(t1, t2, w1=0.5, w2=0.5, D=5):
    """
    the implementation of the similarity measure for paths in the paper "Generating lane-based intersection maps from
    crowd-sourcing big trace data"
    This measure is used for calculating similarity of turning change point pairs(TCPPs) which is based on the spatial distance and angle of two
    Since it is vector segment-based, we adopt interpolation to make two trajectories has the same length
    (i.e., having the same number of trajectory points or say the same number of pairwise segments of t1 and t2).

    :parameters
    - t1:
    - t2:
    - w1:
    - w2:
    - D:

    :return
    the similarity between t1 and t2
    """

    t1, t2 = align_trajectories(np.array(t1), np.array(t2))
    segment_sims = []
    for i in range(1, len(t1)):
        v1 = (t1[i][0]-t1[i-1][0], t1[i][1]-t1[i-1][1])
        v2 = (t2[i][0]-t2[i-1][0], t2[i][1]-t2[i-1][1])
        v1_len = math.hypot(v1[0], v1[1])
        v2_len = math.hypot(v2[0], v2[1])
        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
        diffA = 1 - dot_prod / (v1_len * v2_len)
        d_last_i = np.sqrt((t1[i-1][0]-t2[i-1][0])**2 + (t1[i-1][1]-t2[i-1][1])**2)
        d_i = np.sqrt((t1[i][0] - t2[i][0]) ** 2 + (t1[i][1] - t2[i][1]) ** 2)

        diffD = (d_last_i + d_i)/(2*D)

        sim = w1 * np.exp(-diffD) + w2 * np.exp(-diffA)
        segment_sims.append(sim)
    sim = np.mean(segment_sims)
    if math.isnan(sim):
        print('!!!!!! is nan', t1, t2)
    return sim

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: trajectory_retrival.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2024/1/10 17:59
import time

import numpy as np

from calcualte_distance_matrix import get_trajectories
from utils import *
from DSPD import directed_spd
from pydist.lcss import e_lcss
from pydist.lcst import e_lcst
from pydist.hausdorff import e_hausdorff, e_directed_hausdorff
from pydist.discret_frechet import discret_frechet
from pydist.sspd import e_sspd
from pydist.dtw import e_dtw
from pydist.sowd import owd_grid_brut
from added_baselines import Yang_dist
from pydist.euclidean_distance import euclidean_dist
from scipy.io import savemat, loadmat


def calculate_case2_distance_matrix():
    saved_folder = r'caseStudyData\case2_dist_matrix'
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    shp_data = read_shapefile(
        r'caseStudyData\case2\selected_126_trajectories.shp')
    geos = shp_data['geometry'].values.tolist()

    dist_metrics = ['dspd', 'lcss', 'lcst', 'hausdorff']
    trajs = get_trajectories(geos, simplified_flag=True)
    for metric in dist_metrics[3:4]:

        dist_matrix = np.zeros((len(geos), len(geos)))
        if metric == 'dspd':
            t1 = time.time()
            for i in range(len(trajs)):
                t1 = np.array(trajs[i])
                for j in range(i + 1, len(trajs)):
                    t2 = np.array(trajs[j])
                    dist = directed_spd(t1, t2)
                    dist_matrix[i, j] = dist
                print(f'metric:{metric}, {i} has completed')
            t2 = time.time()
            dist_matrix = dist_matrix + dist_matrix.T
            savemat(f'{saved_folder}\\{metric}.mat', {f'{metric}': dist_matrix, 'cost_time': t2 - t1})

        elif metric == "lcss":
            eps_list = [50, 200]
            for eps in eps_list:
                dist_matrix = np.zeros((len(geos), len(geos)))
                t1 = time.time()
                for i in range(len(trajs)):
                    t1 = np.array(trajs[i])
                    for j in range(i + 1, len(trajs)):
                        t2 = np.array(trajs[j])
                        dist = e_lcss(t1, t2, eps)
                        dist_matrix[i, j] = dist
                    print(f'metric:{metric}, eps: {eps}, {i} has completed')

                t2 = time.time()
                dist_matrix = dist_matrix + dist_matrix.T
                savemat(f'{saved_folder}\\{metric}_{eps}.mat', {f'{metric}': dist_matrix, 'cost_time': t2 - t1})

        elif metric == "lcst":
            eps_list = [50, 200]
            for eps in eps_list:
                dist_matrix = np.zeros((len(geos), len(geos)))
                t1 = time.time()
                for i in range(len(trajs)):
                    t1 = np.array(trajs[i])
                    for j in range(i + 1, len(trajs)):
                        t2 = np.array(trajs[j])
                        dist = e_lcst(t1, t2, eps)
                        dist_matrix[i, j] = dist
                    print(f'metric:{metric}, eps: {eps}, {i} has completed')
                t2 = time.time()
                dist_matrix = dist_matrix + dist_matrix.T
                savemat(f'{saved_folder}\\{metric}_{eps}.mat', {f'{metric}': dist_matrix, 'cost_time': t2 - t1})

        elif metric == "hausdorff":
            t1 = time.time()
            for i in range(len(trajs)):
                t1 = np.array(trajs[i])
                for j in range(i + 1, len(trajs)):
                    t2 = np.array(trajs[j])
                    dist = e_hausdorff(t1, t2)
                    dist_matrix[i, j] = dist
                print(f'metric:{metric}, {i} has completed')
            t2 = time.time()
            dist_matrix = dist_matrix + dist_matrix.T
            savemat(f'{saved_folder}\\{metric}.mat', {f'{metric}': dist_matrix, 'cost_time': t2 - t1})


def selected_top_k_trajectories(metric, traj_id, k=5, eps=500, saved=False):
    folder = r'caseStudyData\case2_dist_matrix'
    try:
        if metric in ['lcss', 'lcst']:
            dist_matrix = load_mat(f'{folder}\\{metric}_{eps}.mat')[metric]
        else:
            dist_matrix = load_mat(f'{folder}\\{metric}.mat')[metric]
    except Exception as e:
        print(e.args)
        return 0

    dist_arr = dist_matrix[traj_id-1]
    indices = np.argpartition(dist_arr, k+1)[:k+1]
    indices = np.delete(indices, np.where(indices == traj_id-1))

    k_smallest_values = dist_arr[indices]
    sorted_indices = indices[np.argsort(k_smallest_values)]
    sorted_indices += 1
    k_smallest_values.sort()
    if metric in ['lcss', 'lcst']:
        # Convert it to a similarity value, the larger the better
        k_smallest_values = [1-value for value in k_smallest_values]
    k_smallest_values = [round(value, 4) for value in k_smallest_values]
    print(metric, list(sorted_indices), k_smallest_values)
    if saved:
        save_top_k_trajectories(shp_data, metric, sorted_indices, traj_id=traj_id, eps=eps)


def save_top_k_trajectories(data_shp, metric, indices, traj_id, eps=200):
    saved_folder = r'results\case2'
    if not os.path.exists(f'{saved_folder}\\{traj_id}'):
        os.mkdir(f'{saved_folder}\\{traj_id}')

    indices = list(indices-1)
    saved_indices = indices[:5]
    saved_indices.append(traj_id-1)
    data_shp = data_shp.loc[saved_indices]
    proximity_level = [1, 2, 3, 4, 5, 0]
    data_shp['proximity_level'] = proximity_level
    if metric in ['lcss', 'lcst']:
        metric = f'{metric}_{eps}'
    save_shapefile(filename=f'{saved_folder}\\{traj_id}\\{traj_id}_{metric}.shp', data=data_shp)


if __name__ == "__main__":
    # you should first get or calculate the distance matrix using the function <calculate_case2_distance_matrix>
    dist_metrics = ['dspd', 'lcss', 'lcst', 'hausdorff']
    for metric in dist_metrics:
        # Trajectory 1. 69
        # Trajectory 2. 104
        # Trajectory 3. 15
        traj_id = 104
        eps = 200
        selected_top_k_trajectories(metric, traj_id=traj_id, k=10, eps=eps, saved=False)



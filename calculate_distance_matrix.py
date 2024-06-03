
import os
import time
import numpy as np
import pickle
import scipy.io as sci

from DSPD import directed_spd, fast_directed_spd
from pydist.discret_frechet import discret_frechet
from pydist.dtw import e_dtw
from pydist.edr import e_edr
from pydist.erp import e_erp
from pydist.euclidean_distance import euclidean_dist
from pydist.frechet import frechet
from pydist.hausdorff import e_directed_hausdorff, e_hausdorff
from pydist.lcss import e_lcss
from pydist.lcst import e_lcst
from pydist.sowd import owd_grid_brut
from pydist.sspd import e_sspd
from pydist.vsem import VSEM_dist
from pydist.sspd_star import sspd_star_dist


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_trajectories(traj_geometries):
    trajs = []
    for idx, geo in enumerate(traj_geometries):
        if str(geo.type) == "MultiLineString":
            geo_coords = [[coord[0], coord[1]] for coord in geo[0].coords]
        else:
            geo_coords = [[coord[0], coord[1]] for coord in geo.coords]
        if len(geo_coords) < 5:
            continue
        geo_coords2 = [(round(coord[0], 3), round(coord[1], 3)) for coord in geo_coords
                       if [round(coord[0], 3), round(coord[1], 3)] not in geo_coords2]
        trajs.append(remove_duplicates(geo_coords2))
    return trajs


def load_list(file_name):
    with open(file_name, 'rb') as file:
        trajs = pickle.load(file)
    return trajs


def save_list(file_name, trajs):
    with open(file_name, 'wb') as file:
        pickle.dump(trajs, file)


def calculate_distance_matrix(trajs, metric='dspd', dataset='', saved_folder=None):
    trajs_numb = len(trajs)
    dist_matrix = np.zeros((trajs_numb, trajs_numb))
    if metric == 'dspd':
        t0 = time.time()
        for i in range(0, len(trajs)):
            t1 = trajs[i]
            for j in range(i + 1, len(trajs)):
                t2 = trajs[j]
                dist = directed_spd(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        dist_matrix = dist_matrix + dist_matrix.T
        t1 = time.time()
        delta_t = t1 - t0
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == 'fast_dspd':
        t0 = time.time()
        for i in range(0, len(trajs)):
            t1 = trajs[i]
            for j in range(i + 1, len(trajs)):
                t2 = trajs[j]
                dist = fast_directed_spd(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        dist_matrix = dist_matrix + dist_matrix.T
        t1 = time.time()
        delta_t = t1 - t0
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "lcss":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        for eps in eps_list:
            t0 = time.time()
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = e_lcss(t1, t2, eps)
                    dist_matrix[i][j] = dist
            dist_matrix = dist_matrix + dist_matrix.T
            t1 = time.time()
            delta_t = t1 - t0
            sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric),
                        {metric: dist_matrix, "cost_time": delta_t})

    if metric == "lcst":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        for eps in eps_list:
            t0 = time.time()
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = e_lcst(t1, t2, eps)
                    dist_matrix[i][j] = dist
                print('eps: {}, {} has completed'.format(eps, i))
            t1 = time.time()
            delta_t = t1 - t0
            dist_matrix = dist_matrix + dist_matrix.T
            sci.savemat(r'{}\{}_{}_{}.mat'.format(saved_folder, dataset, metric, eps),
                        {metric: dist_matrix, "cost_time": delta_t})

    if metric == "hausdorff":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_hausdorff(np.array(t1), np.array(t2))
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "directed_hausdorff":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_directed_hausdorff(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "dtw":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_dtw(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "discrete_frechet":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = discret_frechet(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "frechet":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = frechet(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "sspd":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_sspd(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        dist_matrix = dist_matrix + dist_matrix.T
        t1 = time.time()
        delta_t = t1 - t0
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "sowd":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist1 = owd_grid_brut(np.array(t1), np.array(t2))
                dist2 = owd_grid_brut(np.array(t2), np.array(t1))
                dist = (dist1 + dist2) / 2
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        dist_matrix = dist_matrix + dist_matrix.T
        t1 = time.time()
        delta_t = t1 - t0
        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "edr":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        for eps in eps_list:
            t0 = time.time()
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = e_edr(t1, t2, eps)
                    dist_matrix[i][j] = dist
                print('{} has completed'.format(i))
            t1 = time.time()
            delta_t = t1 - t0
            dist_matrix = dist_matrix + dist_matrix.T
            sci.savemat(r'{}\{}_{}_{}.mat'.format(saved_folder, dataset, metric, eps),
                        {metric: dist_matrix, "cost_time": delta_t})

    if metric == "euclidean":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = euclidean_dist(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T

        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "VSEM":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = euclidean_dist(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T

        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    if metric == "sspd_star":
        t0 = time.time()
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = sspd_star_dist(t1, t2)
                dist_matrix[i][j] = dist
            print('{} has completed'.format(i))
        t1 = time.time()
        delta_t = t1 - t0
        dist_matrix = dist_matrix + dist_matrix.T

        sci.savemat(r'{}\{}_{}.mat'.format(saved_folder, dataset, metric), {metric: dist_matrix, "cost_time": delta_t})

    return dist_matrix


def calculate_simulated_dataset_distance_matrix():
    datasets = ['cross', 'i5', 'i5C', 'i5sim', 'i5simC']
    metrics = ['hausdorff', 'dtw', 'discret_frechet', 'sspd', 'lcss', 'lcst', 'sowd', 'edr', 'euclidean', 'sspd_star', 'VSEM', 'dspd', 'fast_dspd']
    for dataset in datasets[0: 0]:
        for metric in metrics:
            data_shp = read_shapefile(r'data\{}\{}.shp'.format(dataset, dataset))
            data_geometry = data_shp.geometry.values.tolist()
            trajs = get_trajectories(data_geometry)
            saved_folder = r'simulatedData\{}'.format(dataset)
            if not os.path.exists(saved_folder):
                os.mkdir(saved_folder)
            calculate_distance_matrix(trajs, metric, dataset=dataset, saved_folder=saved_folder)


def calculate_geolife_distance_metric():
    datasets = ['RI1', 'RI2', 'RI3', 'RI4']
    metrics = ['dspd', 'hausdorff', 'lcst']
    for dataset in datasets:
        for metric in metrics:
            filename = r'{}\{}\{}.shp'.format(folder, dataset, dataset)
            shp = read_shapefile(filename)
            shp_geos = shp['geometry'].values.tolist()
            trajs = get_trajectories(shp_geos)
            saved_folder = r'caseStudyData\{}_geolife'.format(dataset)
            if not os.path.exists(saved_folder):
                os.mkdir(saved_folder)
            calculate_distance_matrix(trajs, metric, dataset=dataset, saved_folder=saved_folder)


if __name__ == "__main__":
    # calculate the distance matrix of the dspd and baselines
    # calculate_simulated_dataset_distance_matrix()
    pass
   

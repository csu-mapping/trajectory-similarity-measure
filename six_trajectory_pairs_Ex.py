
from shapely.geometry import LineString

from DSPD import directed_spd

from pydist.dtw import *
from pydist.frechet import *
from pydist.discret_frechet import *
from pydist.lcss import e_lcss
from pydist.sspd import *
from pydist.sowd import *
from pydist.hausdorff import *
from pydist.edr import *
from pydist.erp import *
from pydist.lcst import *
from pydist.euclidean_distance import euclidean_dist
from pydist.vsem import VSEM_dist
from pydist.sspd_star import sspd_star_dist
from utils import load_mat


def calculate_distance_matrix(trajs, metric='dspd'):
    trajs_numb = len(trajs)
    dist_matrix = np.zeros((trajs_numb, trajs_numb))
    if metric == 'dspd':

        for i in range(0, len(trajs)):
            t1 = trajs[i]
            for j in range(i + 1, len(trajs)):
                t2 = trajs[j]
                dist = directed_spd(t1, t2)
                dist_matrix[i][j] = dist
            # print('{} has completed'.format(i))
        dist_matrix = dist_matrix + dist_matrix.T

        return dist_matrix

    if metric == "lcss":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        eps_list = [4.9]
        for eps in eps_list:

            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = 1-e_lcss(t1, t2, eps)
                    dist_matrix[i][j] = dist
            dist_matrix = dist_matrix + dist_matrix.T

        return dist_matrix

    if metric == "lcst":
        eps_list = [2, 5, 10, 15, 20, 25, 30, 35]
        eps_list = [4.9]
        for eps in eps_list:

            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = 1-e_lcst(t1, t2, eps)
                    dist_matrix[i][j] = dist

            dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "hausdorff":

        for i in range(len(trajs)):
            t1 = LineString(trajs[i])
            for j in range(i + 1, len(trajs)):              
                t2 = LineString(trajs[j])
                dist = t1.hausdorff_distance(t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T

        return dist_matrix

    if metric == "directed_hausdorff":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_directed_hausdorff(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "dtw":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_dtw(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "discrete_frechet":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = discret_frechet(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "frechet":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = frechet(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "sspd":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = e_sspd(t1, t2)
                dist_matrix[i][j] = dist
        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "sowd":

        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist1 = owd_grid_brut(np.array(t1), np.array(t2))
                dist2 = owd_grid_brut(np.array(t2), np.array(t1))
                dist = (dist1 + dist2) / 2
                dist_matrix[i][j] = dist
        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "edr":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        eps_list = [5.1]
        for eps in eps_list:

            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = 1-e_edr(t1, t2, eps)
                    dist_matrix[i][j] = dist

            dist_matrix = dist_matrix + dist_matrix.T
            return dist_matrix

    if metric == "erp":
        eps_list = [2, 5, 10, 15, 20, 25, 30]
        for eps in eps_list:

            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    t1 = trajs[i]
                    t2 = trajs[j]
                    dist = e_erp(t1, t2, eps)
                    dist_matrix[i][j] = dist

            dist_matrix = dist_matrix + dist_matrix.T
            return dist_matrix

    if metric == "euclidean":

        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = euclidean_dist(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix
    
    if metric == "VSEM":

        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = VSEM_dist(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    if metric == "sspd_star":

        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                t1 = trajs[i]
                t2 = trajs[j]
                dist = sspd_star_dist(t1, t2)
                dist_matrix[i][j] = dist

        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    return dist_matrix


def main():
    data = load_mat(r'simulatedData\six_trajectory_pairs.mat')
    T0 = list(data['T0'])
    T1 = list(data['T1'])
    T2 = list(data['T2'])
    T3 = list(data['T3'])
    T4 = list(data['T4'])
    T5 = list(data['T5'])
    T6 = list(data['T6'])
    T7 = list(data['T7'])
    metrics = ['hausdorff', 'dtw', 'discret_frechet', 'sspd', 'sowd', 'lcss', 'lcst', 'edr', 'euclidean', 'VSEM', 'sspd_star', 'dspd']
    for metric in metrics:
        # T1, T2, T3, T4
        for t2 in [T1, T2, T3, T4]:
            dist = calculate_distance_matrix([T0, t2], metric)
            print(dist[0][0])
        # T5, T6
        for t2 in [T5, T6]:
            dist = calculate_distance_matrix([T5, t2], metric)
            print(dist[0][0])
        

if __name__ == "__main__":
    # main()
    pass

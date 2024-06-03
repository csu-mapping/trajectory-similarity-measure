import os

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_mutual_info_score, adjusted_rand_score
from cluster_evaluation import Adjusted_Rand_index, NMI_index, get_cluster_labels_pred

from pydist.lcst import *
import scipy.io as sci
from utils import read_shapefile, evaluate_predicted_clusters, evaluate_real_clusters, \
    save_shapefile
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan


def get_trajectories(traj_geometries):
    trajs = []
    for geo in traj_geometries:

        if str(geo.type) == "MultiLineString":
            geo_coords = [[coord[0], coord[1]] for coord in geo[0].coords]
        else:
            geo_coords = [[coord[0], coord[1]] for coord in geo.coords]
        trajs.append(geo_coords)

    return trajs


def clustering():
    datasets = ['cross', 'i5', 'i5sim', 'i5C', 'i5simC']
    metrics = ['hausdorff', 'dtw', 'discret_frechet', 'sspd', 'lcss', 'lcst', 'sowd', 'edr', 'euclidean', 'sspd_star',
               'VSEM', 'dspd', 'fast_dspd']
    lcss_metrics = ['lcss_{}'.format(d) for d in [2, 5, 10, 20, 25, 30]]
    edr_metrics = ['edr_{}'.format(d) for d in [2, 5, 10, 20, 25, 30]]
    lcst_metrics = ['lcst_{}'.format(d) for d in [2, 5, 10, 20, 25, 30]]
    metrics.extend(lcss_metrics)
    metrics.extend(edr_metrics)
    metrics.extend(lcst_metrics)
    methods = ['hdbscan', 'hierarchical_ward']
    #
    for method in methods:
        for dataset in datasets:
            for metric in metrics:
                if metric.find("lcss") > -1 or metric.find('lcst') > -1 or metric.find('edr') > -1:
                    metric_ = metric.split('_')[0]
                    dist = sci.loadmat(r'simulatedData\{}\{}_{}'.format(dataset, dataset, metric))[metric_]
                else:
                    dist = sci.loadmat(r'simulatedData\{}\{}_{}.mat'.format(dataset, dataset, metric))[metric]
                data_shp = read_shapefile(r'simulatedData\{}\{}.shp'.format(dataset, dataset))
                data_shp['Id'] = [i for i in range(len(data_shp))]
                real_clusters = evaluate_real_clusters(data_shp)
                real_labels = data_shp.IDX.values.tolist()
                # min_cluster_size requires adjusting to obtain the optimal clustering results
                if method == 'hdbscan':
                    clusters = hdbscan.HDBSCAN(min_cluster_size=4, metric='precomputed',
                                               gen_min_span_tree=True).fit(dist)
                elif method == 'hierarchical_ward':
                    linkage = method.split('_')[-1]
                    clusters = AgglomerativeClustering(n_clusters=len(real_clusters), linkage=linkage,
                                                       affinity='euclidean').fit(dist)
                else:
                    return None

                data_shp['new_IDX'] = clusters.labels_
                clusters_pred = evaluate_predicted_clusters(data_shp)
                clusters = clusters_pred
                labels_pred = get_cluster_labels_pred(clusters)
                ARI = Adjusted_Rand_index(clusters, real_labels)
                NMI, AMI = NMI_index(clusters, real_labels)
                SI = silhouette_score(dist, labels_pred, metric='precomputed')

                if -1 in labels_pred:
                    predicted_clusters_length = len(clusters) - 1
                else:
                    predicted_clusters_length = len(clusters)
                print('predicted clusters: ', predicted_clusters_length)
                print('real clusters: ', len(real_clusters))
                metrics_json = {'dataset': dataset, 'metric': metric, 'method': method, 'eps': 'None',
                                'real_clusters': len(real_clusters),
                                'predicted_clusters': predicted_clusters_length,
                                'SI': SI, 'ARI': ARI, 'AMI': AMI}

                df: DataFrame = pd.json_normalize(metrics_json)
                saved_folder = r'results\simulated_datasets'
                if not os.path.exists(saved_folder):
                    os.mkdir(saved_folder)
                if os.path.exists(r'{}\simulated_datasets_clustering.csv'.format(saved_folder)):
                    df.to_csv(r'{}\simulated_datasets_clustering.csv'.format(saved_folder), header=False,
                              index=False, mode='a+')
                else:
                    df.to_csv(r'{}\simulated_datasets_clustering.csv'.format(saved_folder), index=False)
                saved_shp_folder = r'results\simulated_datasets\{}'.format(dataset)
                if not os.path.exists(saved_shp_folder):
                    os.mkdir(saved_shp_folder)
                save_shapefile(r'{}\{}_{}.shp'.format(saved_shp_folder, method, metric), data_shp)


if __name__ == "__main__":
    pass
    # clustering()

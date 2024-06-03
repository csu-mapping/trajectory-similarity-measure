
import os
import time
import hdbscan
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering
import scipy.io as sci
from sklearn.metrics import silhouette_score

from cluster_evaluation import Adjusted_Rand_index, NMI_index,  get_cluster_labels_pred
from utils import read_shapefile, save_shapefile, evaluate_real_clusters, \
    evaluate_predicted_clusters


def geolife_data_clustering_lcst():
    datasets = ['RI1', 'RI2', 'RI3', 'RI4']

    methods = ['hdbscan', 'hierarchical_average', 'hierarchical_ward']
    eps_list = [2, 5, 10, 15, 20, 25, 30]
    min_cluster_size = 3
    for metric in ['lcss', 'lcst']:
        for method in methods[0:1]:
            for dataset in datasets[0:1]:
                for eps in eps_list:
                    filename = r'caseStudyData\{}_geolife\{}_geolife.shp'.format(dataset, dataset)
                    data_shp = read_shapefile(filename)
                    dist = sci.loadmat(
                        r'caseStudyData\{}_geolife\{}_{}_{}.mat.'.format(dataset, dataset, metric, eps))[
                        metric]
                    real_clusters = evaluate_real_clusters(data_shp)
                    if method == 'hdbscan':
                        clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed',
                                                   gen_min_span_tree=True).fit(dist)
                    elif method in ['hierarchical_average', 'hierarchical_ward']:
                        linkage = method.split('_')[-1]
                        clusters = AgglomerativeClustering(n_clusters=len(real_clusters),
                                                           linkage=linkage, affinity='euclidean').fit(dist)
                    data_shp['new_IDX'] = clusters.labels_
                    real_labels = data_shp.IDX.values.tolist()
                    clusters_pred = evaluate_predicted_clusters(data_shp)

                    clusters = clusters_pred
                    labels_pred = get_cluster_labels_pred(clusters)
                    SI = silhouette_score(dist, labels_pred, metric='precomputed')
                    ARI = Adjusted_Rand_index(clusters, real_labels)

                    NMI, AMI = NMI_index(clusters, real_labels)

                    labels_pred = get_cluster_labels_pred(clusters)
                    if -1 in labels_pred:
                        predicted_clusters_length = len(clusters) - 1
                    else:
                        predicted_clusters_length = len(clusters)

                    print('predicted clusters: ', len(clusters))
                    print('real clusters: ', len(real_clusters))
                    metrics_json = {'dataset': dataset, 'metric': metric, 'method': method, 'eps': eps, 'real_clusters':
                                    len(real_clusters), 'predicted_clusters': predicted_clusters_length,
                                    'SI': SI, 'ARI': ARI, 'AMI': AMI, 'min_cluster_size': min_cluster_size}
                    df: DataFrame = pd.json_normalize(metrics_json)
                    saved_folder = r'results\caseStudy1'
                    if not os.path.exists(saved_folder):
                        os.mkdir(saved_folder)
                    if os.path.exists(r'{}\caseStudy1.csv'.format(saved_folder)):
                        df.to_csv(r'{}\caseStudy1.csv'.format(saved_folder), header=False, index=False, mode='a+')
                    else:
                        df.to_csv(r'{}\caseStudy1.csv'.format(saved_folder), index=False)
                    saved_shp_folder = r'results\case1\shps'
                    if not os.path.exists(saved_shp_folder):
                        os.mkdir(saved_shp_folder)
                    save_shapefile(
                        r'{}\{}_geolife_{}_{}_{}_{}.shp'.format(saved_shp_folder, dataset, method, metric, eps,
                                                                min_cluster_size), data_shp)


def geolife_data_clustering_dspd_hausdorff():
    datasets = ['RI1', 'RI2', 'RI3', 'RI4']
    methods = ['hdbscan', 'hierarchical_average', 'hierarchical_ward']
    metric = 'dspd'
    metric = 'hausdorff'
    if not os.path.exists('results'):
        os.mkdir('results')
        
    min_cluster_size = 3
    for method in methods[0:1]:
        for dataset in datasets[1:]:
            filename = r'caseStudyData\{}_geolife\{}_geolife.shp'.format(dataset, dataset)
            data_shp = read_shapefile(filename)
            dist = sci.loadmat(
                r'caseStudyData\{}_geolife\{}_{}.mat.'.format(dataset, dataset, metric))[metric]
            real_clusters = evaluate_real_clusters(data_shp)

            if method == 'hdbscan':
                clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed',
                                           gen_min_span_tree=True).fit(dist)
            elif method in ['hierarchical_average', 'hierarchical_ward', 'hierarchical_single',
                            'hierarchical_complete']:
                linkage = method.split('_')[-1]
                clusters = AgglomerativeClustering(n_clusters=len(real_clusters),
                                                   linkage=linkage, affinity='euclidean').fit(dist)
            data_shp['new_IDX'] = clusters.labels_
            real_labels = data_shp.IDX.values.tolist()
            clusters_pred = evaluate_predicted_clusters(data_shp)
            clusters = clusters_pred
            labels_pred = get_cluster_labels_pred(clusters)
            SI = silhouette_score(dist, labels_pred, metric='precomputed')
            ARI = Adjusted_Rand_index(clusters, real_labels)
            NMI, AMI = NMI_index(clusters, real_labels)
            labels_pred = get_cluster_labels_pred(clusters)
            if -1 in labels_pred:
                predicted_clusters_length = len(clusters) - 1
            else:
                predicted_clusters_length = len(clusters)
            print('predicted clusters: ', len(clusters))
            print('real clusters: ', len(real_clusters))
            metrics_json = {'dataset': dataset, 'metric': metric, 'method': method, 'eps': 'None', 'real_clusters':
                            len(real_clusters), 'predicted_clusters': predicted_clusters_length,
                            'SI': SI, 'ARI': ARI, 'AMI': AMI, 'min_cluster_size': min_cluster_size}
            df: DataFrame = pd.json_normalize(metrics_json)
            saved_folder = r'results\caseStudy'
            if not os.path.exists(saved_folder):
                os.mkdir(saved_folder)
            if os.path.exists(r'{}\caseStudy.csv'.format(saved_folder)):
                df.to_csv(r'{}\caseStudy.csv'.format(saved_folder), header=False, index=False, mode='a+')
            else:
                df.to_csv(r'{}\caseStudy.csv'.format(saved_folder), index=False)
            saved_shp_folder = r'results\caseStudy\shps'
            if not os.path.exists(saved_shp_folder):
                os.mkdir(saved_shp_folder)
            save_shapefile(
                r'{}\{}_geolife_{}_{}_{}.shp'.format(saved_shp_folder, dataset, method, metric, min_cluster_size), data_shp)


if __name__ == "__main__":
    pass
    # geolife_data_clustering_dspd_hausdorff()
    # geolife_data_clustering_lcst()

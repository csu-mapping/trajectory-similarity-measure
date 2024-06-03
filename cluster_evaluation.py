import os.path

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score


def rand_index(clusters: dict, labels):
    labels_pred = get_cluster_labels_pred(clusters)
    ARI = rand_index(labels, labels_pred)
    print(r'ARI: {}'.format(ARI))
    return ARI


def NMI_index(clusters: dict, labels: list):
    """
    互信息指数, 外部指标
    :param clusters:
    :param labels:
    :return:
    """
    labels_pred = get_cluster_labels_pred(clusters)

    NMI = normalized_mutual_info_score(labels, labels_pred)
    AMI = adjusted_mutual_info_score(labels, labels_pred)
    NMI = round(NMI, 4)
    AMI = round(AMI, 4)
    # print(r'NMI: {}, AMI: {}'.format(NMI, AMI))
    return NMI, AMI


def Adjusted_Rand_index(clusters: dict, labels):
    """
    调整兰德指数, 越接近于1越好, 外部指标
    :param clusters:
    :param labels:
    :return:
    """
    labels_pred = get_cluster_labels_pred(clusters)

    ARI = adjusted_rand_score(labels, labels_pred)
    # print(r'ARI: {}'.format(ARI))
    return round(ARI, 4)


def get_cluster_labels_pred(clusters):
    traj_ids_labels = {}
    for label, cluster in clusters.items():
        for traj_id in cluster:
            traj_ids_labels[traj_id] = label

    traj_ids_labels = sorted(traj_ids_labels.items(), key=lambda x: x[0])
    labels_pred = [item[1] for item in traj_ids_labels]
    return labels_pred


def evaluate_real_data(df: pd.DataFrame):
    groups = df.groupby(by=['IDX'])
    clusters = {}
    for group in groups:
        IDX = group[0]
        Ids = group[1].Id.values.tolist()
        clusters[IDX] = Ids

    return clusters

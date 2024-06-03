
import os
import re, csv, networkx as nx
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.io as scio
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  
a = 6378245.0  
ee = 0.00669342162296594323  


def save_shapefile(filename, data, encoding='utf-8'):
    data.to_file(filename, encoding=encoding)


def read_shapefile(filename, encoding='latin1'):
    data = gpd.GeoDataFrame.from_file(filename, encoding=encoding)
    return data


def save_list(data, filename):
    np.save(filename, data)


def read_list(filename):
    return np.load(filename, allow_pickle=True)


def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as fp:
        fp.write(str(data))


def load_json(filename):
    globals = {
        'nan': 0
    }
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
        data = eval(data, globals)
    return data


def parse(filename, isDirected):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = [row for row in reader]

    print("Reading and parsing the data into memory...")
    if isDirected:
        return parse_directed(data)
    else:
        return parse_undirected(data)


def parse_undirected(data):
    G = nx.Graph()
    nodes = set([row[0] for row in data])
    edges = [(row[0], row[2]) for row in data]

    num_nodes = len(nodes)
    rank = 1 / float(num_nodes)
    G.add_nodes_from(nodes, rank=rank)
    G.add_edges_from(edges)
    pr = nx.pagerank(G, alpha=1)
    return G


def parse_directed(data):
    DG = nx.DiGraph()

    for i, row in enumerate(data):

        node_a = format_key(row[0])
        node_b = format_key(row[2])
        val_a = digits(row[1])
        val_b = digits(row[3])

        DG.add_edge(node_a, node_b)
        if val_a >= val_b:
            DG.add_path([node_a, node_b])
        else:
            DG.add_path([node_b, node_a])
    return DG


def digits(val):
    return int(re.sub(r"\D", "", val))


def format_key(key):
    key = key.strip()
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    return key


def print_results(f, method, results):
    print(method)


def load_mat(filename):
    data = scio.loadmat(filename)
    return data


def json2csv(input_filename, output_filename):
    data = load_json(input_filename)
    df = pd.json_normalize(data)
    df.to_csv(output_filename, index=False)


def evaluate_predicted_clusters(df: pd.DataFrame):
    groups = df.groupby(by=['new_IDX'])
    clusters = {}
    for group in groups:
        IDX = group[0]
        try:
            Ids = group[1].Id.values.tolist()
        except:
            Ids = group[1].id.values.tolist()
        clusters[IDX] = Ids

    return clusters


def evaluate_real_clusters(df: pd.DataFrame):
    groups = df.groupby(by=['IDX'])
    clusters = {}
    for group in groups:
        IDX = group[0]
        try:
            Ids = group[1].Id.values.tolist()
        except:
            Ids = group[1].id.values.tolist()
        clusters[IDX] = Ids

    return clusters


def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    # " 经度: lon ; 纬度: lat "
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    # haversine 公式
    d_lng = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    dist = c * r * 1000
    return dist


def degree_to_meter(degree):
    return degree * (2 * math.pi * 6371004) / 360


def meter_to_degree(meter):
    # 米转为度
    cilometter = 0.0089932202929999989 // 1

    degree = meter / (2 * math.pi * 6371004) * 360
    return degree


def perpendicular_distance(point, line_start, line_end):
    return np.abs((line_end[1] - line_start[1]) * point[0] -
                  (line_end[0] - line_start[0]) * point[1] +
                  line_end[0] * line_start[1] - line_end[1] * line_start[0]) / np.sqrt(
        (line_end[1] - line_start[1]) ** 2 + (line_end[0] - line_start[0]) ** 2)


def douglas_peucker(points, epsilon):
    if len(points) <= 2:
        return points
    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        simplified1 = douglas_peucker(points[:index + 1], epsilon)
        simplified2 = douglas_peucker(points[index:], epsilon)
        simplified = np.vstack((simplified1[:-1], simplified2))
    else:
        simplified = np.array([points[0], points[end]])

    return simplified


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

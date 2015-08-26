import os
import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import elice_utils
import scipy.spatial.distance
from operator import itemgetter

def main():
    #1 #2
    stocks_df, code_to_name = load_data()

    #3
    stocks_df = stocks_df.transpose()

    # 4
    num_components = 10
    pca_array = run_PCA(stocks_df, num_components)

    # 5
    num_clusters = 20
    cluster_labels = run_kmeans(pca_array, num_clusters, list(range(0, num_clusters)))

    # 6
    display_cluster_idx = 0
    print(elice_utils.plot_stocks(stocks_df, pca_array, cluster_labels, code_to_name, display_cluster_idx))

    # 7
    print(get_closest_stocks(stocks_df, pca_array, code_to_name, "005930", 10))

def run_PCA(df, num_components):
    # 4
    pca = sklearn.decomposition.PCA(num_components)
    pca.fit(df)
    pca_array = pca.transform(df)

    return pca_array

def run_kmeans(pca_array, num_clusters, initial_centroid_indices):
    # 5
    initial_centroids = np.array([pca_array[i] for i in initial_centroid_indices])
    classifier = sklearn.cluster.KMeans(n_clusters = num_clusters, init = initial_centroids, n_init = 1)
    classifier.fit(pca_array)

    return classifier.labels_

def get_closest_stocks(df, pca_array, code_to_name, code, num_stocks_show):
    distance_lists = []
    code_index = df.index.values.tolist().index(code)
    code_xy = pca_array[code_index]

    for code, xy in zip(df.index.values, pca_array):
        distance = scipy.spatial.distance.euclidean(code_xy, xy)
        distance_lists.append([code, distance])

    distance_lists = sorted(distance_lists, key=itemgetter(1))
    return [code_to_name[code_xy[0]] for code_xy in distance_lists[0:num_stocks_show]]

def load_data():
    stocks_df = pd.read_csv("./stock_fluctuations.csv")
    stocks_df = stocks_df.set_index('index')
    krx_listed_companies = pd.read_csv("./krx_listed_companies.csv")

    code_to_name = {}
    for code, name in zip(krx_listed_companies['Code'].values, krx_listed_companies['Name'].values):
        z_code = '0' * (6 - len(str(code))) + str(code)
        code_to_name[z_code] = name

    return stocks_df, code_to_name

if __name__ == "__main__":
    main()
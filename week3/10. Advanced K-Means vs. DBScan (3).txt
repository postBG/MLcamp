import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import numpy as np
import pandas as pd
import elice_utils

def main():
    np.random.seed(108)

    # 1
    noisy_circles = pd.read_csv('noisy_circle.csv')
    noisy_circles = noisy_circles.set_index('index')
    blobs = pd.read_csv('blobs.csv')
    blobs = blobs.set_index('index')

    return draw_graph([noisy_circles, blobs])


def run_DBScan(X, eps):
    # 2
    dbscan = sklearn.cluster.DBSCAN(eps)
    dbscan.fit(X)

    return dbscan


def draw_graph(datasets, n_clusters=2):
    plot_num = 1
    alg_names = []
    eps_list = []

    for eps in np.arange(0.1, 1, step=0.1):
        alg_names.append('DBScan eps=%.1f' % eps)
        eps_list.append(eps)

    a = np.arange(len(alg_names)*2)+1
    indices = np.hstack(a.reshape(len(alg_names), 2).T)

    elice_utils.draw_init()

    for dataset in datasets:
        # 3
        tmp_y = dataset.pop('0')
        y = np.array(tmp_y)
        
        tmp_X = dataset
        X = np.array(tmp_X)

        for alg_name, eps in zip(alg_names, eps_list):
            dbscan_result = run_DBScan(X, eps)

            elice_utils.draw_graph(X, dbscan_result, alg_name, plot_num, len(alg_names), indices)
            plot_num += 1

    print(elice_utils.show_graph())

    return dbscan_result


if __name__ == '__main__':
    main()
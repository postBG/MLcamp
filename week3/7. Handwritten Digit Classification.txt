import numpy as np
import sklearn
import sklearn.metrics
import sklearn.cluster
import sklearn.datasets
import sklearn.decomposition
import sklearn.preprocessing
import elice_utils

def main():
    # 1
    digits = sklearn.datasets.load_digits()
    data = sklearn.preprocessing.scale(digits.data)
    
    # 2
    
    # Try looking at different digits by changing index from 0 to 1796.
    print(elice_utils.display_digits(digits, 113))

    # 4
    benchmark(data, digits.target, 1, 64)

def benchmark(data, ground_truth, components_min, components_max):
    np.random.seed(0)

    X = []
    Y = []
    for num_components in range(components_min, components_max):
        X.append(num_components)
        pca_array = run_PCA(data, num_components)
        estimated_classes = run_kmeans(pca_array, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # 4
        score = sklearn.metrics.homogeneity_score(ground_truth, estimated_classes)

        Y.append(score)

    print(elice_utils.benchmark_plot(X, Y))
    return Y

def run_PCA(df, num_components):
    # 3
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(df)
    pca_array = pca.transform(df)

    return pca_array

def run_kmeans(pca_array, num_clusters, initial_centroid_indices):
    # 3
    initial_centroids = np.array([pca_array[i] for i in initial_centroid_indices])
    classifier = sklearn.cluster.KMeans(n_clusters = num_clusters, n_init = 1, init = initial_centroids)

    classifier.fit(pca_array)
    return classifier.labels_

if __name__ == "__main__":
    main()
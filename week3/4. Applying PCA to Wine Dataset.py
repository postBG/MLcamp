import sklearn.decomposition
import numpy as np
import pandas as pd
import elice_utils

def main():
    # 1
    wine_df = pd.read_csv('wine.csv')

    class_df = wine_df.pop('class')
    feature_df = wine_df

    # 2
    pca, pca_array = run_PCA(feature_df, 2)

    # 4
    print(elice_utils.wine_graph(pca_array, class_df))


def run_PCA(dataframe, num_components):
    # 2
    pca = sklearn.decomposition.PCA(num_components)
    pca.fit(dataframe)
    pca_array = pca.transform(dataframe)
    # 3
    return pca, pca_array

def draw_graph(pca_array, class_df, class_names = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']):
    class_array = np.array(class_df)
    plt.figure()
    for c, i, class_name in zip("rgb", [1, 2, 3], class_names):
        plt.scatter(pca_array[class_array == i, 0], pca_array[class_array == i, 1], c=c, label=class_name)
    plt.legend(loc=4)
    plt.title('Principal Components of Wine Dataset')

    plt.show()


if __name__ == '__main__':
    main()
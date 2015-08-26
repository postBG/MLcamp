import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import sklearn.cross_validation
import scipy.spatial.distance
from operator import itemgetter

def main():
    stocks_df, stock_fluctuations_df, code_to_name = load_data()
    # Use stocks_df for your custom SVM classifier.

    features, labels = prepare_svm_features(stock_fluctuations_df, '000030')
    print(svm_with_cross_validation(features, labels))

    #performances, avg_performance = benchmark(stock_fluctuations_df, code_to_name)
    #print("Average performance is %.2lf%%" % (avg_performance * 100))

def benchmark(df, code_to_name):
    performances = []
    for quote in df:
        features, labels = prepare_svm_features(df, quote)
        if len(labels) < 1300: continue

        perf = svm_with_cross_validation(features, labels)
        performances.append((quote, code_to_name[quote], perf))

        print((quote, code_to_name[quote], "%.2f%%" % (perf * 100)))

    avg_performance = np.mean([x[2] for x in performances])
    return performances

def svm_with_cross_validation(features, labels):
    # Do not modify this line.
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(features, labels, test_size=0.2, random_state=0)

    # 3
    clf = sklearn.svm.SVC(kernel = 'rbf').fit(X_train, y_train)

    return clf.score(X_test, y_test)

def prepare_svm_features(df, code):
    #####################################################################
    # After completing the exercise, feel free to change everything     #
    # in this function and try to develop your custom stock fluctuation #
    # predictor! Please share your own features and algorithms.         #
    #####################################################################

    global NUM_DAYS
    global NUM_QUOTES

    # Let features be the fluctuations of all stocks
    features = df.values
    # Remove first and last fluctuations
    features = features[1:-1]

    # Let labels be the fluctuations of given stock code
    labels = df[code].values
    # Remove first 2 labels to match fluctuation of "tomorrow's label" to
    # "today's all stock fluctuations" as we want to predict tomorrow's
    # rise/decline given today's fluctuations
    labels = labels[2:]

    # Get the indices having non-NaN values in labels and not zero
    non_NaN_indices = [i for i in range(0, len(labels)) if \
                       not pd.isnull(labels[i]) and not labels[i] == 0]
    # Filter out NaN labels
    features = features[non_NaN_indices]
    labels = labels[non_NaN_indices]

    # Reduce real-number fluctuations into 3 classes in {-1, 0, 1}
    labels = [np.sign(x) for x in labels]

    return features, labels

def load_data():
    stocks_df = pd.read_csv("./stocks.csv")
    stocks_df = stocks_df.set_index('index')
    stocks_fl_df = pd.read_csv("./stock_fluctuations.csv")
    stocks_fl_df = stocks_fl_df.set_index('index')

    krx_listed_companies = pd.read_csv("./krx_listed_companies.csv")

    code_to_name = {}
    for code, name in zip(krx_listed_companies['Code'].values, krx_listed_companies['Name'].values):
        z_code = '0' * (6 - len(str(code))) + str(code)
        code_to_name[z_code] = name

    return stocks_df, stocks_fl_df, code_to_name

if __name__ == "__main__":
    main()
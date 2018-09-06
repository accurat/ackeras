import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time

from itertools import compress
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.decomposition import PCA

import pdb


class OutlierDetection:

    '''
    The class takes data and detects outliers, if the data has more than 3 dimensions the choice falls on
    using Isolation Forests, otherwise the OLF algorithm. Apart from the data the instantiating parameters are:
    contamination::float -> Value between 0 and 0.3 which rapresents the expected amount of outliers in the dataset
    timecolumn::String -> the name of the timecolumn data, if selected it is used mainly for plotting purposes and it is excluded from the analysis
    plot::boolean -> Do you want a nice seaborn plot?
    '''

    def __init__(self, data, contamination=0.05, timecolumn=None):
        if not isinstance(data, pd.DataFrame):
            raise AssertionError('The dataset is not a pandas dataframe')

        assert 0 < contamination < 0.3
        self.data = data
        self.dimensions = data.shape[1]
        self.contamination = contamination
        self.seed = int(time.mktime(datetime.now().timetuple()))

        if timecolumn:
            self.timeindex = data[timecolumn]
            self.data = data.drop(timecolumn, axis=1)
        else:
            self.timeindex = None
            numerical_types = ['int16', 'int32', 'int64',
                               'float16', 'float32', 'float64']
            self.data = data.select_dtypes(include=numerical_types)

    def isolation_forest_detector(self, n_estimators=2000):
        X = self.data

        parameters = {
            'n_estimators': n_estimators,
            'contamination': self.contamination,
            'max_features': 0.8,
            'random_state': self.seed,
        }

        detector = IsolationForest(**parameters)
        detector.fit(X)
        outliers = detector.predict(X)
        self.isolation_forest = detector
        X['outliners'] = [False if i == 1 else True for i in outliers]
        self.data_analysed = X

        return X

    def olf_detector(self, leaf_size=20, normalize=True):
        if normalize:
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(self.data)
            if X.shape[1] < 2:
                pca = PCA(n_components=2)
                X = pca.fit_transform(self.data)

            self.pca = pca

        else:
            X = self.data

        parameters = {
            'n_neighbors': int(0.05 * X.shape[0]),
            'leaf_size': leaf_size,
            'metric': 'minkowski' if X.shape[1] < 4 else 'cosine',
            'contamination': self.contamination
        }

        detector = LocalOutlierFactor(**parameters)
        outliers = detector.fit_predict(X)

        self.olf = detector
        if isinstance(X, pd.DataFrame):
            X['outliers'] = [False if i == 1 else True for i in outliers]
        else:
            X = pd.DataFrame(X, columns=['First_pc', 'Second_pc'])
            X['outliers'] = [False if i == 1 else True for i in outliers]

        self.data_analysed = X

        return X

    def fit_predict(self, plot=False):

        data_analysed = self.isolation_forest_detector(
        ) if self.dimensions > 10 else self.olf_detector()

        original_data = self.data

        if plot:
            if self.timeindex is not None:

                x = self.timeindex
                y_column = original_data.describe().std().argmax()
                y = original_data[y_column].tolist()
                plt.figure(figsize=(16, 12))
                plt.plot(x, y, c='blue', alpha=0.5)
                plt.xlabel('Time')
                plt.ylabel(y_column)

                filter_out = data_analysed['outliers'].tolist()
                outlier_x = list(compress(x, filter_out))
                outlier_y = list(compress(y, filter_out))

                plt.scatter(outlier_x, outlier_y, c='r', s=30)

                plt.show()

            else:

                x_column = original_data.describe().std().argmax()
                y_column = original_data.drop(
                    x_column, axis=1).describe().std().argmax()

                x = original_data[x_column].tolist()
                y = original_data[y_column].tolist()

                plt.figure(figsize=(16, 12))
                plt.plot(x, y, c='blue', alpha=0.5)

                plt.xlabel(x_column)
                plt.ylabel(y_column)

                filter_out = data_analysed['outliers'].tolist()
                outlier_x = list(compress(x, filter_out))
                outlier_y = list(compress(y, filter_out))

                plt.scatter(outlier_x, outlier_y, c='r', s=30)

                plt.show()

        return data_analysed


if __name__ == "__main__":
    data = pd.read_csv(
        '/Users/andreatitton/Desktop/places_processed.csv')
    detect_outliers = OutlierDetection(
        data, contamination=0.001)

    data_analysed = detect_outliers.fit_predict(plot=True)

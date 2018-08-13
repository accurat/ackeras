from auto_ml.data_cleaning import AccuratPreprocess
from auto_ml.dim_red import RedDimensionality
from auto_ml.clustering import Clustering
from auto_ml.regression import Regression

import time
import pandas as pd


def time_decorater(f):
    def wrapper():
        t1 = time.time()
        f()
        t2 = time.time()
        s = f"Time it took to run the function: {str((t2 - t1))}\n"
        print(s)
    return wrapper


class Pipeline():
    '''
    The parameters of the class are:
    - path: the path to the file
    - supervised: whether the issue is supervised or unsupervised (labelled data or not)
    - reg_class: a tuple consisting of the problem type (classification or regression or None, which equals to autodetect) and the dependent (target) column name 
    - categorical_feautures: a list of the categorical column names (if present)
    - timecolumn: the name of the time data (if present)
    - drop_rest: whether you want to drop all the unprocessed columns (suggested for analysis)
    - extreme_drop: if some column has to be dropped at the end of the preprocess 

    '''

    def __init__(self, path, supervised=False, categorical_feautures=None, timecolumn=None, drop_rest=True, extreme_drop=None, reg_class=None):
        self.path = path
        self.categorical_feautures = categorical_feautures
        self.timecolumn = timecolumn
        self.drop_rest = drop_rest
        self.extreme_drop = extreme_drop
        self.supervised = False
        self.reg_class = reg_class
        self.acp = None

    def preprocess(self):
        print('Preprocessing')
        assert isinstance(self.categorical_feautures, list)
        assert isinstance(self.timecolumn, str)

        self.acp = AccuratPreprocess(path=self.path)
        self.data_processed = self.acp.fit_transform(categorical_feautures=self.categorical_feautures, timecolumn=self.timecolumn,
                                                     save=False, drop_rest=self.drop_rest, outputplot=False, extreme_drop=self.extreme_drop)

        return self.data_processed, self.acp

    def dimensionality_reduction(self):
        # data_red = RedDimensionality(
        #    data_processed, categorical_feautures=acp.embedded_columns, analysis=True, avoid_pca=True).dim_reduction()
        # TODO - deal with data leakage, ref: https://machinelearningmastery.com/data-leakage-machine-learning/ (should be done)

        # data_red.to_csv('reduced_data.csv')

        return 'placeholder'

    def clustering(self):
        print('Clustering...')
        assert self.acp is not None
        self.cluster_class = Clustering(
            self.data_processed, categorical_feautures=self.acp.embedded_columns)
        self.clustered_data = self.cluster_class.fit_predict()
        return self.clustered_data, self.cluster_class

    def regression(self):
        print('Regressing...')
        assert self.acp is not None

        self.regress = Regression(
            self.data_processed, y=self.reg_class[1], problem=self.reg_class[0])

        self.opt_regres, returning = self.regress.fit_predict()

        if isinstance(returning, pd.DataFrame):
            self.labelled_data = returning
            return returning

        else:
            self.opt_coeff_ = returning
            return returning


# ----------TEST RUN-----------------------
# test_params = {'path': '/Users/andreatitton/accurat_places_analytics/ackeras/data/random_data_places.csv',
#                'categorical_feautures': ['Ship Mode', 'Country', 'Segment', 'Category', 'Sub-Category'],
#                'timecolumn': 'Ship Date',
#                'drop_rest': True,
#                'extreme_drop': 'Row ID',
#                'supervised': True,
#                'reg_class': (None, 'Country, France')
#                }

# plp = Pipeline(**test_params)
# processed_data, acp = plp.preprocess()
# processed_data.to_csv('processed_data.csv')
# labelled_data = plp.regression()

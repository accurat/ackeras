from auto_ml.data_cleaning import AccuratPreprocess
from auto_ml.dim_red import RedDimensionality
from auto_ml.clustering import Clustering
from auto_ml.regression import Regression

import time
import pandas as pd
import pdb


def format_list(list_like):

    series_like = pd.Series(list_like)

    series_like = series_like.apply(
        lambda x: x.lower().replace('-', '_').replace(' ', '_'))

    return list(series_like)


def format_string(string):
    return string.lower().replace('-', '_').replace(' ', '_')


class Pipeline():
    '''
    The parameters of the class are: #TODO Update
    - input_data: a pd.DataFrame with the data input
    - categorical_feautures: a list of categorical feautures
    - timecolumn: the datetime columns name
    - extreme_drop: drop this column in a worst case scenario fashion, usually it can be None
    - y: the dependent variable in supervised problems
    - drop_rest: keep it True
    - supervised: whether the problem is supervised or unsupervised

    '''

    def __init__(self, input_data,
                 categorical_feautures=None,
                 timecolumn=None,
                 extreme_drop=None,
                 y=None,
                 drop_rest=True,
                 supervised=False):

        categorical_feautures = format_list(categorical_feautures)

        input_data.columns = format_list(input_data.columns)

        self.input_data = input_data
        self.categorical_feautures = categorical_feautures if len(
            categorical_feautures) != 0 else None

        y = format_list(y[0]) if isinstance(y, list) else format_string(y)

        self.timecolumn = format_list(timecolumn[0]) if isinstance(
            timecolumn, list) else format_string(timecolumn)

        self.extreme_drop = extreme_drop
        self.reg_class = (None, y)
        self.drop_rest = drop_rest
        self.supervised = supervised
        self.acp = None
        self.outputs = None
        self.status = 'Working...'

    def preprocess(self):
        print(f'Preprocessing ...')

        self.acp = AccuratPreprocess(self.input_data)
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

    def process(self):
        try:
            data_processed, _ = self.preprocess()
            cluster_data = self.clustering()
            #coefficients = self.regression()
            self.status = 'Done'

            #TODO - all
            data_processed = data_processed.reset_index().T.to_dict() if isinstance(
                data_processed, pd.DataFrame) else pd.DataFrame(data_processed).T.to_dict()
            cluster_data = cluster_data[0].reset_index().T.to_dict() if isinstance(
                cluster_data[0], pd.DataFrame) else pd.DataFrame(cluster_data[0]).T.to_dict()
            # coefficients = coefficients.reset_index().T.to_dict() if isinstance(
            #     coefficients, pd.DataFrame) else pd.DataFrame(coefficients).T.to_dict()

            outputs = {
                'acp': data_processed,
                'cluster_data': cluster_data,
                # 'coefficients': coefficients,
            }
            self.outputs = outputs
            return self.outputs

        except Exception as e:
            self.status = f'Error happening, we are giving up {e}'

            return None

from ackeras.data_cleaning import AccuratPreprocess
from ackeras.dim_red import RedDimensionality
from ackeras.clustering import Clustering
from ackeras.regression import Regression
from ackeras.classification import Classification

import time
import pandas as pd
import pdb

from itertools import compress

pd.options.mode.chained_assignment = None


def format_list(list_like):

    series_like = pd.Series(list_like)

    series_like = series_like.apply(
        lambda x: x.lower().replace('-', '_').replace(' ', '_'))

    return list(series_like)


def format_string(string):
    return string.lower().replace('-', '_').replace(' ', '_')


class Pipeline():
    '''
    The parameters of the class are:
    - input_data: a pd.DataFrame with the data input
    - categorical_features: a list of categorical feautures
    - timecolumn: the datetime columns name
    - extreme_drop: drop this column in a worst case scenario fashion, usually it can be None
    - y: the dependent variable in supervised problems
    - drop_rest: keep it True
    - supervised: whether the problem is supervised or unsupervised

    '''

    def __init__(self, input_data,
                 categorical_features=None,
                 timecolumn=None,
                 extreme_drop=None,
                 y=None,
                 drop_rest=True,
                 supervised=False,
                 insample=None):

        categorical_features = format_list(categorical_features)
        input_data.columns = format_list(input_data.columns)

        if y is not None:
            y = format_list(y[0]) if isinstance(y, list) else format_string(y)

        for col in input_data.columns:
            if input_data[col].dtype == float:
                input_data[col] = input_data[col].apply(
                    lambda x: int(x) if not pd.isna(x) else x)

        self.insample = insample if insample is not None else int(
            len(input_data)*.9)
        self.categorical_features = categorical_features if len(
            categorical_features) != 0 else None

        self.timecolumn = format_list(timecolumn[0]) if isinstance(
            timecolumn, list) else format_string(timecolumn)
        self.input_data = input_data
        self.y = y
        self.extreme_drop = extreme_drop
        self.reg_class = (None, y)
        self.drop_rest = drop_rest
        self.supervised = supervised
        self.acp = None
        self.outputs = None
        self.status = 'Working...'

    def preprocess(self):
        print(f'Preprocessing ...')
        params = {
            'categorical_features': self.categorical_features,
            'timecolumn': self.timecolumn,
            'save': False,
            'drop_rest': self.drop_rest,
            'outputplot': False,
            'extreme_drop': self.extreme_drop,
        }

        self.acp = AccuratPreprocess(self.input_data)
        self.data_processed = self.acp.fit_transform(**params)

        if self.y is not None:
            if isinstance(self.y, list):
                self.y = self.y[0]

            if self.y not in list(self.data_processed.columns):
                columns = list(self.data_processed.columns)
                boolean = [col.startswith(str(self.y)) for col in columns]
                self.y = list(compress(columns, boolean))[0]

        return self.data_processed, self.acp

    def clustering(self):
        print('Clustering...')
        assert self.acp is not None
        self.cluster_class = Clustering(
            self.data_processed, categorical_features=self.acp.embedded_columns)
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

    def classification(self):
        print('Looking for classes, this takes loads of time...')
        assert self.acp is not None
        data = self.data_processed
        insample = self.insample

        assert isinstance(insample, int)
        X_insample, y_insample = data[:insample].drop(
            self.y, axis=1), data[:insample][self.y]
        X_outsample, y_outsample = data[insample:].drop(
            self.y, axis=1), data[insample:][self.y]

        catcols_X = self.acp.categorical_features.copy()
        catcols_X.remove(self.y)

        params = {
            'categorical_features': catcols_X,
            'analysis': True,
            'outputplot': False,
            'avoid_pca': False,
        }
        dim_red_insample = RedDimensionality(X_insample, **params)
        X_in_pca = dim_red_insample.dim_reduction()

        pca = dim_red_insample.pca_mod
        self.pca = pca
        X_out_pca = pca.fit_transform(X_outsample)

        cl = Classification(X_in_pca, y_insample,
                            X_outsample=X_out_pca, y_outsample=y_outsample)
        join_prob = cl.fit_predict()
        self.classifiers = (cl.opt_frst, cl.opt_svm)

        return join_prob

    def process(self):
        try:
            data_processed, _ = self.preprocess()
            cluster_data = self.clustering()

            coefficients, joint_prob = None, None
            if (self.y is not None):
                joint_prob = self.classification()

                columns = self.acp.label_encoders[self.y].classes_ if self.y in self.acp.label_encoders.keys(
                ) else None

                prob_df = pd.DataFrame(
                    joint_prob, index=self.data_processed[self.insample:].index, columns=columns)

            self.status = 'Done'

            data_processed = pd.DataFrame(
                data_processed).reset_index().T.to_dict()
            cluster_data = pd.DataFrame(
                cluster_data[0]).reset_index().T.to_dict()
            coefficients = pd.DataFrame(coefficients).reset_index().T.to_dict()
            prob_df = pd.DataFrame(prob_df).reset_index().T.to_dict()

            outputs = {
                'acp': data_processed,
                'cluster_data': cluster_data,
                'coefficients': coefficients,
                'probability': prob_df}
            self.outputs = outputs
            return self.outputs

        except Exception as e:
            self.status = f'An error occured, contact 118: {e}'

            return None

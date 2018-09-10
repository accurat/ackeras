import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from sklearn.preprocessing import LabelEncoder
from pandas.io.common import CParserError
from pandas.api.types import CategoricalDtype
from autokeras.preprocessor import OneHotEncoder


class AccuratPreprocess():  # TODO add outlier detection

    def __init__(self, input_data=None, path=None):  # TODO-fix receiving data
        self.path = path
        self.raw_data = input_data
        self.categorical_features = []
        self.label_encoders = {}
        self.approach = None
        self.data = None
        self.embedded_columns = None

    def get_data(self, skip_bad=False, sep=',', encoding='utf_8'):
        '''
        Only to use in case the input is not a DataFrame but a path
        '''

        assert isinstance(self.path, str)
        detect_format = self.path.split('.')[-1]

        if detect_format == 'json':
            df = pd.read_json(self.path)
            self.raw_data = df
            print('Data saved in class')

        elif detect_format == 'csv':
            try:
                df = pd.read_csv(self.path,
                                 error_bad_lines=skip_bad,
                                 sep=sep)
                self.raw_data = df
                print('Data saved in class')

            except FileNotFoundError as e:
                print(e)
            except CParserError:
                print(
                    'Error parsing the csv. Try the correct encoding e.g. get_data(encodeing="ascii")')
        else:
            raise ImportError(
                'Path did not lead to a understandable file type, did not import')

    def deal_na(self, deal='normal', thresh=.5, deal_cat=True):
        assert isinstance(self.raw_data, pd.DataFrame)
        raw_data = self.raw_data

        raw_data = raw_data.dropna(
            axis=1, thresh=int(thresh*raw_data.shape[0]))

        def normal(df):
            nan_cols = df.loc[:, df.isna().any()].columns
            for col in nan_cols:
                series = df[col]
                try:
                    mu = np.mean(series)
                    std = np.std(series)
                    series = series.fillna(np.random.normal(mu, std))
                except (ValueError, TypeError):
                    print(
                        f'Skipping column {col} because not numerical, using ffill instead, deactivate with "deal_cat = False"')
                    if deal_cat:
                        series = series.fillna(method='ffill')
                df[col] = series

            return df

        def median(df):
            nan_cols = df.loc[:, df.isna().any()].columns
            for col in nan_cols:
                series = df[col]
                try:
                    median = np.median(series)
                    series = series.fillna(median)
                except (ValueError, TypeError):
                    print(
                        f'Skipping column {col} because not numerical, using ffill instead, deactivate with "deal_cat = False"')
                    if deal_cat:
                        series = series.fillna(method='ffill')

                df[col] = series
            return df

        def nfill(df):
            nan_cols = df.loc[:, df.isna().any()].columns
            for col in nan_cols:
                series = df[col]
                series = series.fillna(method='ffill').fillna(method='bfill')
                df[col] = series
            return df

        func = {
            'normal': normal,
            'median': median,
            'nfill': nfill,
        }

        assert deal in func.keys()

        filled_data = func[deal](raw_data)
        self.raw_data = filled_data
        print('Got rid of the NaN')
        return filled_data

    def data_encoding(self, categorical_features, label_encoder=False):
        assert isinstance(categorical_features, list)
        assert isinstance(self.raw_data, pd.DataFrame)
        raw_data = self.raw_data

        try:
            cat_pos = [int(a) for a in categorical_features]
            cat_columns = raw_data.columns[cat_pos]
        except ValueError:
            cat_columns = categorical_features

        cat_data = raw_data[cat_columns]

        for col in cat_columns:
            col_data = cat_data[col]
            if len(list(col_data.unique())) > 2:
                print(f'Using label encoder for {col}')
                lb = LabelEncoder()
                label_data = lb.fit_transform(list(col_data))
                self.label_encoders[str(col)] = lb

                cat_data[col] = label_data

            else:
                print(f'Using one hot encoder for {col}')
                one_hot = OneHotEncoder()
                one_hot.fit(col_data)
                one_hot_data = one_hot.transform(col_data)
                self.label_encoders[str(col)] = one_hot

                cat_data = cat_data.drop(col, axis=1)
                renamed_columns = pd.Series(list(one_hot.labels)).apply(
                    lambda x: ', '.join([str(col), str(x)]))
                one_hot_df = pd.DataFrame(
                    one_hot_data, columns=renamed_columns)
                cat_data = pd.concat([cat_data, one_hot_df], axis=1)
                cat_data = cat_data.drop(
                    sorted(list(renamed_columns))[-1], axis=1)

        raw_data = raw_data.drop(cat_columns, axis=1)
        raw_data = pd.concat([raw_data, cat_data], axis=1)
        raw_data[cat_data.columns] = raw_data[cat_data.columns].astype(
            CategoricalDtype())
        self.embedded_columns = cat_data.columns
        self.raw_data = raw_data

        return raw_data

    def datetime_index(self, timecolumn, set_index=False):
        if isinstance(timecolumn, list):
            timecolumn = str(timecolumn[0])

        raw_data = self.raw_data
        sample = raw_data[timecolumn].sample(1).values[0]
        if (isinstance(sample, int) or isinstance(sample, float)):
            print('Treating number as UNIX time')

        try:
            raw_data[timecolumn] = raw_data[timecolumn].apply(
                lambda x: pd.Timestamp(x))
            if set_index:
                raw_data = raw_data.set_index(timecolumn).sort_index()

            self.raw_data = raw_data

        except ValueError:
            print('Not timestamp, nothing changed')

    def fit_transform(self, categorical_features=None, timecolumn=None, save=False, drop_rest=True, outputplot=False, extreme_drop=None):
        '''
        One should just run this function, after calling AccuratPreprocess(data), with:
        categorical_features: the names of the categorical feautures
        timecolumn: the name of the timestamp column
        save: ...
        drop_rest: whether to drop all the variables that have not be encoded (necessary for clustering and supervised learning)
        outputplot: slow plotting of some correlation_matrix, if you are a strong indipendent person you don't need graphs, don't do it!
        '''
        self.categorical_features = categorical_features

        if self.path:
            self.get_data()

        self.deal_na()
        if categorical_features:
            self.data_encoding(categorical_features)
        if timecolumn:
            self.datetime_index(timecolumn, set_index=True)
        self.data = self.raw_data
        if drop_rest:
            self.data = self.data.drop(
                self.data.select_dtypes('object').columns, axis=1)
        if len(extreme_drop) != 0:
            self.data = self.data.drop(extreme_drop, axis=1)
        now = datetime.now()

        filename = f'data_at_{now.hour}_{now.day}_{now.month}.csv'
        if save:
            self.data.to_csv(filename)

        if outputplot:
            print('Plotting now, it could take time, turn "outputplot=False"')
            data_plot = self.data.select_dtypes(exclude='int')
            hue = outputplot[1] if isinstance(outputplot, tuple) else None
            plt.figure(figsize=(15, 10))
            sns.pairplot(data_plot, hue=hue)
            figname = filename.replace('csv', 'png')
            plt.savefig(figname, dpi=400)
            print(f'Saved plot at {figname}')

        return self.data

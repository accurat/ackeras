import pandas as pd
import numpy as np
import time

from datetime import datetime
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.utils import compute_class_weight
from pandas.api.types import CategoricalDtype


class Regression():
    def __init__(self, data, y, problem=None):
        assert isinstance(data, pd.DataFrame)
        assert isinstance(y, str)
        assert problem in ['classification', 'regression', None]

        self.data = data
        self.X_series = data.drop(y, axis=1)
        self.y_series = data[y]
        self.seed = int(time.mktime(datetime.now().timetuple()))

        if problem is None:
            if isinstance(data[y].dtypes, CategoricalDtype):
                self.problem = 'classification'
            else:
                self.problem = 'regression'
            print(f'The problem was set to: {self.problem}')
        else:
            self.problem = problem

        self.opt_regres = None

    def ridge_regression(self, opt_space=np.arange(0.1, 10., 0.5)):
        feautures, target = self.X_series.values, self.y_series.values
        regres = RidgeCV()
        regres.fit(feautures, target)
        self.opt_regres = regres
        self.parameters = regres.coef_

        self.fitted = regres.predict(feautures)
        print(f'Optimum score {regres.scoring}')

        return regres.coef_

    def logistic_regression(self, opt_space=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        feautures, target = self.X_series, self.y_series

        parameters = {
            'C': opt_space,
            'penalty': ('l1', 'l1'),
        }
        ls = LogisticRegression()
        opt_log = GridSearchCV(ls, parameters, verbose=1, cv=5)
        opt_log.fit(feautures, target)
        print(f'Score on test {opt_log.best_score_}')

        self.opt_regres = LogisticRegression(**opt_log.best_params_)
        self.parameters = opt_log.best_params_

        labels = opt_log.predict_proba(feautures)
        self.predict = labels
        new_df = self.data.copy()
        new_df['pred_labels'] = labels[:, 1]
        self.data = new_df

        return self.data

    def fit_predict(self):
        returning = None
        if self.problem == 'classification':
            returning = self.logistic_regression()
        elif self.problem == 'regression':
            returning = self.ridge_regression()

        else:
            raise TypeError(
                '--FLAG-- : Problem not understood, no regression done. Rerun with "problem" not None')

        return self.opt_regres, returning

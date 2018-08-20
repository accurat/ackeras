import pandas as pd
import numpy as np
import time
import pdb

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV


def average_prediction(pred_tree, pred_svm):
    return np.average(np.vstack((pred_tree, pred_svm)), axis=0)


class Classification():
    def __init__(self, X_insample, y_insample, X_outsample=None, y_outsample=None, verbose=100):
        self.X_insample = X_insample
        self.X_outsample = X_outsample
        self.y_insample = y_insample
        self.y_outsample = y_outsample
        self.seed = int(time.mktime(datetime.now().timetuple()))
        self.verbose = verbose
        self.svm_called = False
        self.frst_called = False
        self.opt_svm = SVC(probability=True)
        self.opt_frst = RandomForestClassifier()

        self.svm_space = {
            'kernel': ['rbf'],
            'gamma': np.logspace(-9, 9, num=25, base=10),
            'C':  np.logspace(-9, 9, num=25, base=10)
        }

        self.frst_space = {
            'n_estimators': list(np.arange(10, 30, 2)),
            'max_features': list(np.round(np.arange(0.1, 1., .05), 2)),
            'criterion': ['gini', 'entropy'],
        }

        self.default_evparams = {'scoring': "accuracy",
                                 'cv': StratifiedKFold(n_splits=4),
                                 'verbose': self.verbose,
                                 'population_size': 20,
                                 'gene_mutation_prob': 0.10,
                                 'gene_crossover_prob': 0.5,
                                 'tournament_size': 3,
                                 'generations_number': 2,
                                 'n_jobs': 6
                                 }

    def ev_svm(self):
        ev_params = self.default_evparams
        ev_params['estimator'] = SVC(probability=True)
        ev_params['params'] = self.svm_space

        cv = EvolutionaryAlgorithmSearchCV(**ev_params)
        cv.fit(self.X_insample, self.y_insample)

        clf = cv.best_estimator_

        self.opt_svm = clf

    def ev_tree(self):
        ev_params = self.default_evparams
        ev_params['estimator'] = RandomForestClassifier()
        ev_params['params'] = self.frst_space

        cv = EvolutionaryAlgorithmSearchCV(**ev_params)
        cv.fit(self.X_insample, self.y_insample)

        clf = cv.best_estimator_

        self.opt_frst = clf

    def ensable_prediction(self):
        X_insample, X_outsample = self.X_insample, self. X_outsample
        y_insample, y_outsample = self.y_insample, self.y_outsample
        svm_clf = self.opt_svm
        frst_clf = self.opt_frst

        if not self.svm_called:
            print('SVM not optimized, using default')
            svm_clf.fit(X_insample, y_insample)
        if not self.frst_called:
            print('Random Forest not opimized, using default')
            frst_clf.fit(X_insample, y_insample)

        joint_prob = None
        if X_outsample and y_outsample:
            svm_pred = svm_clf.predict_proba(y_outsample)
            frst_pred = frst_clf.predict_proba(y_outsample)

            joint_prob = average_prediction(frst_pred, svm_pred)

        self.joint_prob = joint_prob

        return joint_prob

    def fit_predict(self):
        self.ev_svm()
        self.ev_tree()

        joint_prob = self.ensable_prediction()

        if joint_prob:
            print(
                'Outputting joint_probability, you can get the classifiers with .opt_svm and .opt_frst')
            return joint_prob

        else:
            print(
                'Careful, no out of sample data, you can still get the classifiers with .opt_svm and .opt_frst')
            return None


# ------------- Test data

data = pd.read_csv(
    '/Users/andreatitton/ackeras/places_processed_data.csv').set_index('order_date').sort_index()

indexing = int(len(data)*.9)
data_params = {
    'X_insample': data[: indexing].drop('country, France', axis=1),
    'y_insample': data[: indexing]['country, France'],
    'X_outsample': data[indexing:].drop('country, France', axis=1),
    'y_outsample': data[indexing:]['country, France'],
}

cl = Classification(**data_params)
pred_data = cl.fit_predict()

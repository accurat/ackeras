import pandas as pd
import numpy as np
import time
import pdb

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from evolutionary_search import EvolutionaryAlgorithmSearchCV


def average_prediction(pred_tree, pred_svm):
    return np.mean([pred_tree, pred_svm], axis=0)


class Classification():
    '''
    The function takes feautures (X_insample) and targets (y_insample) and the (optional) corrisponding out of sample data and fits a support vector machine and a random forest.
    One could just call the "fit_predict" that returns the joint porbability and then access the models in self.opt_svm and self.opt_frst.
    '''

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
                                 'population_size': 10,
                                 'gene_mutation_prob': 0.10,
                                 'gene_crossover_prob': 0.5,
                                 'tournament_size': 3,
                                 'generations_number': 2,
                                 'n_jobs': 10
                                 }

    def ev_svm(self):
        ev_params = self.default_evparams
        ev_params['estimator'] = SVC(probability=True)
        ev_params['params'] = self.svm_space

        cv = EvolutionaryAlgorithmSearchCV(**ev_params)
        cv.fit(self.X_insample, self.y_insample)

        clf = cv.best_estimator_

        self.svm_called = True
        self.opt_svm = clf

    def ev_tree(self):
        ev_params = self.default_evparams
        ev_params['estimator'] = RandomForestClassifier()
        ev_params['params'] = self.frst_space

        cv = EvolutionaryAlgorithmSearchCV(**ev_params)
        cv.fit(self.X_insample, self.y_insample)

        clf = cv.best_estimator_

        self.frst_called = True
        self.opt_frst = clf

    def ensable_prediction(self):

        X_insample, X_outsample = self.X_insample, self. X_outsample
        y_insample, y_outsample = self.y_insample.values, self.y_outsample.values

        svm_clf = self.opt_svm
        frst_clf = self.opt_frst

        if (svm_clf is not None) and (not self.svm_called):
            print('SVM not optimized, using default')
            svm_clf.fit(X_insample, y_insample)

        if not self.frst_called:
            print('Random Forest not opimized, using default')
            frst_clf.fit(X_insample, y_insample)

        joint_prob = None
        if X_outsample is not None:
            frst_pred = frst_clf.predict_proba(X_outsample)
            svm_pred = svm_clf.predict_proba(
                X_outsample) if svm_clf is not None else frst_pred

            joint_prob = average_prediction(frst_pred, svm_pred)

        if y_outsample is not None:
            score = confusion_matrix(
                self.y_outsample, np.argmax(joint_prob, axis=1))
            print(f'The confusion matrix is\n{score}')

        self.joint_prob = joint_prob

        return joint_prob

    def fit_predict(self):

        if self.X_insample.shape[0] < 20000:
            self.ev_svm()
        else:
            print('The dataset is too big for SVM, using only random forest')
            self.opt_svm = None

        self.ev_tree()

        joint_prob = self.ensable_prediction()

        if joint_prob is not None:
            print(
                'Outputting joint_probability, you can get the classifiers with .opt_svm and .opt_frst')
            return joint_prob

        else:
            print(
                'Careful, no out of sample data, you can still get the classifiers with .opt_svm and .opt_frst')
            return None

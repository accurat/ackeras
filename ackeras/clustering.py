import sklearn
import pandas as pd
import numpy as np
import random
import pdb

from datetime import datetime
import time
from sklearn.cluster import KMeans, DBSCAN
from kmodes.kprototypes import KPrototypes
from pandas.api.types import CategoricalDtype

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class Clustering():
    def __init__(self, data, categorical_features=None, pre_k=None):
        """
        The class requires a pd.DataFrame and normalized data to work. 
        It simply fits the data to be clustered.
        """
        self.data = data
        self.categorical_features = categorical_features if np.issubdtype(
            data[categorical_features].dtypes, np.int) else [data.columns.get_loc(c) for c in data.columns if c in categorical_features]
        self.is_cat = False
        self.check_df()
        self.pre_k = pre_k
        self.seed = int(time.mktime(datetime.now().timetuple()))
        self.data_clustered = None
        self.pca = None

    def check_df(self):

        check_data = self.data
        normalize_data = check_data.select_dtypes(float)
        range_data = list(normalize_data.max() - normalize_data.min())
        for n, i in enumerate(range_data):
            if np.abs(i > 3):
                print(
                    f'-- Flag --: the column {normalize_data.columns[n]} does not seem to be normalized')

        if self.categorical_features is not None:
            self.is_cat = True

        elif (isinstance(check_data, pd.DataFrame) and (self.categorical_features is None)):
            cat_data = check_data.select_dtypes(CategoricalDtype())
            if cat_data.shape[1] == 1:
                self.is_cat = True

        else:
            print(
                'Did not found categorical variables, specify at "categorical_features"')

    def dbscan(self):
        cluster_data = self.data
        params = {
            'eps': 0.5,
            'min_samples': 5,
            'metric': None
        }
        params['metric'] = 'euclidean' if cluster_data.shape[1] < 10 else 'cosine'
        dbscan = DBSCAN(**params)
        pred_label = pd.Series(dbscan.fit_predict(cluster_data)).apply(
            lambda x: None if x == -1 else x)

        return pred_label

    def silouhette_analysis(self, cluster_data, pca=False, prototype=False, end_range=20):
        range_n_cluster = list(range(3, end_range, 1))
        sil_avg = []
        for n_cluster in range_n_cluster:
            print(f'Trying cluster {n_cluster}')
            clusterer = KMeans(n_clusters=n_cluster,
                               random_state=self.seed)
            train, test = train_test_split(
                cluster_data, test_size=0.2, random_state=self.seed)
            if pca:
                pca_trans = PCA(n_components=0.9)
                train_pca = pca_trans.fit_transform(train)
                self.pca_trans = pca_trans
                if train_pca.shape[1] < 2:
                    pca_trans = PCA(n_components=2)
                    train_pca = pca_trans.fit_transform(train)
                    self.pca_trans = pca_trans
            train = train_pca if self.pca_trans is not None else train
            cluster_labels = clusterer.fit(train)

            test = self.pca_trans.transform(
                test) if self.pca_trans is not None else test
            cluster_labels = clusterer.predict(test)
            score = silhouette_score(test, cluster_labels)
            print(f'Got score: {score}')
            sil_avg.append(score)

        index = np.argmax(sil_avg)
        opt_k = range_n_cluster[index]
        print(
            f'The best cluster has silhoutte score of {np.max(sil_avg)} k={opt_k}')

        return opt_k

    def kmean(self, pca=False):
        cluster_data = self.data
        new_df = cluster_data.copy()

        opt_k = self.silouhette_analysis(cluster_data, pca=pca)

        km = KMeans(n_clusters=opt_k, random_state=self.seed)
        labels = km.fit_predict(cluster_data)

        new_df['labels'] = labels
        self.data_clustered = new_df

        return new_df

    def kproto(self):  # TODO- solve clustering issue with PCA + K-means
        cluster_data = self.data
        opt_k = self.silouhette_analysis(cluster_data, prototype=True)

        kp = KPrototypes(n_clusters=opt_k)
        kp.fit(cluster_data, categorical=self.categorical_features)
        labels = kp.predict(
            cluster_data, categorical=self.categorical_features)

        cluster_data['labels'] = labels
        self.data_clustered = cluster_data

        return cluster_data

    def fit_predict(self):

        if self.is_cat:
            # print('Using k-prototype because the data is mixed categorical data')
            # clustered_data = self.kproto()
            print('Using KMeans with PCA')
            clustered_data = self.kmean(pca=True)

        else:
            print('Using DBSCAN and Kmeans!')
            clustered_data = self.kmean()
            clustered_data['dbscan'] = self.dbscan()

        self.clustered_data = clustered_data

        return clustered_data

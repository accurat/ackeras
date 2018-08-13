import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

import umap
from sklearn.decomposition import PCA
from auto_ml.data_cleaning import AccuratPreprocess
from sklearn.preprocessing import Normalizer


class RedDimensionality():
    def __init__(self, data, categorical_feautures=None, analysis=False, outputplot=False, avoid_pca=True):
        assert (isinstance(data, pd.DataFrame) or isinstance(data, np.array))
        self.data = data
        self.param = {
            'metric': 'cosine',
            'n_neighbors': int(data.shape[0] * .1),
            'n_components': 2
        }
        self.pca_data = data.drop(categorical_feautures, axis=1)
        self.cat_data = data[categorical_feautures]
        self.outputplot = outputplot
        self.analysis = analysis
        self.n_components = 2
        self.avoid_pca = avoid_pca
        try:
            self.index = data.drop(categorical_feautures, axis=1).index
            self.columns = data.drop(categorical_feautures, axis=1).columns
        except AttributeError:
            self.index, self.columns = None, None

    def umap(self):
        plt_data = self.pca_data.select_dtypes(exclude='object')
        reducer = umap.UMAP(**self.param)
        embedding = reducer.fit_transform(plt_data)

        if self.outputplot:
            print('Plotting figure as: embedded_figure_umap.png')
            plt.figure(figsize=(15, 10))
            emb_df = pd.DataFrame(embedding, columns=[
                                  'First component', 'Second component'])
            sns.scatterplot(data=emb_df)
            plt.savefig('embedded_figure_umap.png', dpi=400)

        return embedding

    def pca(self):
        plt_data = self.pca_data.select_dtypes(exclude='object')
        pca = PCA(n_components=0.9)
        embedding = pca.fit_transform(plt_data)

        return embedding

    def normalization(self):
        plt_data = self.pca_data.select_dtypes(exclude='object')
        normalizer = Normalizer(norm='l2')
        embedding = normalizer.fit_transform(plt_data)

        return embedding

    def dim_reduction(self):
        if (self.analysis and self.avoid_pca):
            print('Normalizing...')
            embedding = self.normalization()

        elif (self.analysis):
            print('Doing PCA...')
            embedding = self.pca()
            if embedding.shape[1] < 4:
                print('PCA gave to few feautures, normalizing...')
                embedding = self.normalization()
        else:
            print('Doing UMAP...')
            embedding = self.umap()

        if self.index is not None:
            embedding = pd.DataFrame(
                embedding, index=self.index, columns=self.columns)
            embedding = pd.concat([self.cat_data, embedding], axis=1)

        print('...done!')
        return embedding

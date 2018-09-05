import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('darkgrid')

import time

from datetime import datetime


class Plot:
    def __init__(self, data, embedded_data=None, labels=None, x=None, y=None):
        assert isinstance(data, pd.DataFrame)
        if embedded_data is not None:
            assert isinstance(embedded_data, pd.DataFrame)

        self.data = data
        self.embedded_data = embedded_data
        self.labels = labels
        self.seed = int(time.mktime(datetime.now().timetuple()))
        self.x = x
        self.y = y

    def pairplot(self):
        plt_data = self.data if self.embedded_data is None else self.embedded_data
        path = f'images/pairplot_{self.seed}.png'
        plt.figure(figsize=(20, 20))
        sns.pairplot(plt_data, hue=self.y)
        plt.savefig(path, dpi=100)
        plt.close()

    def plot_classification(self):
        plt_data = self.data if self.embedded_data is None else self.embedded_data
        path = f'images/classification_{self.seed}.png'
        labels = self.labels
        assert labels is not None

        if self.embedded_data is not None:
            plt_np = plt_data.values
            x = plt_np[:, 0]
            y = plt_np[:, 1]

        else:
            x = plt_data[self.x] if self.x is not None else plt_data[np.random.choice(
                list(plt_data.columns))]
            y = plt_data[self.y] if self.y is not None else plt_data[np.random.choice(
                list(plt_data.columns))]

        plt.figure(figsize=(20, 20))
        plt.scatter(x, y, c=labels)
        plt.savefig(path, dpi=100)
        plt.close()

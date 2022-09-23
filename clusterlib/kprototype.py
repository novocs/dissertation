import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from .analysis.statmethods import SummaryStats
from kneed import KneeLocator


class KPrototypesInterface(SummaryStats):
    def __init__(
            self,
            input_data=pd.DataFrame(),
            categorical_cols=list(),
    ):
        """
        Helper methods for implementing the k-prototypes algorithm from the kmodes python package. It includes
        Args:
            input_data (pandas.DataFrame): the dataset to be clustered
            categorical_cols (list): list of categorical feature names
        """
        self.input_df = input_data
        self.cat_cols = categorical_cols
        self.cat_col_idxs = [input_data.columns.get_loc(col) for col in categorical_cols]

        self.model_dict = dict()
        self.cost_dict = dict()

    def clustering(self, n_clusters, init='Cao', verbose=0, n_jobs=-1, n_init=10):
        """
        Performs k-prototype clustering for a given number of clusters
        Args:
            n_clusters (int): the number of clusters
            init (str): k-modes defined initialisation method
            verbose (int): level of detail to print in the console
            n_jobs (int): number of workers
            n_init (int): the number of times to initialize and run the algorithm before selecting best results
        Returns:
             model class from the k-modes python package
        """
        kproto = KPrototypes(n_clusters=n_clusters, init=init, verbose=verbose, n_jobs=n_jobs, n_init=n_init)
        kproto.fit(self.input_df, categorical=self.cat_col_idxs)
        return kproto

    def elbow_method(
            self, n_cluster_list, init='Cao', verbose=0, n_jobs=-1, n_init=10, overwrite=False, elbow_locator=True
    ):
        """
        Supports the elbow method
        Args:
            n_cluster_list: the list of clusters to run the k-prototype algorithm for
            init (str): k-modes defined initialisation method
            verbose (int): level of detail to print in the console
            n_jobs (int): number of workers
            n_init (int): the number of times to initialize and run the algorithm before selecting best results
            overwrite (bool): whether to overwrite oreviously obtained costs and cost models
            elbow_locator (bool): whether to include an elbow locator in the plot
        """
        for n_cluster in n_cluster_list:
            if n_cluster in self.model_dict and not overwrite:
                continue

            model = self.clustering(n_cluster, init=init, verbose=verbose, n_jobs=n_jobs, n_init=n_init)
            self.cost_dict[n_cluster] = model.cost_
            self.model_dict[n_cluster] = model

        font_size = 12
        plt.rcParams['font.size'] = str(font_size * 1.5)
        plt.figure(figsize=(8, 8))
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Cost')
        plt.title('Elbow method')
        plt.plot(self.cost_dict.keys(), self.cost_dict.values(), marker='x', label='data', linewidth=4, markersize=15)
        if elbow_locator:
            el = KneeLocator(
                list(self.cost_dict.keys()), list(self.cost_dict.values()), direction='decreasing', curve='convex'
            )
            plt.vlines(el.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', label='elbow', color='r', linewidth=4)
        plt.legend()

    def kprototype_cluster_summary(self, n_clusters, data, sort_col=None):
        """
        Summarizes the cluster results of a Kprototype model by aggregating means of numerical features and
        identifying modes of categorical features for each cluster
        Args:
            n_clusters: the number of components in the kprototype model
            data (pandas.DataFrame): the dataset that was clustered
            sort_col (str): a feature to sort the output results by
        Returns:
            (pandas.DataFrame): dataframe containing aggregated results
        """
        model = self.model_dict[n_clusters]
        return self.cluster_summary(model.labels_, data, self.cat_cols, sort_col=sort_col)

    def describe_approach_kprototype(self, n_clusters, data, feature, inherent_feature=None):
        """
        Retrieves information to analyse Kprototype clusters generated from an autoencoded embedding approach
        Args:
            n_clusters: an instance of a kprototype model
            data (pandas.DataFrame): the dataset that was clustered
            feature (str): the feature to be used in comparing cluster similarities using descriptive and inferential
                statistics
            inherent_feature: the feature to be used in evaluating the ability of a clustering algorithm to separate
                a particular feature in its different consituant clustes
        Returns:
            (dict): key contains descriptions and values show the results
        """
        results = dict()
        model = self.model_dict[n_clusters]

        df = pd.DataFrame()
        df[feature] = data[feature].tolist()
        df['cluster_assignment'] = model.labels_

        results['feature'] = feature
        results['param_clusters'] = model.n_clusters
        results['param_num_init'] = model.n_init

        return self._describe_approach(
            results, df, data, feature, inherent_feature=inherent_feature
        )

import pandas as pd
import scipy.stats as st


class SummaryStats:

    @staticmethod
    def cluster_summary(cluster_labels, data, cat_cols, sort_col=None):
        """
        Function to support aggregating means of numerical features and modes of categorical features by cluster
        Args:
            cluster_labels: an array containig cluster assignments for each instance in data
            data (pandas.DataFrame): the dataset that was clustered
            cat_cols (list): categorical columns in the dataset that was clustered
            sort_col (str): a feature to sort the output results by
        Returns:
            (pandas.DataFrame): dataframe containing aggregated results
        """

        # defining variables
        df = data.copy()
        instance_count = 'instance_count'
        cluss_ass = 'cluster_assignments'
        df[cluss_ass] = cluster_labels

        # build aggregation dictionary
        num_cols = [col for col in df.columns if col not in cat_cols]
        agg_map = dict()
        for col in cat_cols:
            agg_map[col] = pd.Series.mode
        for col in num_cols:
            agg_map[col] = 'mean'
        agg_map[cluss_ass] = 'count'
        num_cols.remove(cluss_ass)

        # aggregate metrics by cluster
        grouped_values = df.groupby(cluss_ass).agg(agg_map).rename(columns={cluss_ass: instance_count})
        pd.set_option('display.max_rows', max(150, len(grouped_values)))

        # organising displayed table
        if sort_col is not None:
            column_order = [sort_col, instance_count] + num_cols + cat_cols
            column_order.pop(column_order.index(sort_col, column_order.index(sort_col) + 1))
            return grouped_values[column_order].sort_values(sort_col)

        column_order = [instance_count] + num_cols + cat_cols
        return grouped_values[column_order]

    def _describe_approach(self, results, df, data, feature, inherent_feature=None):
        """
        Function to support retrieval of cluster information for HDBSCAN and GMM
        Args:
            results (dict): dictionary containing cluster specific parameters
            data (pandas.DataFrame): the dataset that was clustered
            feature (str): the feature to be used in comparing cluster similarities using descriptive and inferential
                statistucs
            inherent_feature: the feature to be used in evaluating the ability of a clustering algorithm to separate
                a particular feature in its different consituant clustes
        Returns:
            (dict): key contains descriptions and values show the results
        """

        results.update(self.get_minimum_cluster_size(df))
        results.update(self.get_mean_cluster_feature_range(df, feature))
        results.update(self.get_potentially_mergeable_clusters(df, feature))

        if inherent_feature is not None:
            df[inherent_feature] = data[inherent_feature].tolist()
            results.update(self.count_multicluster_attributes(df, inherent_feature))
        return results

    @staticmethod
    def count_multicluster_attributes(df, inherent_feature):
        """
        Calculates the proportion of feature categories that appear in multiple clusters
            df (pandas.DataFrame): dataframe containing feature and cluster assignments
            feature (str): the feature under consideration
        Returns:
            (dict): key is description and the value is the proportion of multicluster feature categories
        """
        feature_agg = df.groupby(inherent_feature).agg({
            'cluster_assignment': [pd.Series.nunique],
        })

        cma_results = dict()

        cma_results[f'multicluster_{inherent_feature}_count'] = len(
            feature_agg[feature_agg[('cluster_assignment', 'nunique')] > 1]
        )
        cma_results[f'{inherent_feature}_unique_attributes'] = len(feature_agg)
        cma_results[f'multicluster_{inherent_feature}_percent'] = 100 * (
                cma_results[f'multicluster_{inherent_feature}_count'] / len(feature_agg)
        )

        return cma_results

    @staticmethod
    def get_minimum_cluster_size(df):
        """
        Identifies the size of the cluster with the fewest number of instances
            df (pandas.DataFrame): dataframe cluster assignments for each instance
        Returns:
            (dict): key is description and value is minimum cluster size
        """
        cluster_agg = df.groupby('cluster_assignment').agg({
            'cluster_assignment': 'count',
        })

        gmcs_results = dict()
        gmcs_results['min_cluster_size'] = cluster_agg['cluster_assignment'].min()

        return gmcs_results

    @staticmethod
    def get_mean_cluster_feature_range(df, feature):
        """
        Calculates the mean of a provided feature in all clusters and identifies the minimum and maximum mean values
            df (pandas.DataFrame): dataframe containing feature and cluster assignments
            feature (str): the feature to be used in calculating means
        Returns:
            (dict): key is description and value is minimum or maximum mean of the provided feature
        """
        cluster_agg = df.groupby('cluster_assignment').agg({
            feature: 'mean',
        })

        gmcfr_results = dict()
        gmcfr_results[f'min_mean_cluster_{feature}'] = cluster_agg[feature].min()
        gmcfr_results[f'max_mean_cluster_{feature}'] = cluster_agg[feature].max()
        return gmcfr_results

    @staticmethod
    def cluster_means_tester(df, test_feature, sig_level=0.05):
        """
        Generates a dataframe of hypothesis tests results with each instance indicating whether a pair of clusters
        are similar based on the mean and variance of the test_feature
            df (pandas.DataFrame): dataframe containing feature and cluster assignments
            test_feature (str): the feature to be used in hypothesis tests to identify potentially mergeable clusters
        Returns:
            hypothesis_df (pandas.DataFrame): dataframe of hypothesis test results
        """
        cluss_ass_col = 'cluster_assignment'
        hypothesis_df = pd.DataFrame()
        cluster_assignments = sorted(df[cluss_ass_col].unique())
        num_assignments = len(cluster_assignments)

        # applying the bonferroni correction
        sig_level = sig_level / len(cluster_assignments)

        for index_i, assignment_i in enumerate(cluster_assignments):

            sample_i = df[df[cluss_ass_col] == assignment_i][test_feature].tolist()

            for index_j in range(index_i + 1, num_assignments):

                assignment_j = cluster_assignments[index_j]
                sample_j = df[df[cluss_ass_col] == assignment_j][test_feature].tolist()

                row_name = f'cluster_{assignment_i}__cluster{assignment_j}'
                hypothesis_df.loc[row_name, 'cluster1'] = assignment_i
                hypothesis_df.loc[row_name, 'cluster2'] = assignment_j

                # test for difference in sample variance
                p_value_levene = st.levene(sample_i, sample_j)[1]
                hypothesis_df.loc[row_name, 'var_p_value'] = p_value_levene

                if p_value_levene < sig_level:
                    hypothesis_df.loc[row_name, 'variance_result'] = 'Significant'
                    equal_variance = False
                else:
                    hypothesis_df.loc[row_name, 'variance_result'] = 'Not Significant'
                    equal_variance = True

                # test for difference in sample mean
                p_value_t = st.ttest_ind(sample_i, sample_j, equal_var=equal_variance)[1]
                hypothesis_df.loc[row_name, 'mean_p_value'] = p_value_t
                if p_value_t < sig_level:
                    hypothesis_df.loc[row_name, 'mean_result'] = 'Significant'
                else:
                    hypothesis_df.loc[row_name, 'mean_result'] = 'Not Significant'

        return hypothesis_df

    def get_potentially_mergeable_clusters(self, df, feature):
        """
        Calculates the proportion of cluster combinations that are potentially mergeable based on inferential statistics
            df (pandas.DataFrame): dataframe containing feature and cluster assignments
            feature (str): the feature to be used in hypothesis tests to identify potentially mergeable clusters
        Returns:
            (dict): key is description and value is the proportion of potentially mergable clusters
        """
        hp = self.cluster_means_tester(df, feature)
        mask = (hp['variance_result'] == 'Not Significant') & (hp['mean_result'] == 'Not Significant')
        potential_merges = 100 * len(hp[mask]) / len(hp)

        return {f'percent_potentially_mergeable': potential_merges}
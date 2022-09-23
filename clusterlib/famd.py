import numpy as np
import prince
import time
import matplotlib.pyplot as plt


class FAMDRecat:
    def __init__(
            self,
            input_data,
            categorical_columns=list(),
    ):
        """
        Class for reccategorising input data and generating FAMD principal components
        Args:
            input_data (pandas.DataFrame): the input data
            categorical_columns (lsit): the column names of categorical features
        """

        self.input_data = input_data
        self.cat_cols = categorical_columns
        self.num_cols = [col for col in input_data.columns if col not in categorical_columns]

        # placeholders
        self.agg_df = None  # stores recatagorized features
        self.famd_pc = None  # stores FAMD principal components
        self.famd = None  # stores the fitted FAMD prince class

    def aggregation_function(self, threshold=0.1, unique_attr_threshold=50):
        """
        Categorizes categorical features into groups based on their frequency of occurrence. The recategorized
        features are stored in a class attribute agg.df
        Args:
            threshold: if the frequency of a category is less than that of the threshold multiplied by the mode
                category, then the feature is recategorized
            unique_attr_threshold: if the feature under consideration has more categories than the unique attribute
                threshold then it is recategorized
        """
        agg_df = self.input_data.copy()

        for col in self.cat_cols:
            num_categories = agg_df[col].nunique()

            if num_categories <= unique_attr_threshold:
                continue

            attribute_freq_df = agg_df[col].value_counts()
            attribute_freqs = attribute_freq_df.tolist()

            category_mode = attribute_freqs[0]

            data_to_merge = attribute_freq_df[attribute_freq_df < threshold * category_mode]

            categories_to_merge = data_to_merge.index.tolist()[::-1]
            freqs_to_merge = data_to_merge.tolist()[::-1]

            old_new_category_map = dict()
            counter = 1
            freq_sum = 0

            for index, category in enumerate(categories_to_merge):
                freq_sum += freqs_to_merge[index]

                if freq_sum > category_mode:
                    freq_sum = freqs_to_merge[index]
                    counter += 1
                    old_new_category_map[category] = f'other {counter}'

                old_new_category_map[category] = f'other {counter}'

            agg_df[col] = agg_df[col].replace(old_new_category_map)

        self.agg_df = agg_df

    def get_principal_components(
            self, n_components, n_iter=3, copy=True, check_input=True,  random_state=42
    ):
        """
        Generates FAMD principal components. The generated components are stored in a class attribute famd_pc
        Args:
            n_components: number of principal components to generate
            n_iter: the number of iterations used in computing the singular value decomposition
            copy: generate new object for results
            check_input: if True, checks the input consistency
            random_state: number to ensure reproducibility of results

        """

        famd_input = self.input_data.copy()

        if self.agg_df is not None:
            famd_input = self.agg_df.copy()

        for col in self.cat_cols:
            famd_input[col] = famd_input[col].astype('object')

        start_time = time.time()

        self.famd = prince.FAMD(
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            check_input=check_input,
            random_state=random_state
        )

        self.famd_pc = self.famd.fit_transform(famd_input[self.num_cols + self.cat_cols])
        end_time = time.time() - start_time
        print(f'The end time is {end_time}')

    def plot_explained_inertia(self):
        plt.plot(range(1, len(self.famd.explained_inertia_) + 1), np.cumsum(self.famd.explained_inertia_))
        plt.ylabel('Explained Inertia')
        plt.xlabel('Number of Principal Components')
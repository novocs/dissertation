import pandas as pd
import numpy as np
import hdbscan

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.mixture import GaussianMixture

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Reshape
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from seaborn import color_palette
import matplotlib.pyplot as plt
from .analysis.statmethods import SummaryStats


class AutoEncoder(SummaryStats):
    def __init__(
            self,
            input_data,
            output_data=None,
            max_vector_size=50,
            hidden_dims=[500, 500, 2000],
            autoencoded_embedding_size=2,
            categorical_columns=list(),
            dense_activation='relu',
            kernal_initializer='glorot_uniform',
            learned_embedding=True,
            scale=True,
            loss_function=losses.MeanSquaredError(),
            metrics=['mae']
    ):
        """
        The class supports building, fitting and storing autoencoder models discussed in the dissertation paper.
        It incorporates certain pre-processing steps: standard scaling of numerical features and nominal
        encoding of categorical features. The class also supports the generation of learned embeddings and
        HDBSCAN clustering of autoencoded embeddings and analysis of clustering results.

        Args:
            input_data (pandas.DataFrame): contains the data inputted into the autoencoder model.
            output_data (pandas.DataFrame): optional argument specifying the exact output representation of the input
                dataset. It should be noted that all integers in the dataset must be represented as floats.
                By default, the output data consists of nominal encoded categorical features and standard
                scaled numerical features.
            max_vector_size (int): This is to set the maximum vector size that can be used in a learned embedding.
            hidden_dims (list of int): This sets the dimensions of the hidden layers in the encoder. The decoder
                hidden dimensions is a mirror version
            autoencoded_embedding_size (int): The number of dimensions in thw autoencoded embedding layer.
            categorical_columns (list): a list of categorical column names in the input pandas dataframe
            dense_activation (str or class): the string or class representation of tensorflow activation functions
                to be used in the hidden latyers
            kernal_initializer (str or class): the string or class to be used in representing the tensorflow
                initialization method to be used in each hidden layer.
            learned_embedding (bool): indicator of whether learned embeddings should be used to represent categorical
                input to the autoencoder
            scale (bool): indicator of whether numerical features should be scaled
            loss_function (str or class): tensorflow representation of loss functions
            metrics (list): list of tensorflow representation of metrics to be tracked
        """

        self.X_train = input_data.copy()
        self.max_vector_size = max_vector_size
        self.hidden_dims = hidden_dims
        self.autoencoded_embedding_size = autoencoded_embedding_size
        self.categorical_cols = categorical_columns
        self.standard_scaler = StandardScaler()
        self.dense_layer_activation = dense_activation
        self.initializer = kernal_initializer
        self.learned_embedding = learned_embedding
        self.scale = scale
        self.output = output_data
        self.embedding = None
        self.gmm_bic_dict = dict()
        self.loss = loss_function
        self.metrics = metrics

        # identify numerical columns
        self.numerical_cols = list()
        for col in self.X_train.columns:
            if col not in self.categorical_cols:
                self.numerical_cols.append(col)

        # arrange column based on model input
        self.X_train = self.X_train[self.categorical_cols + self.numerical_cols]

        # fit standard scaler and nominal encoder
        self.ordinal_encoders = self.fit_ordinal_encoders()
        self.standard_scaler.fit(self.X_train[self.numerical_cols])

    def build_autoencoder_model(self):
        """
        Builds the autoencoder model. The for loop for generating the encoder and decoder hidden layers (lines 140-149)
        was partly inspired by code seen in the N2D documentation. Reference provided below.
        Advanced Usage — n2d 0.3.1 documentation. N2d.readthedocs.io.
        Retrieved July 27 , 2022, from https://n2d.readthedocs.io/en/latest/extending.html
        """
        # placeholders
        model_input_layers = list()
        cat_embeddings = list()

        numerical_cols = self.numerical_cols.copy()
        cat_cols = self.categorical_cols.copy()

        if not self.learned_embedding:
            numerical_cols += cat_cols
            cat_cols = list()

        # building learned embedding for categorical columns
        for cat_col in cat_cols:
            # count the unique atttributes
            unique_attrs = self.X_train[cat_col].nunique()

            # size of embedding vectors
            vector_size = int(min(1 + unique_attrs ** 0.25, self.max_vector_size))

            # define input layer for categorical column
            cat_input_layer = Input(shape=(1,))

            # define embedding layer
            embedding_layer = Embedding(unique_attrs + 1, vector_size, input_length=1)(cat_input_layer)
            reshaped_embedding = Reshape(target_shape=(vector_size,))(embedding_layer)

            # store layers for concatenation
            model_input_layers.append(cat_input_layer)
            cat_embeddings.append(reshaped_embedding)

        # building numerical input layer
        autoencoder_input_layers = cat_embeddings
        numerical_size = len(numerical_cols)
        model_input_layers.append(Input(shape=(numerical_size,)))
        autoencoder_input_layers.append(model_input_layers[-1])

        # create encoder inputs
        if self.learned_embedding:
            encoder_model_inputs = Concatenate(axis=-1)(autoencoder_input_layers)
        else:
            encoder_model_inputs = autoencoder_input_layers[-1]

        # creating encoder
        encoded = encoder_model_inputs
        for dim in self.hidden_dims:
            encoded = Dense(dim, activation=self.dense_layer_activation, kernel_initializer=self.initializer)(encoded)
        encoded = Dense(self.autoencoded_embedding_size, name='autencoded_embedding')(encoded)

        # creating decoder
        decoded = encoded
        for dim in self.hidden_dims[::-1]:
            decoded = Dense(dim, activation=self.dense_layer_activation, kernel_initializer=self.initializer)(decoded)

        if self.learned_embedding:
            decoded = Dense(encoder_model_inputs.shape[-1])(decoded)

        output_dim = len(self.categorical_cols + self.numerical_cols)
        if self.output is not None:
            output_dim = self.output.shape[1]

        decoded = Dense(output_dim)(decoded)

        # models for extracting encoder inputs and encoded data
        self.encoder = Model(model_input_layers, encoded)
        self.encoder_inputs = Model(model_input_layers, encoder_model_inputs)

        # creating autoencoder
        self.autoencoder = Model(inputs=model_input_layers, outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss=self.loss, metrics=self.metrics)
        self.autoencoder.summary()

    def fit_ordinal_encoders(self):
        """
        Function for encoding nominal data as integers
        Returns:
            ordinal_encoders (dict): a dictionary with keys as column name and value as ordinal encoder instances
        """

        ordinal_encoders = {}

        for column in self.categorical_cols:
            encoder = OrdinalEncoder()
            encoder.fit(np.array(self.X_train[column]).reshape(-1, 1))
            ordinal_encoders[column] = encoder

        return ordinal_encoders

    def input_data(self, input_df):
        """
        Reorganizes input data based on created model, and scales numerical features
        Args:
             input_df: the input data provided by user
        Returns:
            a representation of the input data for tensorflow
        """
        input_data = []
        data = input_df.copy()

        # add categorical data
        for col in self.categorical_cols:
            encoder = self.ordinal_encoders[col]
            input_col = np.array(data[col]).reshape(-1, 1)
            data[col] = encoder.transform(input_col)
            data[col] = data[col].astype(float)

            if self.learned_embedding:
                input_data.append(data[col])

        # scale and add numerical data
        if self.scale:
            numerical_data = self.standard_scaler.transform(data[self.numerical_cols])
            data.loc[:, self.numerical_cols] = numerical_data

        if not self.learned_embedding:
            return data[self.categorical_cols + self.numerical_cols]

        input_data.append(numerical_data)
        return input_data

    def output_data(self, data):
        """
        Standardizes numerical columns in output and converts integers to floats
        Args:
             data: the input data provided by user
        Returns:
            a representation of the output data for tensorflow
        """
        output_data = self.output

        if output_data is not None:
            return output_data

        output_data = data.copy()

        for col in self.categorical_cols:
            encoder = self.ordinal_encoders[col]
            output_col = np.array(data[col]).reshape(-1, 1)
            output_data[col] = encoder.transform(output_col)
            output_data[col] = output_data[col].astype(float)

        if self.scale and self.output is None:
            output_data.loc[:, self.numerical_cols] = self.standard_scaler.transform(data[self.numerical_cols])

        for col in output_data.columns.tolist():
            output_data[col] = output_data[col].astype(float)

        return output_data

    def fit_model(self, epochs=100, batch_size=256, filedir='models/', file_prefix='famd', early_stop=10):
        """
        Fits the autoencoder model. The best model based on the loss function will be stored at the relative file path
        and would be of the form filedir/file_prefix/dim2embedding_relu_batch256.h5
        Args
            epochs (int): the number of epochs
            batch_size (int): the number of instances fed to the model at a time during training
            filedir (str): the relative directoty for storing model results
            file_prefix (str): the prefix to the name of created files.
        :return:
        """
        train_data = self.input_data(self.X_train)
        train_result = self.output_data(self.X_train)
        monitor = 'loss'
        validation_data = None

        filename = self.model_file_name_generator(batch_size)
        self.filepath = f"{filedir}/{file_prefix}{filename}"

        checkpoint = ModelCheckpoint(
            filepath=self.filepath,
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        earlystopping = EarlyStopping(monitor='loss', patience=early_stop)

        self.history = self.autoencoder.fit(
            train_data, train_result, validation_data=validation_data, epochs=epochs, batch_size=batch_size,
            callbacks=[checkpoint, earlystopping]
        )

        train_mae = self.autoencoder.evaluate(train_data, train_result)
        print(f"Train Error: {train_mae}")

        plt.figure(figsize=(8, 8))
        plt.plot(self.history.history['loss'], label='train set')

    def load_best_model(self, filepath=None):
        """
        Loads the best created model from a stored file
        Args:
           filepath: the relative path where the model is stored
        """
        if self.filepath is not None:
            filepath = self.filepath

        if filepath is None:
            raise Exception(
                "The autoencoder model must be fitted first or a filepath to load the model from must be provided"
            )

        self.autoencoder = load_model(filepath)
        self.autoencoder.get_layer(name='autencoded_embedding').output
        self.encoder = Model(self.autoencoder.inputs, self.autoencoder.get_layer(name='autencoded_embedding').output)

    def model_file_name_generator(self, batch_size):
        """
        Generates suffix of file name for storing best autoencoder results
        Args:
            batch_size (int): the number of instances fed to the autoncoder model at a time during training.
        Returns:
             (str): filepath suffix
        """
        return f"dim{self.autoencoded_embedding_size}embedding_{self.dense_layer_activation}_batch{batch_size}.h5"

    def extract_autoencoded_embedding(self):
        self.load_best_model()
        self.embedding = self.encoder.predict(self.input_data(self.X_train))

        if self.autoencoded_embedding_size == 2:
            plt.figure(figsize=(8, 8))
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1])

    def hdbscan_clustering(self, min_cluster_size=2000, min_samples=30, metric='euclidean', generate_plot=True):
        """
        Performs HDBSCAN clustering of an autoencoded emebedding
        Args:
            min_cluster_size: smallest acceptable cluster size to be used as HDBSCAN parameter
            min_samples: the minimum samples near a point to be considered a caore point. It is also a HDBSCAN parameter
            metric: a HDBSCAN parameter for the space used in evaluating distance
            generate_plot: if true, generates visual plots for 2D embeddings
        Returns:
             hdbscan_model: an instance of a fitted hdbscan model
        """
        if self.embedding is None:
            self.extract_autoencoded_embedding()

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric
        ).fit(self.embedding)

        if self.autoencoded_embedding_size == 2 and generate_plot:
            palette = color_palette(palette='Paired', n_colors=max(model.labels_) + 1)

            cluster_colors = [
                palette[x] if x >= 0
                else (0.5, 0.5, 0.5)
                for x in model.labels_
            ]

            plt.figure(figsize=(8, 8))
            plt.scatter(
                self.embedding[:, 0], self.embedding[:, 1], s=30, linewidth=0, c=cluster_colors, alpha=0.25
            )

        return model

    def gmm_clustering(
            self, n_components=25, n_init=10, covariance_type='full', init_params='k-means++', generate_plot=True
    ):
        """
        Performs GMM clustering of an autoencoded emebedding
        Args:
            n_components (int): the number of components
            n_init (int): the number of times the model is ran prior to selecting the most probable results
            covariance_type (str): the type of cavariance specified by scikit-learn
            init_params (str): the scikit-learn specified methods of generating initialization parameters
            generate_plot: if true, generates visual plots for 2D embeddings
        Returns:
             gmm: an instance of a fitted GMM model
        """
        gmm = GaussianMixture(
            n_components=n_components,
            n_init=n_init,
            covariance_type=covariance_type,
            init_params=init_params
        ).fit(self.embedding)

        if self.autoencoded_embedding_size == 2 and generate_plot:
            labels = gmm.predict(self.embedding)
            palette = color_palette(palette='Paired', n_colors=n_components)

            cluster_colors = [
                palette[x] if x >= 0
                else (0.5, 0.5, 0.5)
                for x in labels
            ]

            plt.figure(figsize=(8, 8))
            plt.scatter(
                self.embedding[:, 0], self.embedding[:, 1], s=30, linewidth=0, c=cluster_colors, alpha=0.25
            )

        return gmm

    def gmm_bic_plot(
            self, components_list, n_init=10, covariance_type='full', init_params='k-means++', generate_plot=True,
            overwrite=False
    ):
        """
        Performs GMM clustering of an autoencoded emebedding for a range of components, extracts the BIC score and
        generates a plot.
        Args:
            components_list (list of int): the list of components
            n_init (int): the number of times the model is ran prior to selecting the most probable results
            covariance_type (str): the type of cavariance specified by scikit-learn
            init_params (str): the scikit-learn specified methods of generating initialization parameters
            generate_plot: if true, generates plots for the BIC versus number of components
            overwrite: overwrite previously generated results
        Returns:
             (dict): keys are the number of components and values are the BIC score
        """

        for num_component in components_list:
            if num_component in self.gmm_bic_dict and not overwrite:
                continue

            gmm_model = self.gmm_clustering(
                n_components=num_component, n_init=n_init, covariance_type=covariance_type, generate_plot=False,
                init_params=init_params
            )
            self.gmm_bic_dict[num_component] = gmm_model.bic(self.embedding)

        self.gmm_bic_dict = dict(sorted(self.gmm_bic_dict.items()))

        if generate_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(self.gmm_bic_dict.keys(), self.gmm_bic_dict.values(), label='BIC')
            plt.legend(loc='best')
            plt.xlabel('n_components')

        return self.gmm_bic_dict

    def hdbscan_cluster_summary(self, hdbscan_model, data, cat_cols, sort_col=None):
        """
        Summarizes the cluster results of an HDBSCAN model by aggregating means of numerical features and
        identifying modes of categorical features for each cluster
        Args:
            hdbscan_model: an instance of a hdbscan model
            data (pandas.DataFrame): the dataset that was clustered
            cat_cols (list): categorical columns in the dataset that was clustered
            sort_col (str): a feature to sort the output results by
        Returns:
            (pandas.DataFrame): dataframe containing aggregated results
        """
        return self.cluster_summary(hdbscan_model.labels_, data, cat_cols, sort_col=sort_col)

    def gmm_cluster_summary(self, gmm_model, data, cat_cols, sort_col=None):
        """
        Summarizes the cluster results of a GMM model by aggregating means of numerical features and
        identifying modes of categorical features for each cluster
        Args:
            gmm_model: an instance of a gmm model
            data (pandas.DataFrame): the dataset that was clustered
            cat_cols (list): categorical columns in the dataset that was clustered
            sort_col (str): a feature to sort the output results by
        Returns:
            (pandas.DataFrame): dataframe containing aggregated results
        """
        return self.cluster_summary(gmm_model.predict(self.embedding), data, cat_cols, sort_col=sort_col)

    def describe_approach_hdbscan(self, hdbscan_model, data, feature, inherent_feature=None):
        """
        Retrieves information to analyse HDBSCAN clusters generated from an autoencoded embedding approach
        Args:
            hdbscan_model: an instance of a hdbscan model
            data (pandas.DataFrame): the dataset that was clustered
            feature (str): the feature to be used in comparing cluster similarities using descriptive and inferential
                statistucs
            inherent_feature: the feature to be used in evaluating the ability of a clustering algorithm to separate
                a particular feature in its different consituant clustes
        Returns:
            (dict): key contains descriptions and values show the results
        """
        results = dict()

        df = pd.DataFrame()
        df[feature] = data[feature].tolist()
        df['cluster_assignment'] = hdbscan_model.labels_

        results['feature'] = feature
        results['param_minclustersize'] = hdbscan_model.min_cluster_size
        results['param_minsamples'] = hdbscan_model.min_samples

        return self._describe_approach(results, df, data, feature, inherent_feature=inherent_feature)

    def describe_approach_gmm(self, gmm_model, data, feature, inherent_feature=None):
        """
        Retrieves information to analyse GMM clusters generated from an autoencoded embedding approach
        Args:
            gmm_model: an instance of a GMM model
            data (pandas.DataFrame): the dataset that was clustered
            feature (str): the feature to be used in comparing cluster similarities using descriptive and inferential
                statistics
            inherent_feature: the feature to be used in evaluating the ability of a clustering algorithm to separate
                a particular feature in its different consituant clustes
        Returns:
            (dict): key contains descriptions and values show the results
        """
        results = dict()

        df = pd.DataFrame()
        df[feature] = data[feature].tolist()
        df['cluster_assignment'] = gmm_model.predict(self.embedding)

        results['feature'] = feature
        results['param_components'] = gmm_model.n_components
        results['param_num_init'] = gmm_model.n_init

        return self._describe_approach(results, df, data, feature, inherent_feature=inherent_feature)

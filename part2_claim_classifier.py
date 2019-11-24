import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ParameterGrid

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

MODEL_FILE_NAME = "part2_claim_classifer.h5"


class ClaimClassifier:
    def __init__(
        self, epochs=50, batch_size=64, architecture=[4, 4], dropout=True, model=None
    ):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture
        self.dropout = dropout
        self.model = model

    @staticmethod
    def over_sampler(features, labels):
        _, counts = np.unique(labels, return_counts=True)
        nb_to_pick = counts[0] - counts[1]
        idx = np.where(labels == 1)[0]
        random_sampled_features = np.random.choice(idx, nb_to_pick)

        random_sampled_features = [features[i] for i in random_sampled_features]

        random_sampled_features = np.array(random_sampled_features)
        features = np.vstack((random_sampled_features, features))
        labels = np.concatenate((np.ones(nb_to_pick), labels))

        size = features.shape[0]
        idx = np.arange(0, size)
        np.random.shuffle(idx)
        features_ = np.zeros(features.shape)
        labels_ = np.zeros(labels.shape)
        for i in range(size):
            features_[:][i] = features[:][idx[i]]
            labels_[i] = labels[idx[i]]

        features = features_
        labels = labels_
        return features, labels

    def _preprocessor(self, X_raw, y_raw=None):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A clean data set that is used for training and prediction.
        """

        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(X_raw)

        # Oversample
        if y_raw is not None:
            X_res, y_res = self.over_sampler(scaled_data, y_raw)

            return X_res, y_res
        return scaled_data

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded
        y_raw : numpy.ndarray (optional)
            A one dimensional numpy array, this is the binary target variable

        Returns
        -------
        model: The keras model fully trained on the input data
        """

        # Code necessary to visualise loss and metrics with tensorboard
        id_str = f"{self.architecture}{self.batch_size}{self.epochs}{self.dropout}"
        tbCallback = TensorBoard(log_dir=f"logs/test{id_str}")

        # Preprocess data
        X_clean, y_raw = self._preprocessor(X_raw, y_raw)

        # Get input and output layer dimensions
        input_dim = X_clean.shape[1]
        num_classes = len(np.unique(y_raw)) - 1

        # Create neural network
        model = Sequential()
        # Input layer
        model.add(Dense(self.architecture[0], input_dim=input_dim, activation="relu"))
        if self.dropout:
            model.add(Dropout(0.2))
        # Hidden layer
        model.add(
            Dense(
                self.architecture[1],
                kernel_initializer="glorot_uniform",
                activation="relu",
            )
        )
        if self.dropout:
            model.add(Dropout(0.2))
        # Output layer
        model.add(
            Dense(
                num_classes, kernel_initializer="glorot_uniform", activation="sigmoid"
            )
        )
        # Compile model
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        # Fit model to data
        model.fit(
            X_clean,
            y_raw,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=[tbCallback],
        )
        self.model = model
        self.save_model(model)
        return model

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            binary values corresponding to the prediction on whether the individual
            made a claim or not.
        """

        try:
            X_clean = self._preprocessor(X_raw)
            predictions = self.model.predict_classes(X_clean)

        except AttributeError:
            raise Exception(
                "There is no model saved on this class, please run ClaimClassifier.fit() first."
            )
        return predictions

    def predict_proba(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)"""

        try:
            X_clean = self._preprocessor(X_raw)
            predictions = self.model.predict(X_clean)

        except AttributeError:
            raise Exception(
                "There is no model saved on this class, please run ClaimClassifier.fit() first."
            )

        return predictions

    def evaluate_architecture(self, X_raw, y_raw):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # Preprocess data
        X_clean = self._preprocessor(X_raw)
        # Generate predictions
        predictions_binary = self.predict(X_clean)
        predictions_proba = self.predict_proba(X_clean)
        # Retrieve evaluation metrics
        cm = confusion_matrix(y_raw, predictions_binary)
        roc_auc = roc_auc_score(y_raw, predictions_proba)

        print(f"roc_auc: {roc_auc}")
        print(f"Confusion matrix \n {cm}")
        return roc_auc, cm

    @staticmethod
    def save_model(model):
        model.save(MODEL_FILE_NAME)


def ClaimClassifierHyperParameterSearch():  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    # Load data
    data = np.genfromtxt("part2_data.csv", delimiter=",")
    features = data[1:, :-2]
    labels = data[1:, -1]

    # Separate data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=12
    )

    # Define hyperparameters dictionary
    hyperparameters = {
        "epochs": [100, 200],
        "batch_size": [4, 8, 32, 64, 128],
        "architecture": [[2, 4], [2, 2], [4, 4], [4, 8]],
        "dropout": [True, False],
    }
    # Generate all combinations of hyperparameters
    all_params = list(ParameterGrid(hyperparameters))

    # Initialise eval metric lists
    roc_auc = []
    cm = []
    for params in all_params:
        print(f"Parameters used:{params}")
        # Generate class instance with new parameters
        cc = ClaimClassifier(**params)
        cc.fit(x_train, y_train)
        # Evaluate teh architecture
        roc_i, cm_i = cc.evaluate_architecture(x_test, y_test)
        roc_auc.append(roc_i)
        cm.append(cm_i)
    return all_params, roc_auc, cm


def load_model():
    """Loads the keras model stored."""

    model = keras.models.load_model(MODEL_FILE_NAME)
    cc = ClaimClassifier(model=model)
    return cc

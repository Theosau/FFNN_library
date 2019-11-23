import pickle
import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ParameterGrid

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard


class ClaimClassifier:
    def __init__(self, epochs=50, batch_size=16, architecture=[4, 4], dropout=True):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture
        self.dropout = dropout
        self.metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.SensitivityAtSpecificity(0.4),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]

    def over_sampler(self, features, labels):
        unique, counts = np.unique(labels, return_counts=True)
        nb_to_pick = counts[0] - counts[1]
        idx = np.where(labels==1)[0]
        random_sampled_features = np.random.choice(idx, nb_to_pick)


        random_sampled_features = [features[i] for i in random_sampled_features]


        random_sampled_features = np.array(random_sampled_features)
        features = np.vstack((random_sampled_features, features))
        labels = np.concatenate((np.ones(nb_to_pick,), labels))

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
            # full_data = np.hstack((scaled_data, y_raw.reshape(-1, 1)))
            # df = pd.DataFrame(full_data)
            # df = df.drop_duplicates().values
            # x = df[:, :-1]
            # y = df[:, -1]
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
        ?
        """
        id_str = f"{self.architecture}{self.batch_size}{self.epochs}{self.dropout}"
        tbCallback = TensorBoard(log_dir=f"logs/test{id_str}")

        X_clean, y_raw = self._preprocessor(X_raw, y_raw)

        # Config
        input_dim = X_clean.shape[1]
        num_classes = len(np.unique(y_raw)) - 1

        model = Sequential()
        model.add(Dense(self.architecture[0], input_dim=input_dim, activation="relu"))
        if self.dropout:
            model.add(Dropout(0.2))
        model.add(
            Dense(
                self.architecture[1],
                kernel_initializer="glorot_uniform",
                activation="relu",
            )
        )
        if self.dropout:
            model.add(Dropout(0.2))
        model.add(
            Dense(
                num_classes, kernel_initializer="glorot_uniform", activation="sigmoid"
            )
        )
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=self.metrics
        )

        model.fit(
            X_clean,
            y_raw,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=[tbCallback],
        )
        self.model = model

#         self.save_model()

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
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        try:
            X_clean = self._preprocessor(X_raw)
            predictions = self.model.predict(X_clean)
            probability = np.count_nonzero(predictions) / np.size(X_clean)

        except AttributeError:
            raise (
                "There is no model saved on this class, please run ClaimClassifier.fit() first."
            )

        return probability

    def predict_classes(self, X_raw):
        try:
            X_clean = self._preprocessor(X_raw)
            predictions = self.model.predict_classes(X_clean)

        except AttributeError:
            raise (
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
        X_clean = self._preprocessor(X_raw)
        predictions = self.model.predict_classes(X_clean)
        cm = confusion_matrix(y_raw, predictions)
        scores = self.model.evaluate(X_clean, y_raw)
        roc_auc = roc_auc_score(y_raw, predictions)
        del self.model
        print(f"roc_auc: {roc_auc}")
        print(f"accuracy: {scores[0]}")
        print(f"Confusion matrix \n {cm}")
        print(f"Evaluation scores {scores}\n\n")
        return roc_auc, scores, cm

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch():  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    data = np.genfromtxt("part2_data.csv", delimiter=",")
    features = data[1:, :-3]
    labels = data[1:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=12
    )

    hyperparameters = {
        "epochs": [100, 200, 400],
        "batch_size": [32, 64, 128],
        "architecture": [[4, 4], [4, 8], [4, 12]],
        "dropout": [True, False],
    }
    roc_auc = []
    accuracy = []
    scores = []
    cm = []
    all_params = list(ParameterGrid(hyperparameters))
    for params in all_params:
        print(f"Parameters used:{params}")
        cc = ClaimClassifier(**params)
        cc.fit(x_train, y_train)
        res = cc.evaluate_architecture(x_test, y_test)
        roc_auc.append(res[0])
        scores.append(res[1])
        accuracy.append(res[1][0])
        cm.append(res[2])
    return roc_auc, all_params, scores, accuracy, cm

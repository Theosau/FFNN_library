import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam


class PricingModel:
    def __init__(
        self, linear_model=False, calibrate_probabilities=False, base_classifier=None
    ):
        """Initialise placeholder instance variables."""

        self.mean_claim = None  # Placeholder for mean claim
        self.encoder = None  # Placeholder for onehot encoder
        self.pca_model = None  # Placeholder for fitted pca model
        self.calibrate = (
            calibrate_probabilities
        )  # Boolean on whether to calibrate or not
        self.linear_model = linear_model  # Run the linear model

        # This is the placeholder for our model
        if base_classifier is None:
            self.base_classifier = self.build_base_classifier()
        else:
            self.base_classifier = base_classifier

    @staticmethod
    def build_base_classifier():
        """Initialise keras neural network."""

        model = Sequential()
        return model

    @staticmethod
    def oversampler(features, labels):
        """Run random oversampler"""

        labels = np.reshape(labels, (len(labels),))

        # Get number of 0s and 1s
        _, counts = np.unique(labels, return_counts=True)
        # Difference between 0s and 1s count
        nb_to_pick = counts[0] - counts[1]
        # Indices corresponding to 1 labels that need to be oversampled
        idx = np.where(labels == 1)[0]

        # Randomly pick rows to duplicate
        random_sampled_features = np.random.choice(idx, nb_to_pick)
        random_sampled_features = [features[i] for i in random_sampled_features]
        random_sampled_features = np.array(random_sampled_features)

        # Features and labels reshaping
        features = np.vstack((random_sampled_features, features))
        labels = np.concatenate((np.ones(nb_to_pick), labels))

        # Shuffle both features and labels
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

    def run_pca(self, X_continuous):
        """Run PCA on clean continuous data."""

        # Hardcode number of pca features to keep
        n = 9

        # If we are training we need to create the PCA model, otherwise use already fitted model
        if self.pca_model is None:
            pca = PCA(n_components=n)
            pca.fit(X_continuous)

            # Save PCA model for testing
            self.pca_model = pca

        # Transform data into 9 PCA components
        float_dict = {}
        reduced_data = self.pca_model.fit_transform(X_continuous)
        for i in range(n):
            float_dict["PCA_{}".format(i + 1)] = reduced_data[:, i]
        pca_reduced_data = pd.DataFrame.from_dict(float_dict)

        return pca_reduced_data

    def run_onehot(self, data, one_hot_strings):
        """One hot encode categorical features"""

        # For testing we use the already fitted one hot encoder
        if self.encoder is not None:
            enc = self.encoder
        else:
            enc = OneHotEncoder()
            enc.fit(data[one_hot_strings])
            self.encoder = enc

        # Return dataframe with one hot encoded categorical features
        one_hot_data = pd.DataFrame(enc.transform(data[one_hot_strings]).toarray())

        return one_hot_data

    def _preprocessor(self, X_raw):
        """Data preprocessing function.
 
        This function prepares the features of the data for training,
        evaluation, and prediction.
 
        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
 
        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """

        # Keep drv_sex2 column and replace nans with N
        X_raw["drv_sex2"].fillna("N", inplace=True)
        # Drop all columns with nans
        X_raw.dropna(inplace=True, axis=1)

        # Categorical features we want to keep and one hot encode
        one_hot_cols = [
            "pol_coverage",
            "pol_pay_freq",
            "pol_payd",
            "pol_usage",
            "drv_drv2",
            "drv_sex1",
            "drv_sex2",
            "vh_fuel",
            "vh_type",
        ]

        # One hot encode data
        one_hotted_data = self.run_onehot(X_raw, one_hot_cols)

        # Run PCA on continuous features
        PCA_cols = X_raw.select_dtypes(exclude=[object]).columns
        float_data = X_raw[PCA_cols]

        # Normalize and scale data appropriately
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(float_data)

        X_clean_continuous = pd.DataFrame(scaled_data, columns=PCA_cols)
        pca_reduced_data = self.run_pca(X_clean_continuous)

        # Merge one hot encoded categorical features and PCA columns into dataframe
        X_clean = pd.concat([pca_reduced_data, one_hotted_data], axis=1)

        return X_clean

    def fit(self, X_raw, y_raw, claims_raw, epochs=100, batch_size=128):
        """Classifier training function.
 
        Here you will use the fit function for your classifier.
 
        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims
 
        Returns
        -------
        self: (optional)
            an instance of the fitted model
 
        """

        # Get the mean claim
        claim_made_idx = np.where(claims_raw != 0)[0]
        self.mean_claim = np.mean(claims_raw[claim_made_idx])  # mean claim severity

        # Preprocess data
        X_clean = self._preprocessor(X_raw)

        # Features and labels come out of oversampler as np arrays rather than dataframes
        features, labels = self.oversampler(X_clean.values, y_raw.values)

        # Neural network input and output shape
        input_shape = features.shape
        num_classes = len(np.unique(labels)) - 1

        # Non linear model
        if self.linear_model is False:
            # Input layer
            self.base_classifier.add(
                Dense(
                    36,
                    input_dim=input_shape[1],
                    kernel_constraint=maxnorm(3),
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
            self.base_classifier.add(Dropout(0.3))
            # Hidden layer
            self.base_classifier.add(
                Dense(
                    24,
                    kernel_constraint=maxnorm(3),
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
            self.base_classifier.add(Dropout(0.3))
            # Hidden layer
            self.base_classifier.add(
                Dense(
                    12,
                    kernel_constraint=maxnorm(3),
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
            self.base_classifier.add(Dropout(0.3))
            # Output layer
            self.base_classifier.add(
                Dense(
                    num_classes,
                    kernel_constraint=maxnorm(3),
                    kernel_initializer="glorot_uniform",
                    activation="sigmoid",
                )
            )
        else:  # Linear model: logistic regression unit
            self.base_classifier.add(
                Dense(
                    1,
                    input_dim=input_shape[1],
                    kernel_initializer="glorot_uniform",
                    activation="sigmoid",
                )
            )
        # Compule moel
        self.base_classifier.compile(
            loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"]
        )

        # Fit model to data with or without calibration
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, features, labels
            )
        else:
            self.base_classifier = self.base_classifier.fit(
                features, labels, epochs=epochs, batch_size=batch_size, verbose=0
            ).model

        # Save model as a pickle
        self.save_model()

        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.
 
        Here you will implement the predict function for your classifier.
 
        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
 
        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # Predict claim probabilities if a model has already been generated
        try:
            X_clean = self._preprocessor(X_raw)
            predictions = self.base_classifier.predict(X_clean)
        except AttributeError:
            raise Exception(
                "There is no model saved on this class, please run PricingModel.fit() first."
            )
        return predictions

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.
 
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
        # Pricing strategy
        if self.linear_model:
            phi = 0.25
        else:
            phi = 0.16
        res = (
            self.predict_claim_probability(X_raw) * self.mean_claim * phi
            + self.mean_claim * 0.02
        )

        # Reshape results for testing set
        res = res.reshape(len(res))
        return res

    def save_model(self):
        """Saves the class instance as a pickle file."""

        # Change pickle file name according to model
        if self.linear_model:
            file_name = "part3_pricing_model_linear.pickle"
        else:
            file_name = "part3_pricing_model.pickle"
        with open(file_name, "wb") as target:
            pickle.dump(self, target)

    def evaluate_model(self, X_raw, y_raw):
        """Evaluate model according to ROC and confusion matrix"""

        # Preprocess
        X_clean = self._preprocessor(X_raw)
        # Fetch class predictions
        predictions = self.base_classifier.predict_classes(X_clean)
        # Fetch probabilities of having made a claim
        predictions_proba = self.base_classifier.predict(X_clean)
        # Generate confusion matrix
        cm = confusion_matrix(y_raw, predictions)
        # Retrieve evaluation scores from metrics stored in model
        scores = self.base_classifier.evaluate(X_clean, y_raw)
        # Get ROC-AUC score
        roc_auc = roc_auc_score(y_raw, predictions_proba)

        # Display results
        print(f"roc_auc: {roc_auc}")
        print(f"accuracy: {scores[0]}")
        print(f"Confusion matrix \n {cm}")
        print(f"Evaluation scores {scores}\n\n")

        return roc_auc, scores, cm


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0
    )
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method="sigmoid", cv="prefit"
    ).fit(X_cal, y_cal)
    return calibrated_classifier

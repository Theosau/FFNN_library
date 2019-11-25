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
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.linear_model = linear_model
        if base_classifier is None:
            self.base_classifier = self.build_base_classifier()
        else:
            self.base_classifier = base_classifier

    @staticmethod
    def build_base_classifier():
        model = Sequential()
        return model

    @staticmethod
    def oversampler(features, labels):
        labels = np.reshape(labels, (len(labels),))

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

    def run_pca(self, corrected_float_data, bTrain=True):
        n = 10

        if bTrain:
            # FIND THE PCA FIT
            pca = PCA(n_components=n)  # len(data_float.columns[:-1]))
            pca.fit(corrected_float_data)

            # SAVE PCA FOR TESTING...
            self.pca_model = pca

        # TRANSFORM DATA INTO n = 15 PCA COMPONENTS
        float_dict = {}
        reduced_data = self.pca_model.fit_transform(corrected_float_data)
        for i in range(n):
            float_dict["PCA_{}".format(i + 1)] = reduced_data[:, i]
        pca_reduced_data = pd.DataFrame.from_dict(float_dict)

        return pca_reduced_data

    def run_onehot(self, data, one_hot_strings):

        if "encoder" in self.__dict__:
            enc = self.encoder
        else:
            enc = OneHotEncoder()
            enc.fit(data[one_hot_strings])
            self.encoder = enc

        one_hot_data = pd.DataFrame(enc.transform(data[one_hot_strings]).toarray())
        return one_hot_data

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw, bTrain=False):
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

        X_raw["drv_sex2"].fillna("N", inplace=True)
        X_raw.dropna(inplace=True, axis=1)

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
        one_hotted_data = self.run_onehot(
            X_raw, one_hot_cols
        )  # NEW X_RAW_DISCRETISED TO ONEHOT

        PCA_cols = X_raw.select_dtypes(exclude=[object]).columns
        float_data = X_raw[PCA_cols]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(float_data)
        corrected_float_data = pd.DataFrame(scaled_data, columns=PCA_cols)
        pca_reduced_data = self.run_pca(
            corrected_float_data, bTrain
        )  # NEW X_RAW_FLOAT

        X_clean = pd.concat([pca_reduced_data, one_hotted_data], axis=1)

        return X_clean

    def fit(
        self,
        X_raw,
        y_raw,
        claims_raw,
        nn_size=32,
        epochs=100,
        batch_size=128,
    ):
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
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])  # MEAN CNST FOR SEVERITY

        X_clean = self._preprocessor(X_raw, bTrain=True)
        y_raw = pd.DataFrame.from_dict({"made_claim": y_raw})

        # OUTPUTS NOT A DATAFRAME ANYMORE BUT A NUMPY ARRAY
        features, labels = self.oversampler(
            X_clean.values, y_raw.values
        )

        # NN CONFIG
        input_shape = features.shape
        num_classes = len(np.unique(labels)) - 1


        # MLP
        if self.linear_model is False:
            # Input layer
            self.base_classifier.add(
                Dense(
                    nn_size,
                    input_dim=input_shape[1],
                    kernel_constraint=maxnorm(
                        3
                    ),
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
            self.base_classifier.add(Dropout(0.3))
            # Hidden layer
            self.base_classifier.add(
                Dense(
                    nn_size,
                    kernel_constraint=maxnorm(
                        3
                    ),
                    kernel_initializer="glorot_uniform",
                    activation="relu",
                )
            )
            self.base_classifier.add(Dropout(0.3))
            # Output layer
            self.base_classifier.add(
                Dense(
                    num_classes,
                    kernel_constraint=maxnorm(
                        3
                    ),
                    kernel_initializer="glorot_uniform",
                    activation="sigmoid",
                )
            )
        else:
            # DO THIS FOR A LOGISTIC REGRESSION UNIT
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

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, features, labels
            )
        else:
            self.base_classifier = self.base_classifier.fit(
                features, labels, epochs=epochs, batch_size=batch_size, verbose=0
            ).model

        # Save model as a pickly
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
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        if self.linear_model:
            threshold = 8
        else:
            threshold = 2.28
            # 2.28
        res = self.predict_claim_probability(X_raw) * self.y_mean * threshold
        res = res.reshape(len(res))
        return res

    def save_model(self):
        """Saves the class instance as a pickle file."""

        # Change pickle file name according to model
        if self.linear_model:
            file_name = 'part3_pricing_model_linear.pickle'
        else:
            file_name = 'part3_pricing_model.pickle'
        with open(file_name, "wb") as target:
            pickle.dump(self, target)


    def evaluate_model(self, X_raw, y_raw):
        """Architecture evaluation utility.
            TESTING, X_raw,y_raw are evaluation datasets
        """
        X_clean = self._preprocessor(X_raw, bTrain=False)
        predictions = self.base_classifier.predict_classes(X_clean)
        predictions_proba = self.base_classifier.predict(X_clean)
        cm = confusion_matrix(y_raw, predictions)
        scores = self.base_classifier.evaluate(X_clean, y_raw)
        roc_auc = roc_auc_score(y_raw, predictions_proba)

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


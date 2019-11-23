from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense
import keras

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel(object):
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
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
        self.base_classifier = self.build_base_classifier()

    def build_base_classifier(self):
        model = Sequential()
        return model

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(X_raw)
        return  scaled_data

    def fit(self, X_raw, y_raw, claims_raw):
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
        self.y_mean = np.mean(claims_raw[nnz]) # MEAN CNST FOR SEVERITY

        X_CLEAN = self.prepare_data_preprocessing(X_raw)
        CLAIM = pd.DataFrame.from_dict({'claim_amount':claims_raw})
        Y_RAW = pd.DataFrame.from_dict({'made_claim':y_raw})

        # NN CONFIG
        input_shape = X_CLEAN.shape
        num_classes = 1
        epochs = 100
        batch_size = 128
        METRICS = [
              keras.metrics.TruePositives(name='tp'),
              keras.metrics.FalsePositives(name='fp'),
              keras.metrics.SensitivityAtSpecificity(1),
              keras.metrics.TrueNegatives(name='tn'),
              keras.metrics.FalseNegatives(name='fn'),
              keras.metrics.BinaryAccuracy(name='accuracy'),
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
        ]

        # Input layer
        self.base_classifier.add(Dense(12,input_dim=input_shape[1], activation= 'relu'))
        # Hidden layer
        self.base_classifier.add(Dense(12, kernel_initializer = 'glorot_uniform',activation = 'relu'))
        # Output layer
        self.base_classifier.add(Dense(num_classes, kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))
        # Compile model
        self.base_classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [METRICS])

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_CLEAN, Y_RAW)
        else:
            self.base_classifier = self.base_classifier.fit(X_CLEAN, Y_RAW, epochs = epochs, batch_size = batch_size, verbose = 1)
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
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)


        return  # return probabilities for the positive class (label 1)

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
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)
        return None

    def get_data(self):
        data = pd.read_table('part3_data.csv',delimiter = ',',index_col = 0)
        data.drop_duplicates()
        data['drv_sex2'].fillna('N',inplace = True)
        data.dropna(inplace = True)
        return data

    def prepare_data_preprocessing(self,data):
        # Data is a pandas dataframe of X_raw
        string_cols = data.select_dtypes([object]).columns
        one_hot_strings = string_cols[[1,2,3,5,6,7]] # Selected columns for cosideration (one-hot)

        PCA_cols = data.select_dtypes(exclude = [object]).columns
        float_data = data[PCA_cols] #exclude claim and made claims
        corrected_float_data = pd.DataFrame(self._preprocessor(float_data),columns = PCA_cols)

        pca_reduced_data = self.PCA_complete(corrected_float_data,PCA_cols,n=15) # NEW X_RAW_FLOAT
        one_hotted_data = self.onehot_complete(data,one_hot_strings) # NEW X_RAW_DISCRETISED TO ONEHOT
        X_RAW = pd.concat([pca_reduced_data,one_hotted_data],axis = 1)
        # CLAIM = pd.DataFrame.from_dict({'claim_amount':data['claim_amount'].values})
        # Y = pd.DataFrame.from_dict({'made_claim':data['made_claim'].values})

        return X_RAW

    def PCA_complete(self,corrected_float_data,PCA_cols,n):
        # Reduced dimension number
        n = 15

        # FIND THE PCA FIT
        pca = PCA(n_components=n)#len(data_float.columns[:-1]))
        pca.fit(corrected_float_data)
        var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        print(np.sum(pca.explained_variance_ratio_))

        # TRANSFORM DATA INTO n = 15 PCA COMPONENTS
        float_dict = {}
        reduced_data = pca.fit_transform(corrected_float_data[PCA_cols[:-2]])
        for i in range(n):
            float_dict['PCA_{}'.format(i+1)] = reduced_data[:,i]
        pca_reduced_data = pd.DataFrame.from_dict(float_dict)

        # plt.plot(range(0,n+1), [0] + list(var))
        # plt.show()

        return pca_reduced_data

    def onehot_complete(self,data,one_hot_strings):
        one_hotted_dict = {}

        for hot in one_hot_strings:
            values = data[hot].values
            # # Value encode
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(values)
            # # binary encode
            onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            for n, index in enumerate(np.unique(values)):
                one_hotted_dict['{}_{}'.format(hot,index)] = onehot_encoded[:,n]
        one_hot_data = pd.DataFrame.from_dict(one_hotted_dict)
        return one_hot_data

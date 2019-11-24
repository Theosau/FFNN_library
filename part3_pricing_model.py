from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from keras.regularizers import l2

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

# VERSION ALVARO

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
    def __init__(self,linear_model = False, calibrate_probabilities=False):
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

    def fit(self, X_raw, y_raw, claims_raw, nn_size = 32, epochs = 100, batch_size = 128, regularise = 0.001):
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

        X_CLEAN = self.prepare_data_preprocessing(X_raw,bTrain=True)
        CLAIM = pd.DataFrame.from_dict({'claim_amount':claims_raw})
        Y_RAW = pd.DataFrame.from_dict({'made_claim':y_raw})

        # OUTPUTS NOT A DATAFRAME ANYMORE BUT A NUMPY ARRAY
        features, labels, claims = self.over_sampler(X_CLEAN.values,Y_RAW.values,CLAIM.values)

        # #SPLIT TEST TRAIN DATASETS
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.85, random_state=0)

        # # SAVE FOR EVALUATION...
        # self.X_test = X_test
        # self.y_test = y_test

        # NN CONFIG
        input_shape = features.shape
        num_classes = 1
        epochs = epochs
        batch_size = batch_size
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
        # MLP
        if self.linear_model is False:
            # Input layer
            self.base_classifier.add(Dense(nn_size,input_dim=input_shape[1],kernel_regularizer=l2(regularise), bias_regularizer=l2(regularise),
            kernel_initializer = 'glorot_uniform', activation= 'relu'))
            # Hidden layer
            # self.base_classifier.add(Dense(nn_size,kernel_regularizer=l2(regularise), bias_regularizer=l2(regularise),
            # kernel_initializer = 'glorot_uniform',activation = 'relu'))
            # # Hidden layer
            self.base_classifier.add(Dense(nn_size,kernel_regularizer=l2(regularise), bias_regularizer=l2(regularise),
            kernel_initializer = 'glorot_uniform',activation = 'relu'))
            # Output layer
            self.base_classifier.add(Dense(num_classes, kernel_regularizer=l2(regularise), bias_regularizer=l2(regularise),
            kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))
            # Compile model
        else:
            # DO THIS FOR A LOGISTIC REGRESSION UNIT
            self.base_classifier.add(Dense(nn_size,input_dim=input_shape[1], kernel_initializer = 'glorot_uniform', activation= 'relu'))
        self.base_classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [METRICS])

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, features, labels)
        else:
            self.base_classifier = self.base_classifier.fit(features, labels, epochs = epochs, batch_size = batch_size, verbose = 0)
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
            X_CLEAN = self.prepare_data_preprocessing(X_raw)
            predictions = self.base_classifier.predict(X_CLEAN)
        except AttributeError:
            raise (
                "There is no model saved on this class, please run ClaimClassifier.fit() first."
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

        return self.predict_claim_probability(X_raw) * self.y_mean

    def over_sampler(self, features, labels, claims):
        labels = np.reshape(labels,(len(labels),))
        claims = np.reshape(labels,(len(claims),))

        unique, counts = np.unique(labels, return_counts=True)
        nb_to_pick = counts[0] - counts[1]
        idx = np.where(labels==1)[0]

        random_sampled_features = np.random.choice(idx, nb_to_pick)

        random_sampled_features = [features[i] for i in random_sampled_features]

        random_sampled_features = np.array(random_sampled_features)

        features = np.vstack((random_sampled_features, features))
        labels = np.concatenate((np.ones(nb_to_pick,), labels))
        claims = np.concatenate((np.ones(nb_to_pick,), claims))

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
        return features, labels, claims

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

    def prepare_data_preprocessing(self,data,bTrain = True):
        # Data is a pandas dataframe of X_raw
        string_cols = data.select_dtypes([object]).columns
        one_hot_strings = string_cols[[0,1,2,3,5,6,7,8,9,11]] # Selected columns for cosideration (one-hot)
        # [0,1,2,3,5,6,7,11]
        PCA_cols = data.select_dtypes(exclude = [object]).columns
        float_data = data[PCA_cols] #exclude claim and made claims
        corrected_float_data = pd.DataFrame(self._preprocessor(float_data),columns = PCA_cols)

        pca_reduced_data = self.PCA_complete(corrected_float_data,PCA_cols,bTrain) # NEW X_RAW_FLOAT
        one_hotted_data = self.onehot_complete(data,one_hot_strings) # NEW X_RAW_DISCRETISED TO ONEHOT
        X_RAW = pd.concat([pca_reduced_data,one_hotted_data],axis = 1)
        # CLAIM = pd.DataFrame.from_dict({'claim_amount':data['claim_amount'].values})
        # Y = pd.DataFrame.from_dict({'made_claim':data['made_claim'].values})

        return X_RAW

    def PCA_complete(self,corrected_float_data,PCA_cols,bTrain = True):
        # Reduced dimension number
        n = 15

        if bTrain:
            # FIND THE PCA FIT
            pca = PCA(n_components=n) #len(data_float.columns[:-1]))
            pca.fit(corrected_float_data)
            var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
            print(np.sum(pca.explained_variance_ratio_))

            # SAVE PCA FOR TESTING...
            self.pca_model  = pca

        # TRANSFORM DATA INTO n = 15 PCA COMPONENTS
        float_dict = {}
        reduced_data = self.pca_model.fit_transform(corrected_float_data[PCA_cols[:-2]])
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

    def evaluate_model(self, X_raw, y_raw):
        """Architecture evaluation utility.
            TESTING, X_raw,y_raw are evaluation datasets
        """
        X_clean = self.prepare_data_preprocessing(X_raw,bTrain=False)
        predictions = self.base_classifier.model.predict_classes(X_clean)
        predictions_proba = self.base_classifier.model.predict(X_clean)
        cm = confusion_matrix(y_raw, predictions)
        scores = self.base_classifier.model.evaluate(X_clean, y_raw)
        roc_auc = roc_auc_score(y_raw, predictions_proba)

        print(f"roc_auc: {roc_auc}")
        print(f"accuracy: {scores[0]}")
        print(f"Confusion matrix \n {cm}")
        print(f"Evaluation scores {scores}\n\n")

        return roc_auc, scores, cm

def run_all(self,save_model = False,fraction = 0.85):
    # FUNCTION USED TO EXCTRACT ALL VALUES OF INTEREST, MAINLY USED FOR DEBUGGING
    P3 = PricingModel()

    data = P3.get_data()

    indexes = np.arange(np.shape(data)[0])
    np.random.shuffle(indexes)

    train_indexes, test_indexes = indexes[:int(fraction*len(indexes))], indexes[int(fraction*len(indexes)):]
    data_train, data_test = data.iloc[train_indexes], data.iloc[test_indexes]

    # TRAIN THE MLP Classifier
    P3.fit(data_train[data_train.columns[:-2]],data_train.made_claim,data_train.claim_amount)

    # EVALUATE ON THE TEST DATASETS
    roc_auc, scores, cm = self.evaluate_model(data_train[data_train.columns[:-2]],data_train.made_claim)

    if save_model:
        self.save_model()

    return roc_auc, scores, cm

# import pandas as pd
# import numpy as np
# import pickle
#
# from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
#
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
#
#
# import imblearn
#
# class ClaimClassifier:
#     def __init__(self,):
#         """
#         Feel free to alter this as you wish, adding instance variables as
#         necessary.
#         """
#         pass
#
#     def _preprocessor(self, X_raw):
#         """Data preprocessing function.
#
#         This function prepares the features of the data for training,
#         evaluation, and prediction.
#
#         Parameters
#         ----------
#         X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
#             A numpy array, this is the raw data as downloaded
#
#         Returns
#         -------
#         X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
#             A clean data set that is used for training and prediction.
#         """
#
#         scaler = preprocessing.StandardScaler()
#         scaled_data = scaler.fit_transform(X_raw)
#
#         return  scaled_data
#
#     def fit(self, X_raw, y_raw):
#         """Classifier training function.
#
#         Here you will implement the training function for your classifier.
#
#         Parameters
#         ----------
#         X_raw : numpy.ndarray
#             A numpy array, this is the raw data as downloaded
#         y_raw : numpy.ndarray (optional)
#             A one dimensional numpy array, this is the binary target variable
#
#         Returns
#         -------
#         ?
#         """
#         X_clean = self._preprocessor(X_raw)
#
#         # Config
#         input_shape = X_clean.shape
#         num_classes = 1
#         epochs = 50
#         batch_size = 64
#         # pos = labels.sum()
#         # neg = len(labels)-labels.sum()
#         # weight_for_0 = (1 / neg)*(pos+neg)/2.0
#         # weight_for_1 = (1 / pos)*(pos+neg)/2.0
#         # class_weight = {0: weight_for_0, 1: weight_for_1}
#         METRICS = [
#               keras.metrics.TruePositives(name='tp'),
#               keras.metrics.FalsePositives(name='fp'),
#               keras.metrics.SensitivityAtSpecificity(1),
#               keras.metrics.TrueNegatives(name='tn'),
#               keras.metrics.FalseNegatives(name='fn'),
#               keras.metrics.BinaryAccuracy(name='accuracy'),
#               keras.metrics.Precision(name='precision'),
#               keras.metrics.Recall(name='recall'),
#               keras.metrics.AUC(name='auc'),
#         ]
#
#         model = Sequential()
#         # Input layer
#         model.add(Dense(12,input_dim=8, activation= 'relu'))
#
#         # Hidden layer
#         model.add(Dense(20, kernel_initializer = 'glorot_uniform',activation = 'relu'))
#
#         # Output layer
#         model.add(Dense(
#             num_classes, kernel_initializer = 'glorot_uniform',
#             activation = 'sigmoid'))
#         # Compile model
#
#         # model.compile(loss = get_cohen_kappa_score(weights=class_weight), optimizer = 'adam', metrics = [METRICS])
#         model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [METRICS])
#
#         # # Fit model
#         model.fit(
#             X_clean, y_raw,
#             epochs = epochs, batch_size = batch_size, verbose = 0,
#         #     class_weight=class_weight
#         )
#         self.model = model
#
#         self.save_model()
#         return model
#
#     def predict(self, X_raw):
#         """Classifier probability prediction function.
#
#         Here you will implement the predict function for your classifier.
#
#         Parameters
#         ----------
#         X_raw : numpy.ndarray
#             A numpy array, this is the raw data as downloaded
#
#         Returns
#         -------
#         numpy.ndarray
#             A one dimensional array of the same length as the input with
#             values corresponding to the probability of beloning to the
#             POSITIVE class (that had accidents)
#         """
#
#         try:
#             X_clean = self._preprocessor(X_raw)
#             predictions = self.model.predict(X_clean)
#             probability = np.count_nonzero(predictions)/np.size(X_clean)
#
#         except AttributeError:
#             raise("There is no model saved on this class, please run ClaimClassifier.fit() first.")
#
#         return  predictions
#
#     def evaluate_architecture(self):
#         """Architecture evaluation utility.
#
#         Populate this function with evaluation utilities for your
#         neural network.
#
#         You can use external libraries such as scikit-learn for this
#         if necessary.
#         """
#         pass
#
#     def save_model(self):
#         with open("part2_claim_classifier.pickle", "wb") as target:
#             pickle.dump(self, target)
#
#
# def ClaimClassifierHyperParameterSearch():  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
#     """Performs a hyper-parameter for fine-tuning the classifier.
#
#     Implement a function that performs a hyper-parameter search for your
#     architecture as implemented in the ClaimClassifier class.
#
#     The function should return your optimised hyper-parameters.
#     """
#
#     return  # Return the chosen hyper parameters

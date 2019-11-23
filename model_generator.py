import numpy
import keras
import tensorflow as tf

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.utils import np_utils
import random

from os import mkdir

from keras.callbacks import TensorBoard


def compute_cm(X_train, y_train, model):
    """Computes the confusion matrix for a given dataset and tree."""
    predictions = train_predictions(model, X_train)

    confusion_matrix = numpy.zeros((2, 2))

    for i in range(y_train.shape[0]):  # For each row from the validation set

        if (
            predict(i, predictions, y_train)
        ):  # If the tree predicted the label correctly
            confusion_matrix[
                y_train[i], y_train[i]
            ] += 1  # At position (correct label, correct label) we add 1

        else:  # If the tree predicted wrongly
            confusion_matrix[
                int(y_train[i]), int(predictions[i])
            ] += 1  # At position (correct label, predicted label) we add 1
    return confusion_matrix

def accuracy(X, y, model):
    C = compute_cm(X, y, model)
    return ((C[0,0] + C[1,1])/(sum(C)[0] + sum(C)[1]))

def metric_test(y_pred, y_true):
    #print(y_pred.shape, y_true.shape)
    a = tf.keras.backend.sum(y_true) - tf.keras.backend.sum(y_pred)
    #a = numpy.log(a)
    return a

    #return y_true.sum()- y_pred.sum()
    

def model_v0_trial(num_inputs, num_classes):
    #create model
    #opt = keras.optimizers.SGD(lr=0.01)
    
    model = Sequential()
    #model.add(BatchNormalization())
    
    model.add(Dense(num_inputs, input_dim = num_inputs, kernel_initializer = 'random_uniform', activation = 'relu'))
    model.add(Dense(num_inputs, input_dim = 4, kernel_initializer = 'RandomNormal', activation = 'relu'))
    #model.add(Dense(10, input_dim = 4, kernel_initializer = 'RandomNormal', activation = 'relu'))
    
    model.add(Dense(num_classes, kernel_initializer = 'RandomNormal', activation = 'sigmoid'))
    #compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])#,metric_test])
    
#     model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'binary_accuracy','sparse_categorical_accuracy', metric_test])
    return model

def train_model(X_train, y_train, X_test, y_test, model, epochs, batch_size, folder):
    tbCallback = TensorBoard(log_dir="logs/"+ folder)
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size = batch_size, verbose = 2, 
              callbacks = [tbCallback]) #switch epochs to 40
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose = 0)
    print("Baseline error : %.2f%%" %(100-scores[1]*100))
    return

def train_model_batch(training, validation, model, epochs, folder):
    tbCallback = TensorBoard(log_dir="logs/"+ folder)
    model.fit_generator(generator=training, validation_data = validation, epochs = epochs, verbose = 2, 
              callbacks = [tbCallback]) #switch epochs to 40
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose = 0)
    print("Baseline error : %.2f%%" %(100-scores[1]*100))
    return

def train_predictions(model, file):
    # Predict 
    predictions = model.predict_classes(file)
    print("predictions done")
    return predictions

def predict(num, predictions, line):
    # test it out
    #print(line[num])
    #print("I predicted a %s" %(predictions[num]))
    if line[num] == predictions[num]:
        return True
    else :
        return False


def save_model(model, model_name):
    model_json = model.to_json()
    model_directory = "./models/" + model_name
    mkdir(model_directory)
    model_name = model_directory + "/" + model_name
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name+".h5")
    print("Saved model as " + model_name)
    return

def load_model(model_name):
    model_name = "./models/"+model_name+"/"+model_name
    json_file = open(model_name+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print("Model loaded " + model_name)
    return loaded_model

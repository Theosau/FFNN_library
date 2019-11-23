from keras.utils import Sequence
import numpy as np
import ast
# import model_generator as mg

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

from keras.utils import plot_model

def partitionize(Xraw, Yraw, ratio = (3,1)):
    size = Xraw.shape[0]
    idx = np.arange(0, size)
    np.random.shuffle(idx)
    X_shuffle = np.zeros(Xraw.shape)
    Y_shuffle = np.zeros(Yraw.shape)
    for i in range(size):
        X_shuffle[:][i] = Xraw[:][idx[i]]
        Y_shuffle[i] = Yraw[idx[i]]

    Xraw = X_shuffle
    Yraw = Y_shuffle

    partition = {"train" : [], "validation" : []}
    labels = {}
    pos = 0
    while pos != size -1 :
        try :
            for i in range(pos, pos + ratio[0], 1):
                idx = str(list(Xraw[i]))
                partition["train"].append(idx)
                labels[idx] = int(Yraw[i])

            pos = pos + ratio[0]

            for i in range(pos, pos + ratio[1], 1):
                print(i)
                idx = str(list(Xraw[i]))
                partition["validation"].append(idx)
                labels[idx] = int(Yraw[i])
                
            pos = pos+ratio[1]
        except IndexError:
            return partition, labels

    return partition, labels

class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(8,1), n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.array(ast.literal_eval(ID)).reshape(8,1)

            # Store class
            y[i] = self.labels[ID]
        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


def model_v0_trial(num_inputs, num_classes):
    #create model
    
    model = Sequential()
    
    model.add(Dense(4, input_shape=(num_inputs,1,), kernel_initializer = 'random_uniform', activation = 'relu'))
    plot_model(model, to_file='model.png')
    model.add(Dense(8, input_dim = 4, kernel_initializer = 'RandomNormal', activation = 'relu'))
    
    model.add(Dense(num_classes, kernel_initializer = 'RandomNormal', activation = 'sigmoid'))
    #compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def train_model_batch(training, validation, model, epochs, folder):
    tbCallback = TensorBoard(log_dir="logs/"+ folder)
    model.fit_generator(generator=training, validation_data = validation, epochs = epochs, verbose = 2, 
              callbacks = [tbCallback]) #switch epochs to 40
    # Final evaluation of the model
    scores = model.evaluate_generator(validation, verbose = 0)
    print("Baseline error : %.2f%%" %(100-scores[1]*100))
    return


# Parameters
params = {'dim': (8,1),
          'batch_size': 2,
          'n_classes': 2,
          'shuffle': True}

Xraw = np.array([
                [1,1,1,1,1,1,1,1],
                [2,2,2,2,2,2,2,2],
                [3,3,3,3,3,3,3,3],
                [4,4,4,4,4,4,4,4],
                [5,5,5,5,5,5,5,5],

                ])
Yraw = np.array([[0],[1],[1],[0],[0]])

partition, labels = partitionize(Xraw,Yraw, ratio = (3,1))

training_generator = DataGenerator(partition["train"], labels, **params)
validation_generator = DataGenerator(partition["validation"], labels, **params)

model = model_v0_trial(8, 1)

train_model_batch(training = training_generator, validation = validation_generator, model=model, epochs=10, folder="")

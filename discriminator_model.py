# references:
# https://gist.github.com/neilslater/40201a6c63b4462e6c6e458bab60d0b4
# https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.



import numpy as np
import os
import pandas as pd
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")
    
def load_data(datasetNumber=1,numToLoad=0):
    prepend = 'discrimData'
    trainDataName = ''
    if datasetNumber == 1:
        trainDataName = prepend+'_train.h5' 
    else:
        trainDataName = prepend+'_'+str(datasetNumber)+'_train.h5'
    print('loading file name: ' + trainDataName)
    trainData = h5py.File(trainDataName, 'r')
    if numToLoad == 0:
        y_train = trainData['/status'][()]
        X_train = trainData['/imgData'][()]
    else:
        y_train = trainData['/status'][0:numToLoad]
        X_train = trainData['/imgData'][:,:,0:numToLoad]
    prop_length = y_train.shape[0]
    X_train = X_train[:,:,0:prop_length]
    X_train = np.swapaxes(X_train, 0,2)
    X_train = np.expand_dims(X_train, axis=3)
    print('Read images')
    return X_train, y_train

def read_and_normalize_data(m_x=None, s_x=None, datasetNumber=1,numToLoad=0, devPortion = 0.1): #, m_y=None, s_y=None
    train_data, train_target = load_data(datasetNumber=datasetNumber, numToLoad=numToLoad)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    if m_x is not None:
        pass
    else:
        m_x = train_data.mean()
    if s_x is not None:
        pass
    else:
        s_x = train_data.std()
    print ('Train mean, sd:', m_x, s_x)
    train_data -= m_x
    train_data /= s_x
    dev_index = []
    train_index = []
    for ii in range(train_target.shape[0]):
        if ii % int(1/devPortion) == 0:
            dev_index.append(ii)
        else:
             train_index.append(ii)
    dev_data = train_data[dev_index,:,:,:]
    dev_target = train_target[dev_index]
    train_data = train_data[train_index,:,:,:]
    train_target = train_target[train_index]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, dev_data, dev_target, m_x, s_x

def create_model():
    par_kernel_size = 5
    step_stride = (2,2)
    l2regval = 1e-3*1*0

    model = Sequential()
    model.add(Flatten())

    model.add(Dense(3, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(2, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(2, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1))
    model.add(Activation(activation="sigmoid"))
    opt = SGD(lr=0.001, momentum=0.0) # Adam(lr=0.001*2, beta_1=0.9, beta_2=0.99)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 

    return model
def train_model(batch_size = 10, nb_epoch = 20, train_loops=1):

    X_train, y_train, X_valid, y_valid, m_x, s_x = read_and_normalize_data(numToLoad=0) #, m_y, s_y
    append = '_discriminator_7'
    model = create_model()
    with open('mvalues'+append+'.txt', 'w') as f:
        f.write("m_x: %f, s_x: %f" % (m_x, s_x) ) # , m_y: %f, s_y: %f  , m_y, s_y



    # define the checkpoint
    filepath = "model"+append+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    accuracyHistory = []
    valAccuracyHistory = []
    lossHistory = []
    valLossHistory = []
    for ii in range(train_loops):
        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=(nb_epoch+ii*nb_epoch), verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list, shuffle=True, initial_epoch=(ii*nb_epoch) )
        print("model history: ", history.history)
        lossHistory.extend(history.history['loss'])
        valLossHistory.extend(history.history['val_loss'])
        accuracyHistory.extend(history.history['acc'])
        valAccuracyHistory.extend(history.history['val_acc'])
        plt.figure()
        plt.plot(accuracyHistory,label='train')
        plt.plot(valAccuracyHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('mse'+append+'.pdf')
        plt.close()
        plt.plot(lossHistory,label='train')
        plt.plot(valLossHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('loss'+append+'.pdf')
        plt.close()

    predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    compare = pd.DataFrame(data={'original':y_valid.reshape((y_valid.shape[0],)),
            'prediction':predictions_valid.reshape((y_valid.shape[0],))})
    compare.to_csv('compare'+append+'.csv')

    return model


train_model(batch_size = 200, nb_epoch = 5, train_loops=25)
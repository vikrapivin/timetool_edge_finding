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
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("TkAgg")

def full_load_data(datasetNumber=833200,numToLoad=0, extName = '_train', inDirectory='/mnt/data1/viktor/Downloads/cs230pj/runs_dl/disc_transfer/'):
    dataName = ''
    dataName = str(datasetNumber) + extName +'.h5'
    print('loading file name: ' + dataName)
    trainData = h5py.File(inDirectory+dataName, 'r')
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
    print('Read data images')
    return X_train, y_train

def full_read_and_normalize_data(m_x, s_x, datasetNumber=833200, numToLoad=0, extName = '_train'):
    data_data, data_target = full_load_data(datasetNumber=datasetNumber, extName = extName, numToLoad=numToLoad)
    data_data = np.array(data_data, dtype=np.float32)
    data_target = np.array(data_target, dtype=np.float32)

    print ('Data data mean, sd:', m_x, s_x)
    data_data -= m_x
    data_data /= s_x
    print('Data shape:', data_data.shape)
    print(data_data.shape[0], ' data samples')
    return data_data, data_target

def create_model():
    l2regval = 1e-3*1*0

    model = Sequential()
    model.add(Flatten(input_shape=(1920, 110, 1)))

    model.add(Dense(3, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(10, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(10, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1))
    model.add(Activation(activation="sigmoid"))
    opt = SGD(lr=0.001, momentum=0.0) # Adam(lr=0.001*2, beta_1=0.9, beta_2=0.99)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 

    return model

def train_model(batch_size = 10, nb_epoch = 20, train_loops=1, runsToTrainOn = []):
    # num_samples = 1999
    # cv_size = 499
    countNum = None
    initCountNum = 500
    try:
        countNumFile = open("countNum.txt", 'r+')
        countNum = int(countNumFile.read()) + 1
        countNumFile.seek(0)
        countNumFile.write(str(countNum))
        countNumFile.truncate()
    except FileNotFoundError:
        countNumFile = open("countNum.txt", 'x')
        countNum = initCountNum
        countNumFile.seek(0)
        countNumFile.write(str(countNum))
        countNumFile.truncate()
    except:
        countNumFile = open("countNum.txt", 'r+')
        countNum = initCountNum
        countNumFile.seek(0)
        countNumFile.write(str(countNum))
        countNumFile.truncate()
    # model = load_model("model_discriminator_7.h5")
    model = create_model()
    old_model = load_model("model_discriminator_7.h5")
    # for ii in model.layers:
    #     ii.trainable = False
    # model.layers[3].trainable = True
    # model.layers[5].trainable = True
    # model.layers[7].trainable = True
    model.layers[1].set_weights(old_model.layers[1].get_weights())
    model.layers[1].trainable = False
    opt = Adam(lr=0.001*2, beta_1=0.9, beta_2=0.99)#SGD(lr=0.001, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 

    # X_train, y_train, m_x, s_x, m_y, s_y = read_and_normalize_train_data(numToLoad=0,loadBackground=True)
    m_x, s_x = 1930.602783, 631.627380
    X_train, y_train = full_read_and_normalize_data(m_x, s_x, datasetNumber=runsToTrainOn[len(runsToTrainOn)-1], numToLoad=0, extName = '_train')
    X_valid, y_valid = full_read_and_normalize_data(m_x, s_x, datasetNumber=runsToTrainOn[len(runsToTrainOn)-1], numToLoad=0, extName = '_dev')
    for ii in runsToTrainOn[1:len(runsToTrainOn)]:
        X_train_2, y_train_2 = full_read_and_normalize_data(m_x, s_x, datasetNumber=ii, numToLoad=0, extName = '_train')
        X_valid_2, y_valid_2 = full_read_and_normalize_data(m_x, s_x, datasetNumber=ii, numToLoad=0, extName = '_dev')
        X_train = np.concatenate((X_train,X_train_2),axis=0)
        y_train = np.concatenate((y_train,y_train_2),axis=0)
        X_valid = np.concatenate((X_valid,X_valid_2),axis=0)
        y_valid = np.concatenate((y_valid,y_valid_2),axis=0)
        X_train_2 = None
        y_train_2 = None
        y_valid_2 = None
        X_valid_2 = None


    append = '_disc_transfer_' + str(countNum) + '_layers_3_5_7_trainable'
    with open('mvalues'+append+'.txt', 'w') as f:
        f.write("m_x: %f, s_x: %f" % (m_x, s_x) )



    # define the checkpoint
    filepath = "model"+append+"_val_acc.h5"
    filepath2 = "model"+append+"_acc.h5"
    filepath3 = "model"+append+"_loss.h5"
    filepath4 = "model"+append+"_val_loss.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint(filepath2, monitor='acc', verbose=2, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='accuracy', verbose=2, save_best_only=True, mode='max')
    checkpoint3 = ModelCheckpoint(filepath3, monitor='loss', verbose=2, save_best_only=True, mode='min')
    checkpoint4 = ModelCheckpoint(filepath4, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,checkpoint2,checkpoint3, checkpoint4]
    accuracyHistory = []
    valAccuracyHistory = []
    lossHistory = []
    valLossHistory = []
    for ii in range(train_loops):
        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=(nb_epoch+ii*nb_epoch), verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list, shuffle=True, initial_epoch=(ii*nb_epoch) )
        print("model history: ", history.history)
        lossHistory.extend(history.history['loss'])
        valLossHistory.extend(history.history['val_loss'])
        # accuracyHistory.extend(history.history['acc'])
        # valAccuracyHistory.extend(history.history['val_acc'])
        accuracyHistory.extend(history.history['accuracy'])
        valAccuracyHistory.extend(history.history['val_accuracy'])
        plt.figure()
        plt.plot(accuracyHistory,label='train')
        plt.plot(valAccuracyHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('accuracy'+append+'.pdf')
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

runsToTrainWith = ['833205', '833220', '833240', '833264']

train_model(batch_size = 100, nb_epoch = 5, train_loops=200, runsToTrainOn = runsToTrainWith)
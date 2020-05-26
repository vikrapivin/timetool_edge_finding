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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")
    
def load_train(datasetNumber=1,numToLoad=0,loadBackground=False):
    trainDataName = ''
    addBkgr = ''
    if loadBackground is True:
        addBkgr = '_bkgrDiv'
    if datasetNumber == 1:
        trainDataName = 'runName'+addBkgr+'_train.h5' 
    else:
        trainDataName = 'runName_'+str(datasetNumber)+addBkgr+'_train.h5'
    print('loading file name: ' + trainDataName)
    trainData = h5py.File(trainDataName, 'r')
    if numToLoad == 0:
        y_train = trainData['/tagData'][()]
        X_train = trainData['/imgData'][()]
    else:
        y_train = trainData['/tagData'][0:numToLoad]
        X_train = trainData['/imgData'][:,:,0:numToLoad]
    prop_length = y_train.shape[0]
    X_train = X_train[:,:,0:prop_length]
    X_train = np.swapaxes(X_train, 0,2)
    X_train = np.expand_dims(X_train, axis=3)
    print('Read train images')
    return X_train, y_train

def load_dev(datasetNumber=1,numToLoad=0,loadBackground=False):
    devDataName = ''
    addBkgr = ''
    if loadBackground is True:
        addBkgr = '_bkgrDiv'
    if datasetNumber == 1:
        devDataName = 'runName'+addBkgr+'_dev.h5'
    else:
        devDataName = 'runName_'+str(datasetNumber)+addBkgr+'_dev.h5'
    print('loading file name: ' + devDataName)
    devData = h5py.File(devDataName, 'r')
    if numToLoad == 0:
        y_dev = devData['/tagData'][()]
        X_dev = devData['/imgData'][()]
    else:
        y_dev = devData['/tagData'][0:numToLoad]
        X_dev = devData['/imgData'][:,:,0:numToLoad]
    prop_length = y_dev.shape[0]
    X_dev = X_dev[:,:,0:prop_length]
    X_dev = np.swapaxes(X_dev, 0,2)
    X_dev = np.expand_dims(X_dev, axis=3)
    print('Read dev images')
    return X_dev, y_dev

def read_and_normalize_train_data(m_x=None, s_x=None, m_y=None, s_y=None, datasetNumber=1,numToLoad=0,loadBackground=False):
    train_data, train_target = load_train(datasetNumber=datasetNumber, numToLoad=numToLoad, loadBackground=loadBackground)
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
    if m_y is not None:
        pass
    else:
        m_y = train_target.mean()
    if s_y is not None:
        pass
    else:
        s_y = train_target.std()

    print ('Train mean, sd:', m_x, s_x)
    print ('Target mean, sd:', m_y, s_y )
    train_data -= m_x
    train_data /= s_x
    train_target -= m_y
    train_target /= s_y
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, m_x, s_x, m_y, s_y

def read_and_normalize_dev_data(m_x, s_x, m_y, s_y, datasetNumber=1,numToLoad=0,loadBackground=False):
    dev_data, dev_target = load_dev(datasetNumber=datasetNumber, numToLoad=numToLoad, loadBackground=loadBackground)
    dev_data = np.array(dev_data, dtype=np.float32)
    dev_target = np.array(dev_target, dtype=np.float32)

    print ('Train mean, sd:', m_x, s_x)
    print ('Target mean, sd:', m_y, s_y )
    dev_data -= m_x
    dev_data /= s_x
    dev_target -= m_y
    dev_target /= s_y
    print('Dev shape:', dev_data.shape)
    print(dev_data.shape[0], 'dev samples')
    return dev_data, dev_target

def create_model():
    nb_filters = 8
    par_kernel_size = 5
    step_stride = (2,2)
    l2regval = 1e-3*1
    # nb_conv = 5

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=par_kernel_size, strides=step_stride, input_shape=(1920, 110, 1), padding="same", kernel_regularizer=regularizers.l2(l2regval),
    bias_regularizer=regularizers.l2(l2regval) )) # kernel_regularizer=l1(0.001),
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(nb_filters, kernel_size=par_kernel_size, strides=step_stride, padding="same", kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(nb_filters, kernel_size=par_kernel_size, strides=step_stride, padding="same", kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Conv2D(nb_filters, kernel_size=par_kernel_size, strides=step_stride, padding="same", kernel_regularizer=regularizers.l2( l2regval),
    # bias_regularizer=regularizers.l2(l2regval)))
    # model.add(LeakyReLU(alpha=0.1))
    # # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Conv2D(nb_filters, kernel_size=par_kernel_size, strides=step_stride, padding="same", kernel_regularizer=regularizers.l2( l2regval),
    # bias_regularizer=regularizers.l2(l2regval)))
    # model.add(LeakyReLU(alpha=0.1))
    # # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization(momentum=0.8))

    # model.add(Convolution2D(nb_filters, kernel_size=3,
    #                         padding='same',
    #                         input_shape=(1920, 110) ) ) #border_mode='valid'
    # model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    #model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    #model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    # model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(128, kernel_regularizer=regularizers.l2( l2regval),
    # bias_regularizer=regularizers.l2(l2regval)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(64, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(16, kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.05))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001*2, beta_1=0.9, beta_2=0.99), metrics=['mse', 'mae', 'mape', 'cosine']) #Adadelta())
    # , beta_1 = 0.8
    return model
def train_model(batch_size = 10, nb_epoch = 20, train_loops=1):
    # num_samples = 1999
    # cv_size = 499

    X_train, y_train, m_x, s_x, m_y, s_y = read_and_normalize_train_data(numToLoad=0,loadBackground=True)
    X_valid, y_valid = read_and_normalize_dev_data(m_x, s_x, m_y, s_y,numToLoad=0,loadBackground=True)
    X_train_2, y_train_2, m_x, s_x, m_y, s_y = read_and_normalize_train_data(m_x=m_x, s_x=s_x, m_y=m_y, s_y=s_y, datasetNumber=2, numToLoad=0,loadBackground=True)
    X_valid_2, y_valid_2 = read_and_normalize_dev_data(m_x, s_x, m_y, s_y, datasetNumber=2,numToLoad=0,loadBackground=True)
    X_train = np.concatenate((X_train,X_train_2),axis=0)
    y_train = np.concatenate((y_train,y_train_2),axis=0)
    X_valid = np.concatenate((X_valid,X_valid_2),axis=0)
    y_valid = np.concatenate((y_valid,y_valid_2),axis=0)
    X_train_2 = None
    y_train_2 = None
    y_valid_2 = None
    X_valid_2 = None
    append = '_small_7filt_5kernel_22stride_1en3_1reg_3cnn_2dense_wBkgr_9'
    model = create_model()
    with open('mvalues'+append+'.txt', 'w') as f:
        f.write("m_x: %f, s_x: %f, m_y: %f, s_y: %f" % (m_x, s_x, m_y, s_y) )



    # define the checkpoint
    filepath = "model"+append+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    lossHistory = []
    valLossHistory = []
    mseHistory = []
    valmseHistory = []
    for ii in range(train_loops):
        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=(nb_epoch+ii*nb_epoch), verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list, shuffle=True, initial_epoch=(ii*nb_epoch) )
        print("model history: ", history.history)
        lossHistory.extend(history.history['loss'])
        valLossHistory.extend(history.history['val_loss'])
        mseHistory.extend(history.history['mean_squared_error'])
        valmseHistory.extend(history.history['val_mean_squared_error'])
        #print(history.history.keys())
        #dict_keys(['val_loss', 'val_mse', 'val_mae', 'val_mape', 'val_cosine', 'loss', 'mse', 'mae', 'mape', 'cosine'])
        plt.figure()
        plt.plot(mseHistory,label='train')
        plt.plot(valmseHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('mse'+append+'.pdf')
        plt.close()
        plt.figure()
        plt.plot(lossHistory,label='train')
        plt.plot(valLossHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('loss'+append+'.pdf')
        plt.close()
        # print(lossHistory)

    predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    compare = pd.DataFrame(data={'original':y_valid.reshape((y_valid.shape[0],)),
            'prediction':predictions_valid.reshape((y_valid.shape[0],))})
    compare.to_csv('compare'+append+'.csv')

    return model


train_model(batch_size = 100, nb_epoch = 5, train_loops=25)
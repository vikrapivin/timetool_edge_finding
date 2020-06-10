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
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# lr_schedule = ExponentialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=10000,
#     decay_rate=0.9)
import matplotlib.pyplot as plt
import matplotlib
import random
from keras.models import load_model
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

def full_load_data(datasetNumber=833200,numToLoad=0,loadBackground=False,inDirectory='/mnt/data1/viktor/Downloads/cs230pj/final_model_1/'):
    dataName = ''
    addBkgr = ''
    if loadBackground is True:
        addBkgr = 'bkgrDiv_'
    dataName = 'runName_'+ addBkgr + str(datasetNumber)+'_train.h5'
    print('loading file name: ' + dataName)
    trainData = h5py.File(inDirectory+dataName, 'r')
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
    print('Read data images')
    return X_train, y_train

def full_read_and_normalize_data(m_x, s_x, m_y, s_y, datasetNumber=833200, numToLoad=0, loadBackground=True):
    data_data, data_target = full_load_data(datasetNumber=datasetNumber, numToLoad=numToLoad, loadBackground=loadBackground)
    data_data = np.array(data_data, dtype=np.float32)
    data_target = np.array(data_target, dtype=np.float32)

    print ('Data data mean, sd:', m_x, s_x)
    print ('Data target mean, sd:', m_y, s_y )
    data_data -= m_x
    data_data /= s_x
    data_target -= m_y
    data_target /= s_y
    print('Data shape:', data_data.shape)
    print(data_data.shape[0], ' data samples')
    return data_data, data_target

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

def create_model(nb_filters = (8, 16, 32),par_kernel_size = 5, step_stride = (2,2),l2regval = 1e-3*1, denseParams = (64, 16), optParams =(0.001, 0.9, 0.9, 0.00001) ):
    # nb_filters = 8
    # par_kernel_size = 5
    # step_stride = (2,2)
    # l2regval = 1e-3*1
    # nb_conv = 5

    model = Sequential()
    model.add(Conv2D(nb_filters[0], kernel_size=par_kernel_size, strides=step_stride[0], input_shape=(1920, 110, 1), padding="same", kernel_regularizer=regularizers.l2(l2regval),
    bias_regularizer=regularizers.l2(l2regval) )) # kernel_regularizer=l1(0.001),
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(nb_filters[1], kernel_size=par_kernel_size, strides=step_stride[1], padding="same", kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))


    model.add(Conv2D(nb_filters[2], kernel_size=par_kernel_size, strides=step_stride[2], padding="same", kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(nb_filters[3], kernel_size=par_kernel_size, strides=step_stride[3], padding="same", kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))
    model.add(Dropout(0.1))
    

    model.add(Flatten())

    model.add(Dense(denseParams[0], kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval)))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))

    model.add(Dense(denseParams[1], kernel_regularizer=regularizers.l2( l2regval),
    bias_regularizer=regularizers.l2(l2regval/10)))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.1))

    # model.add(Dense(denseParams[2], kernel_regularizer=regularizers.l2( l2regval),
    # bias_regularizer=regularizers.l2(l2regval)))
    # # model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.2))

    # model.add(Dense(denseParams[3], kernel_regularizer=regularizers.l2( l2regval),
    # bias_regularizer=regularizers.l2(l2regval)))
    # # model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.1))
    # model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=optParams[0], beta_1=optParams[1], beta_2=optParams[2], decay=optParams[3]), metrics=['mse', 'mae', 'mape', 'cosine']) #Adadelta())
    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=optParams[0], beta_1=optParams[1], beta_2=optParams[2], decay=optParams[3]), metrics=['mse', 'mae', 'mape', 'cosine']) #Adadelta())
    # , beta_1 = 0.8
    return model
def train_model(batch_size = 10, nb_epoch = 20, train_loops=1, runsToTrainOn = []):
    # num_samples = 1999
    # cv_size = 499
    countNum = None
    initCountNum = 1000
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
    nb_filters = (13, 21, 33, 44) # (15, 15, 35, 50)
    par_kernel_size = 3
    # step_stride = [(2,2), (1,1), (1,1), (1,1)]
    step_stride = [(2,2), (2,2), (2,2), (2,2)]
    l2regval = 5e-6
    denseParams = (40, 10)
    optParams =(0.01, 0.9, 0.99, 0.0001) 
    print('training with parameters\nbatch_size: ' + str(batch_size) + '\nnb_epoch: ' + str(nb_epoch) + '\ntrain_loops: ' + str(train_loops) + '\nrunToTrainOn: ' + str(runsToTrainOn) + '\ncountNum: ' + str(countNum))
    print('nb_filters: '  + str(nb_filters))
    print('l2regval: '  + str(l2regval))
    print('par_kernel_size: '  + str(par_kernel_size))
    print('step_stride: '  + str(step_stride))
    print('denseParams: '  + str(denseParams))
    print('optParams: '  + str(optParams))
    model = create_model(nb_filters=nb_filters, par_kernel_size=par_kernel_size, step_stride=step_stride, l2regval=l2regval, denseParams=denseParams, optParams=optParams)
    
    model.summary()
    # old_model = load_model("/mnt/data1/viktor/Downloads/cs230pj/final_model_1/model_full_data_mae_60_200_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_mean_absolute_error.h5")
    old_model = load_model("/mnt/data1/viktor/Downloads/cs230pj/final_model_1/model_full_data_mae_60_200_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_mean_squared_error.h5")
    
    numbersToFreeze = [0, 1, 4, 5, 8, 9, 12, 13, 17, 18]
    for ii in numbersToFreeze:
        print(ii)
        model.layers[ii].set_weights(old_model.layers[ii].get_weights())
        model.layers[ii].trainable = False
    # opt = SGD(lr=0.001, momentum=0.9) 
    opt = Adam(lr=optParams[0], beta_1=optParams[1], beta_2=optParams[2], decay=optParams[3])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae', 'mape', 'cosine'])  # mean_absolute_error



    # X_train, y_train, m_x, s_x, m_y, s_y = read_and_normalize_train_data(numToLoad=0,loadBackground=True)
    m_x, s_x = (0.95750654, 0.114258304)
    m_y, s_y = (828.119, 92.08046)
    X_train, y_train = full_read_and_normalize_data(m_x, s_x, m_y, s_y, datasetNumber=runsToTrainOn[len(runsToTrainOn)-1], numToLoad=0, loadBackground=True)
    X_valid, y_valid = read_and_normalize_dev_data(m_x, s_x, m_y, s_y, datasetNumber=runsToTrainOn[len(runsToTrainOn)-1],numToLoad=0,loadBackground=True)

    append = '_full_data_mae_'+ str(countNum) + '_' + str(batch_size) + '_bs_' + str(nb_filters[0]) + '_filt_' + str(par_kernel_size) + '_kernel_' + str(step_stride[0][0]) + str(step_stride[0][1]) + '_stride_' + str(l2regval).replace('.','_') + '_cnn_2dense_wBkgr'
    with open('mvalues'+append+'.txt', 'w') as f:
        f.write("m_x: %f, s_x: %f, m_y: %f, s_y: %f" % (m_x, s_x, m_y, s_y) )



    # define the checkpoint
    filepath = "model"+append+"_val_mean_squared_error.h5"
    filepath2 = "model"+append+"_val_loss.h5"
    filepath3 = "model"+append+"_loss.h5"
    filepath4 = "model"+append+"_val_mean_absolute_error.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    # checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=2, save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=2, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    checkpoint3 = ModelCheckpoint(filepath3, monitor='loss', verbose=2, save_best_only=True, mode='min')
    # checkpoint4 = ModelCheckpoint(filepath4, monitor='val_mean_absolute_error', verbose=2, save_best_only=True, mode='min')
    checkpoint4 = ModelCheckpoint(filepath4, monitor='val_mae', verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,checkpoint2,checkpoint3, checkpoint4]
    lossHistory = []
    valLossHistory = []
    mseHistory = []
    valmseHistory = []
    maeHistory = []
    valmaeHistory = []


    for ii in range(train_loops):
        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=(nb_epoch+ii*nb_epoch), verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list, shuffle=True, initial_epoch=(ii*nb_epoch) )
        print("model history: ", history.history)
        lossHistory.extend(history.history['loss'])
        valLossHistory.extend(history.history['val_loss'])
        # mseHistory.extend(history.history['mean_squared_error'])
        # valmseHistory.extend(history.history['val_mean_squared_error'])
        mseHistory.extend(history.history['mse'])
        valmseHistory.extend(history.history['val_mse'])
        # maeHistory.extend(history.history['mean_absolute_error'])
        # valmaeHistory.extend(history.history['val_mean_absolute_error'])
        maeHistory.extend(history.history['mae'])
        valmaeHistory.extend(history.history['val_mae'])
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
        plt.figure()
        plt.plot(maeHistory,label='train')
        plt.plot(valmaeHistory,label='dev')
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.savefig('mae'+append+'.pdf')
        plt.close()
    
    # predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    # compare = pd.DataFrame(data={'original':y_valid.reshape((y_valid.shape[0],)),
    #         'prediction':predictions_valid.reshape((y_valid.shape[0],))})
    # compare.to_csv('compare'+append+'.csv')

    return model

runsToTrainWith = [123457]
train_model(batch_size = 20, nb_epoch = 2, train_loops=200000, runsToTrainOn = runsToTrainWith)
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
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


new_model = load_model("model_discriminator_ver_1.h5")
# try on test set
devData = h5py.File('runName'+'_test.h5', 'r')
# try on dev set by uncommenting
# devData = h5py.File('runName_2'+'_bkgrDiv'+'_dev.h5', 'r')
# isDataLabeled = True
# devData = h5py.File('noFit_bkgrDiv.h5', 'r')
# y_dev = devData['/tagData'][()]
# X_dev = devData['/imgData'][()]
y_dev = None
X_dev = devData['/imgData'][:,:,200:300]
# X_dev = devData['/imgData'][()]
X_dev = np.swapaxes(X_dev, 0,2)
X_dev = np.expand_dims(X_dev, axis=3)
print('Read dev images')
# values obtained when training the NN.
# Train mean, sd: 1930.6028 631.6274

m_x = 1930.6028
s_x = 631.6274
X_dev -= m_x
X_dev /= s_x
# predictions_valid = new_model.predict(X_dev, batch_size=50, verbose=1)
# print(predictions_valid.T)
# if isDataLabeled is True:
#     print(y_dev.T)

f = h5py.File('833228.h5', 'r')
dirOfF = list(f['run_833228/opal_1/'].keys())
# print(dirOfF)
dirOfF = dirOfF[1:]
lengthBack = len(dirOfF)
bkImg = np.zeros(f['run_833228/opal_1/'+ dirOfF[0] + '/detector_data'][70:180,:].shape)
for ii in range(lengthBack):
    #print(f['run_833228/opal_1/'+ dirOfF[ii] + '/detector_data'].shape)
    bkImg = bkImg*(ii) + f['run_833228/opal_1/'+ dirOfF[ii] + '/detector_data'][70:180,:]
    bkImg = bkImg/(ii +1)
print("Read background")
print(bkImg.shape)
bkImg = np.swapaxes(bkImg, 0,1)
print(bkImg.shape)
bkImg = np.expand_dims(bkImg, axis=0)
print(bkImg.shape)
bkImg = np.expand_dims(bkImg, axis=3)
print(bkImg.shape)
bkImg -= m_x
bkImg /= s_x
predictions_valid = new_model.predict(bkImg, batch_size=1, verbose=1)
print(predictions_valid)
startingVal = 0
endingVal = 730
X_dev[:,startingVal:endingVal,:,:] = bkImg[0,startingVal:endingVal,:,:]
predictions_valid = new_model.predict(X_dev, batch_size=50, verbose=1)
print(predictions_valid.T)
# for indexTest in range(300):
#     firstImg = X_dev[indexTest,:,:,:]
#     if isDataLabeled is True:
#         firstImgLine = y_dev[indexTest]
#     firstImgLine_predict = predictions_valid[indexTest]
#     firstImg *= s_x
#     firstImg += m_x
#     if isDataLabeled is True:
#         firstImgLine *= s_y
#         firstImgLine += m_y
#     firstImgLine_predict *= s_y
#     firstImgLine_predict += m_y

#     firstImg = np.squeeze(firstImg)
#     firstImg = firstImg.T
#     plt.figure()
#     ax1=plt.subplot(2, 1, 1)
#     ax2=plt.subplot(2, 1, 2)
#     ax1.imshow(firstImg,aspect=7)
#     # plt.imshow(firstImg/bkImg,aspect=7)
#     if isDataLabeled is True:
#         ax1.axvline(firstImgLine,linewidth=2, color='r')
#         print("Line fit from SACLA: " + str(firstImgLine))
#     ax1.axvline(firstImgLine_predict,linewidth=2, color='b')
#     print("Prediction by NN: " + str(firstImgLine_predict))
#     d_2dsum = np.sum(firstImg[50:80,:],axis=0)
#     print(firstImg.shape)
#     ax2.plot(d_2dsum)
#     if isDataLabeled is True:
#         ax2.axvline(firstImgLine,linewidth=1, color='r')
#     ax2.axvline(firstImgLine_predict,linewidth=1, color='b')
#     # fig = plt.gcf()
#     # fig.set_size_inches(20, 100)
#     plt.show()

# compare = pd.DataFrame(data={'prediction':predictions_valid.reshape((X_dev.shape[0],))})
# compare.to_csv('compare'+'testDisc'+'.csv')
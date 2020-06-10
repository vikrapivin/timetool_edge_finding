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
#import cv2
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


# new_model = load_model("model_disc_transfer_506_layers_3_5_7_trainable_val_loss.h5")
new_model = load_model("model_disc_transfer_2_520_layers_3_5_7_trainable_loss.h5")

new_model.summary()

loadStart = 100
loadEnd = loadStart+1000

# devData = h5py.File('../runName_bkgrDiv_833264_test.h5', 'r')
devData = h5py.File('test_TF_data_1_train.h5', 'r')
# devData = h5py.File('../runName_bkgrDiv_833190_NaN_List.h5', 'r') 
# runnum = '833240'
# devData = h5py.File(''+runnum+'_dev.h5', 'r')
isDataLabeled = True
if isDataLabeled is True:
    # y_dev = devData['/status'][loadStart:loadEnd]
    y_dev = devData['/status'][()]
# X_dev = devData['/imgData'][:,:,loadStart:loadEnd]
X_dev = devData['/imgData'][()]

# remove excess from file; not necessary for new h5 as i fixed this in my data creation
if isDataLabeled is True:
    prop_length = y_dev.shape[0]
    X_dev = X_dev[:,:,0:prop_length]

bkImg = np.load('../bkImg.npy')
X_dev *= bkImg[:,:,np.newaxis]
X_dev = np.swapaxes(X_dev, 0,2)
X_dev = np.expand_dims(X_dev, axis=3)
print('Read test images')


# values obtained when training the NN.
# Train mean, sd: 1920.6244 626.4308
# Target mean, sd: 827.39343 91.95158
m_x, s_x = 1930.602783, 631.627380
X_dev -= m_x
X_dev /= s_x

predictions_valid = new_model.predict(X_dev, batch_size=50, verbose=1)
# print(predictions_valid.T)
if isDataLabeled is True:
    print(y_dev.T)


# for indexTest in range(loadEnd-loadStart):
#     # if predictions_valid[indexTest] < 0.5:
#     #     continue
#     firstImg = X_dev[indexTest,:,:,:]
#     if isDataLabeled is True:
#         imgQual_truth = y_dev[indexTest]
#     imgQual_predict = predictions_valid[indexTest]
#     firstImg *= s_x
#     firstImg += m_x

#     firstImg = np.squeeze(firstImg)
#     firstImg = firstImg.T
#     plt.figure()
#     ax1=plt.subplot(4, 1, 1)
#     ax3=plt.subplot(4, 1, 2)
#     ax2=plt.subplot(4, 1, 3)
#     ax4=plt.subplot(4, 1, 4)
#     ax1.imshow(firstImg,aspect=7)
#     print("Prediction by NN: " + str(round(imgQual_predict[0])) + ' ' + str(imgQual_predict))
#     if isDataLabeled is True:
#         print("Ground Value: " + str(imgQual_truth))
#     ax3.imshow(firstImg/bkImg,aspect=7)
#     d_2dsum = np.sum(firstImg[50:80,:],axis=0)/31
#     print(firstImg.shape)
#     ax2.plot(d_2dsum)
#     bkdivimg = firstImg/bkImg
#     d_2dsum = np.sum(bkdivimg[50:80,:],axis=0)/31
#     ax4.plot(d_2dsum)
#     plt.show()

compare = pd.DataFrame(data={'original':y_dev.reshape((y_dev.shape[0],)),
        'prediction':predictions_valid.reshape((predictions_valid.shape[0],))})
# compare = pd.DataFrame(data={'prediction':predictions_valid.reshape((predictions_valid.shape[0],))})
# compare.to_csv('compare'+'_testmodel_'+runnum+'.csv')
compare.to_csv('compare'+'_testmodel'+'.csv')
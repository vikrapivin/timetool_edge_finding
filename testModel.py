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


# new_model = load_model("model_temp_larger_density_10_filters_batch_100_larger_reg_more_net_more_lr.h5")
# new_model = load_model("model_full_data_mae_1009_20_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_mean_absolute_error.h5")
# new_model = load_model("model_full_data_mae_1009_20_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_mean_squared_error.h5")
new_model = load_model("model_full_data_mae_1016_20_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_mean_squared_error.h5")
# new_model = load_model("model_full_data_mae_1009_20_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_val_loss.h5")
# new_model = load_model("model_full_data_mae_1009_20_bs_13_filt_3_kernel_22_stride_5e-06_cnn_2dense_wBkgr_loss.h5")
# try on test set
# devData = h5py.File('runName'+'_test.h5', 'r')
# try on dev set by uncommenting
# devData = h5py.File('runName_2'+'_bkgrDiv'+'_dev.h5', 'r')
isDataLabeled = True
# devData = h5py.File('test_data_test.h5', 'r')
devData = h5py.File('runName_123457_bkgrDiv_dev.h5', 'r')
# devData = h5py.File('test_edge_data_1_test.h5', 'r')
# y_dev = devData['/tagData'][()]
# X_dev = devData['/imgData'][()]
# y_dev = None
if isDataLabeled is True:
    # y_dev = devData['/tagData'][200:500]
    y_dev = devData['/tagData'][()]
# X_dev = devData['/imgData'][:,:,200:500]
X_dev = devData['/imgData'][()]
if isDataLabeled is True:
    prop_length = y_dev.shape[0]
    X_dev = X_dev[:,:,0:prop_length]
X_dev = np.swapaxes(X_dev, 0,2)
X_dev = np.expand_dims(X_dev, axis=3)
print('Read dev images')

m_x, s_x = (0.95750654, 0.114258304)
m_y, s_y = (828.119, 92.08046)
X_dev -= m_x
X_dev /= s_x
if isDataLabeled is True:
    y_dev -= m_y
    y_dev /= s_y
predictions_valid = new_model.predict(X_dev, batch_size=50, verbose=1)
# print(predictions_valid.T)
if isDataLabeled is True:
    print(y_dev.T)

bkImg = np.load('../bkImg.npy')

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
#     ax1=plt.subplot(3, 1, 1)
#     ax3=plt.subplot(3, 1, 2)
#     ax2=plt.subplot(3, 1, 3)
#     ax1.imshow(firstImg,aspect=7)
#     # plt.imshow(firstImg/bkImg,aspect=7)
#     if isDataLabeled is True:
#         ax1.axvline(firstImgLine,linewidth=2, color='r')
#         print("Line fit from SACLA: " + str(firstImgLine))
#     ax1.axvline(firstImgLine_predict,linewidth=2, color='b')
#     print("Prediction by NN: " + str(firstImgLine_predict))
#     ax3.imshow(firstImg*bkImg,aspect=7)
#     d_2dsum = np.sum(firstImg[50:80,:],axis=0)/31
#     print(firstImg.shape)
#     ax2.plot(d_2dsum)
#     if isDataLabeled is True:
#         ax2.axvline(firstImgLine,linewidth=1, color='r')
#     ax2.axvline(firstImgLine_predict,linewidth=1, color='b')
#     # fig = plt.gcf()
#     # fig.set_size_inches(20, 100)
#     plt.show()

compare = pd.DataFrame(data={'original':y_dev.reshape((y_dev.shape[0],)),
        'prediction':predictions_valid.reshape((y_dev.shape[0],))})
compare.to_csv('compare'+'testmodel'+'.csv')
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 20num_filter_scale1

@author: tmquan
"""

from __future__ import print_function
from Utility import *

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Deconvolution2D
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import regularizers

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=1, mode=2)(prev) # Specifying the axis and mode allows for later merging
    # prev = Activation('relu')(prev)
    prev = Convolution2D(feat_maps_out, 3, 3, activation='relu', border_mode='same')(prev) 
    prev = BatchNormalization(axis=1, mode=2)(prev) # Specifying the axis and mode allows for later merging
    # prev = Activation('relu')(prev)
    prev = Convolution2D(feat_maps_out, 3, 3, activation='relu', border_mode='same')(prev) 
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Convolution2D(feat_maps_out, 1, 1, activation='relu', border_mode='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection
	
def get_uplusnet():
	inputs = Input((640, 640, 1))
	nb_filters = 64

	conv1a = Convolution2D(nb_filters*1, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1a = Residual(nb_filters*1, nb_filters*1, conv1a)
	conv1a = Convolution2D(nb_filters*1, 3, 3, activation='relu', border_mode='same')(conv1a)
	conv1a = Dropout(p=0.25)(conv1a)
	pool1a = MaxPooling2D(pool_size=(2, 2))(conv1a)

	conv2a = Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same')(pool1a)
	conv2a = Residual(nb_filters*2, nb_filters*2, conv2a)
	conv2a = Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same')(conv2a) 
	conv2a = Dropout(p=0.25)(conv2a)
	pool2a = MaxPooling2D(pool_size=(2, 2))(conv2a)

	conv3a = Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same')(pool2a)
	conv3a = Residual(nb_filters*4, nb_filters*4, conv3a)
	conv3a = Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same')(conv3a)
	conv3a = Dropout(p=0.25)(conv3a)
	pool3a = MaxPooling2D(pool_size=(2, 2))(conv3a)

	conv4a = Convolution2D(nb_filters*8, 3, 3, activation='relu', border_mode='same')(pool3a)
	conv4a = Residual(nb_filters*8, nb_filters*8, conv4a)
	conv4a = Convolution2D(nb_filters*8, 3, 3, activation='relu', border_mode='same')(conv4a)
	conv4a = Dropout(p=0.25)(conv4a)
	pool4a = MaxPooling2D(pool_size=(2, 2))(conv4a)

	conv5a = Convolution2D(nb_filters*16, 3, 3, activation='relu', border_mode='same')(pool4a)
	conv5a = Residual(nb_filters*16, nb_filters*16, conv5a)
	conv5a = Dropout(p=0.25)(conv5a)
	conv5a = Convolution2D(nb_filters*16, 3, 3, activation='relu', border_mode='same')(conv5a)

	up6a   = Convolution2D(nb_filters*8, 3, 3, activation='relu', border_mode='same')(conv5a)
	# up6a   = UpSampling2D(size=(2, 2))(up6a)
	up6a   = Deconvolution2D(nb_filters*8, 3, 3, 
							 input_shape=(nb_filters*8, 40, 40),
							 output_shape=(1, 80, 80, nb_filters*8), 							 
							 subsample=(2, 2), 
							 activation='relu',
							 border_mode='same')(up6a)
	conv6a = merge([up6a, conv4a], mode='sum')
	conv6a = Dropout(p=0.25)(conv6a)
	conv6a = Residual(nb_filters*8, nb_filters*8, conv6a)
	conv6a = Convolution2D(nb_filters*8, 3, 3, activation='relu', border_mode='same')(conv6a)

	up7a   = Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same')(conv6a)
	# up7a   = UpSampling2D(size=(2, 2))(up7a)
	up7a   = Deconvolution2D(nb_filters*4, 3, 3, 
							 input_shape=(nb_filters*4, 80, 80),
							 output_shape=(1, 160, 160, nb_filters*4), 							 
							 subsample=(2, 2), 
							 activation='relu',
							 border_mode='same')(up7a)
	conv7a = merge([up7a, conv3a], mode='sum')
	conv7a = Dropout(p=0.25)(conv7a)
	conv7a = Residual(nb_filters*4, nb_filters*4, conv7a)
	conv7a = Convolution2D(nb_filters*4, 3, 3, activation='relu', border_mode='same')(conv7a)

	up8a   = Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same')(conv7a)
	# up8a   = UpSampling2D(size=(2, 2))(up8a)
	up8a   = Deconvolution2D(nb_filters*2, 3, 3, 
							 input_shape=(nb_filters*4, 160, 160),
							 output_shape=(1, 320, 320, nb_filters*2), 							 
							 subsample=(2, 2), 
							 activation='relu',
							 border_mode='same')(up8a)
	conv8a = merge([up8a, conv2a], mode='sum')
	conv8a = Dropout(p=0.25)(conv8a)
	conv8a = Residual(nb_filters*2, nb_filters*2, conv8a)
	conv8a = Convolution2D(nb_filters*2, 3, 3, activation='relu', border_mode='same')(conv8a)

	up9a   = Convolution2D(nb_filters*1, 3, 3, activation='relu', border_mode='same')(conv8a)
	# up9a   = UpSampling2D(size=(2, 2))(up9a)
	up9a   = Deconvolution2D(nb_filters*1, 3, 3, 
							 input_shape=(nb_filters*1, 320, 320),
							 output_shape=(1, 640, 640, nb_filters*1), 							 
							 subsample=(2, 2), 
							 activation='relu',
							 border_mode='same')(up9a)

	conv9a = merge([up9a, conv1a], mode='sum')
	conv9a = Dropout(p=0.25)(conv9a)
	conv9a = Residual(nb_filters*1, nb_filters*1, conv9a)
	conv9a = Convolution2D(nb_filters*1, 3, 3, activation='relu', border_mode='same')(conv9a)
	
	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9a) 
	
	model = Model(input=inputs, output=conv10)
	
	model.compile(optimizer=Nadam(lr=0.0001), loss='mae')
	
	model.summary()
	return model
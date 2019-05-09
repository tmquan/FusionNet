# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""


from Utility 	import *
from Model 		import *
from Augment  	import *
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
######################################################################################
def augment_data(X, y):
	progbar = Progbar(X.shape[0])
	for k in range(X.shape[0]):

		image = np.squeeze(X[k])
		label = np.squeeze(y[k])
		
		# print image.shape
		# image, label  = augment_img(image, label)
		image, label = doElastic(image, label, Verbose=False)
		# image, label = doSquareRotate(image, label)
		# image, label = doFlip(image, label)
		image, label = addNoise(image, label)
		# image, label = doGaussFilter(image, label)


		image = np.expand_dims(image, axis=-1)
		label = np.expand_dims(label, axis=-1)

		# Save back the pair of images
		X[k] = image
		y[k] = label

		progbar.add(1) 

	return X, y 
######################################################################################		
def get_model():
	model = get_uplusnet()
	return model
######################################################################################    
def train():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	
	
	X = np.reshape(X, (30, 512, 512, 1)); # tf dim_ordering
	y = np.reshape(y, (30, 512, 512, 1)); # tf dim_ordering
	
	
	# Extend radius
	radius = 64
	X = np.pad(X, ((0,0), (radius, radius), (radius, radius), (0,0)), 'reflect'); # tf dim_ordering
	y = np.pad(y, ((0,0), (radius, radius), (radius, radius), (0,0)), 'reflect'); # tf dim_ordering
	
	# Preprocess the label
	y = y/255

	y  = y.astype('float32')
	X  = X.astype('float32')
	
	# Merging 7 times
	X0 = X
	y0 = y
	
	X1 = X[::1,::1,::-1,::1]
	y1 = y[::1,::1,::-1,::1]
	
	X2 = X[::1,::-1,::1,::1]
	y2 = y[::1,::-1,::1,::1]
	
	X3 = np.transpose(X, (0, 2, 1, 3))
	y3 = np.transpose(y, (0, 2, 1, 3))
	
	X4 = scipy.ndimage.interpolation.rotate(X, angle=90, axes=(1,2))
	y4 = scipy.ndimage.interpolation.rotate(y, angle=90, axes=(1,2))
	
	X5 = scipy.ndimage.interpolation.rotate(X, angle=180, axes=(1,2))
	y5 = scipy.ndimage.interpolation.rotate(y, angle=180, axes=(1,2))
	
	X6 = scipy.ndimage.interpolation.rotate(X, angle=270, axes=(1,2))
	y6 = scipy.ndimage.interpolation.rotate(y, angle=270, axes=(1,2)) 
	
	X = np.concatenate((X0, X1, X2, X3, X4, X5, X6), axis=0);	
	y = np.concatenate((y0, y1, y2, y3, y4, y5, y6), axis=0);	
	##########
	print "Y median", np.median(y)
	print "X shape", X.shape
	print "X dtype", X.dtype
	print "Y shape", y.shape
	print "Y dtype", y.dtype
	
	
	nb_iter 		= 1001
	epochs_per_iter = 2 
	batch_size 		= 1
	
	
	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	
	model = get_uplusnet()
	model.reset_states()
	# model.load_weights("model_300.hdf5")
	# graph = to_graph(model, show_shape=True)
	# graph.write_png("model.png")
	plot(model, to_file='model.png', show_shapes=True)

	nb_folds 	= 3
	kfolds 		= KFold(len(y), nb_folds)

	# Perform cross validation on the data
	for iter in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(iter + 1, nb_iter))  
		print('-'*50) 
		
		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print('Augmenting data for training...')
			X_train, y_train = augment_data(X_train, y_train) # Data augmentation for training 
			X_valid, y_valid = augment_data(X_valid, y_valid) # Data augmentation for training 
			
			
			print "X_train", X_train.shape
			print "y_train", y_train.shape
			
			# Normalize
			# X_train = X_train/255.0
			# X_valid = X_valid/255.0
			from keras.preprocessing.image import ImageDataGenerator
			train_datagen = ImageDataGenerator(
				featurewise_center=False,
    			samplewise_center=True,
    			featurewise_std_normalization=False,
    			samplewise_std_normalization=True,
    			rescale=1/255.0)
			train_datagen.fit(X_train)
			valid_datagen = ImageDataGenerator(
				featurewise_center=False,
    			samplewise_center=True,
    			featurewise_std_normalization=False,
    			samplewise_std_normalization=True,
    			rescale=1/255.0)
			valid_datagen.fit(X_valid)

			# checkpoint the best model
			filepath		= "model-best.hdf5"
			checkpoint 		= ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
			
			
			callbacks_list 	= [TensorBoard(log_dir='/home/Pearl/quantm/isbi12_210_keras_uplus/logs/')]

			# datagen = ImageDataGenerator(
			# 	    samplewise_center=True,
			# 	    samplewise_std_normalization=True,
			# 	    zca_whitening=True)

			history = model.fit(X_train, y_train, 
							verbose=1, 
							shuffle=True, 
							nb_epoch=epochs_per_iter,
							batch_size=batch_size, 
							callbacks=callbacks_list,
							validation_data=(X_valid, y_valid)
							)
			# history = model.fit_generator(
			# 				train_datagen.flow(X_train, y_train, batch_size=batch_size),
			# 				verbose=1, 
			# 				samples_per_epoch=len(X_train),
			# 				nb_epoch=epochs_per_iter,
			# 				callbacks=callbacks_list,
			# 				validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size),
			# 				nb_val_samples=len(X_valid)
			# 				)
			# list all data in history
			# print(history.history.keys())
			# Plot
			# # summarize history for accuracy
			# plt.plot(history.history['acc'])
			# plt.plot(history.history['val_acc'])
			# plt.title('model accuracy')
			# plt.ylabel('accuracy')
			# plt.xlabel('epoch')
			# plt.legend(['train', 'test'], loc='upper left')
			# plt.show()
			# summarize history for loss
			# plt.plot(history.history['loss'])
			# plt.plot(history.history['val_loss'])
			# plt.title('model loss')
			# plt.ylabel('loss')
			# plt.xlabel('epoch')
			# plt.legend(['train', 'test'], loc='upper left')
			# plt.show()
		if iter%10==0:
			fname = 'models/model_%03d.hdf5' %(iter)
			model.save_weights(fname, overwrite=True)
		
if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="2"
	os.environ["KERAS_BACKEND"]="tensorflow"
	# from keras import backend
	from tensorflow.python.client import device_lib
	train()
	
	

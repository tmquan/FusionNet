from Utility import *
from Model import *
	
def deploy():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	print "X.shape", X.shape
	print "y.shape", y.shape
	
	X = np.expand_dims(X, axis=1)
	y = np.expand_dims(y, axis=1)
	# Extend radius
	radius = 64
	X = np.pad(X, ((0,0), (0,0), (radius, radius), (radius, radius)), 'reflect');
	y = np.pad(y, ((0,0), (0,0), (radius, radius), (radius, radius)), 'reflect');
	
	X = np.transpose(X, (0, 2, 3, 1))
	y = np.transpose(y, (0, 2, 3, 1))
	
	
	print "X.shape", X.shape
	print "y.shape", y.shape
	
	X_deploy = X
	
	print "X_deploy.shape", X_deploy.shape
	
	
	# Load model
	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model = get_uplusnet()
	model.load_weights("models/model_120.hdf5")
	
	print('Predicting on data...')
	pred_recon  = model.predict(X_deploy,  batch_size=1, verbose=1)
	
    
	#print pred_recon
	#print X_deploy
	print "pred_recon.shape", pred_recon.shape
	# pred_recon  = np.reshape(pred_recon, (30,1,640, 640))
	pred_recon  = np.array(pred_recon[:,:,:,0])
	pred_recon = pred_recon[...,radius:-radius, radius:-radius]
	# pred_recon = np.transpose(pred_recon, (0, 3, 1, 2))
	skimage.io.imsave('result_train.tif', pred_recon)
	# # Post processing
	# strel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	# strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	# strel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# pred_recon  *= 255
	# pred_recon  = pred_recon.astype(np.uint8)
	
	# pred_recon  = cv2.erode(pred_recon,strel,iterations = 1)
	# pred_recon  = cv2.morphologyEx(pred_recon, cv2.MORPH_OPEN, strel,iterations = 2)
	
	# plt.imshow(np.hstack( (
							# np.squeeze(X_deploy), 
							# # np.squeeze(y_deploy), 
							# #np.squeeze(y_deploy),
							# 255*(pred_recon),
							
						  # ), 
						  # ) , cmap = plt.get_cmap('gray'))
	# plt.show()	
	
	# # Save stack
	# X_deploy = X
	# pred_recon  = model_recon.predict(X_deploy, num_batch=None)
 	# pred_recon  *= 255
	# pred_recon  = pred_recon.astype(np.uint8)
	# for k in range(pred_recon.shape[0]):
		# tmp = np.squeeze(pred_recon[k,:,:,:])
		# tmp = cv2.erode(tmp,strel,iterations = 1)
		# tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, strel,iterations = 2)
		# pred_recon[k,:,:,:] = tmp
	# arr2img('deploy.tif', pred_recon)
	
if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
	os.environ["KERAS_BACKEND"]="tensorflow"
	deploy()

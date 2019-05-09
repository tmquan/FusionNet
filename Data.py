from Utility import *
from Augment import *

train_volume_file = "data/train-inputs.tif"
train_labels_file = "data/train-labels.tif"
test_volume_file  = "data/test-inputs.tif"

def img2arr(imageFile):
	"""
	Read images tif files and store as np array
	"""
	img = skimage.io.imread(imageFile)
	print "File name: ", imageFile
	print "Shape    : ", img.shape
	return img

def data():
	# train_P
	# train_image = img2arr(train_volume_file)
	# train_label = img2arr(train_labels_file)
	# test_image  = img2arr(test_volume_file)
	
	# np.save("X_train.npy", train_image)
	# np.save("y_train.npy", train_label)
	# np.save("X_test.npy",  test_image)
	
	# train_volume_file = "data/train_256x256x128.tif"
	# train_labels_file = "data/segm_256x256x128.tif"	
	# train_volume_file = "data/train_512x512x128.tif"
	# train_labels_file = "data/segm_512x512x128.tif"
	
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)
	test_image = img2arr(test_volume_file)
	
	np.save("X_train.npy", train_image)
	np.save("y_train.npy", train_label)
	np.save("X_test.npy", test_image)
	
if __name__ == '__main__': 
	data()
	# train_image = img2arr(train_volume_file)
	# train_label = img2arr(train_labels_file)

	# # Test this image
	# image = train_image[0,:,:]
	# label = train_label[0,:,:]
	# image, label = elastic(image, label, Verbose=True)
#from __future__ import print_function

# import mxnet as mx

import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import skimage.measure

import pydot
import cv2
import sys
import logging
import os
import dicom
import scipy
import re 
import natsort
import time
import csv
import random
import subprocess
from matplotlib.pyplot 				import ion
from matplotlib.pyplot 				import cm
from sklearn.cross_validation 		import KFold # For cross_validation
from sklearn.metrics 				import mean_squared_error
from skimage.restoration 			import denoise_tv_chambolle
from skimage 						import exposure
from skimage 						import data, img_as_float
from scipy.misc 					import imresize
from scipy 							import ndimage
from scipy.ndimage.interpolation 	import map_coordinates
from scipy.stats 					import norm
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils.generic_utils 		import Progbar
from multiprocessing 				import Process
from joblib 						import Parallel, delayed
from random 						import randint
from graphviz 						import Digraph 
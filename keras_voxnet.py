from __future__ import print_function
import numpy as np
import tarfile
import cStringIO as StringIO
import zlib
import os
from random import shuffle

from keras.models import Sequential
from keras.optimizers import adagrad,adadelta, adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend as K

DATA_DIR = "./norm/"

def one_hot(labels,n_classes=2):
	new_labels = np.zeros((len(labels),n_classes))
	new_labels[np.arange(len(labels)),list(labels)]=1 
	return new_labels

def batch_load(srcfile):
	arr = np.load(srcfile)
	X_train,Y_train = arr["images"],arr["labels"]
	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3],1)
	Y_train = one_hot(Y_train)
	return X_train,Y_train

def set_model(nb_filters,input_shape,n_classes):
	# Kernel Size
	kernel_size0 = (7,7,7)
	kernel_size1 = (5,5,5)
	kernel_size2 = (3,3,3)

	# Max Pooling Size
	pool_size = (2,2,2)

	# Define Model
	model = Sequential()
	print("Input: "+str(input_shape))

	# Convolutional Added Layer 1
	model.add(Convolution3D(8,kernel_size0[0],kernel_size0[1],kernel_size0[2],
		border_mode='same',input_shape=input_shape,subsample=(2,2,2)))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	print("Conv1: "+str(model.layers[-1].output_shape))

	# Convolutional Layer 1
	model.add(Convolution3D(16,kernel_size1[0],kernel_size1[1],kernel_size1[2]))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	print("Conv2: "+str(model.layers[-1].output_shape))

	# Convolutional Layer 2
	model.add(Convolution3D(32,kernel_size2[0],kernel_size2[1],kernel_size2[2]))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add(Dropout(0.2))
	print("Conv4: "+str(model.layers[-1].output_shape))

	# Convolutional Layer Added 2
	model.add(Convolution3D(32,kernel_size2[0],kernel_size2[1],kernel_size2[2]))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add(Dropout(0.3))
	print("Conv5: "+str(model.layers[-1].output_shape))

	# Fully Connected Layer
	model.add(Flatten())
	model.add(Dense(128,W_regularizer=l2(0.01)))
	print("Full Connected 1: "+str(model.layers[-1].output_shape))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	model.add(Dense(n_classes,W_regularizer=l2(0.01)))
	#model.add(BatchNormalization())
	model.add(Activation('softmax'))

	# Compile
	#optimizer = adagrad(lr=0.001)
	optimizer = adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])
	return model


if __name__=="__main__":
	# MODEL INPUTS
	n_classes = 2
	n_channels = 1
	batch_size = 20
	n_epoch = 10

	#MODEL
	nb_filters = 20
	input_shape = (224,224,224,n_channels)
	mymodel = set_model(nb_filters,input_shape,n_classes)

	# LOAD INPUTS
	for e in range(n_epoch):
		print("===== EPOCH "+str(e)+" =====")
		all_files = [DATA_DIR+x for x in os.listdir(DATA_DIR)]
		for batchfile in all_files:
			print("- Loading Inputs for Batch: "+batchfile)
			X_train,Y_train = batch_load(batchfile)
			print("Training Shape: "+str(X_train.shape))
			print("Training Labels: "+str(Y_train.shape))

			mymodel.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,verbose=1)
		X_train = None
		Y_train = None
		# TEST DATA
		X_test,Y_test = batch_load("./norm_test/batch36.npz")
		print("Test Shape: "+str(X_test.shape))
		print("Test Labels: "+str(Y_test.shape))
		score = mymodel.evaluate(X_test, Y_test, verbose=0)
		predict = mymodel.predict(X_test,verbose=0)
		X_test = None
		Y_test = None 
		print('Test score:', score[0])
		print('Test accuracy: ', score[1])
		print('Predictions: ',predict)
	model_json = mymodel.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	mymodel.save("model.h5")
	print("Saved model to disk")
	del mymodel

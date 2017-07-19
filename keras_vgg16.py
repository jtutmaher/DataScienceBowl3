# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.optimizers import Adam
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

import numpy as np

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        inputs_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `tf` dim ordering)
            or `(3, 224, 244)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.dim_ordering=K.image_dim_ordering(),
    '''
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=512,
                                      min_size=48,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout( 0.25 )(x)
    
    # Block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout( 0.25 )(x)

    # Block 3
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Dropout( 0.25 )(x)

    # Block 4
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Dropout( 0.25 )(x)

    # Block 5
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Dropout( 0.25 )(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout( 0.5 )(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout( 0.5 )(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    elif weights != None:
        model.load_weights( weights )
    return model

def one_hot(labels,n_classes=2):
	new_labels = np.zeros((len(labels),n_classes))
	new_labels[np.arange(len(labels)),list(labels)]=1 
	return new_labels

def prepare_data( imgs, label_file ):
    labels = one_hot(label_file)
    img_mean = np.mean( imgs,axis=0 )
    img_std  = np.std( imgs )
    imgs -= img_mean
    imgs /= img_std
    return imgs, labels

def train(model, imgs_train, imgs_labels):
    model.fit(imgs_train, imgs_labels, batch_size=4, nb_epoch=20, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])


if __name__ == "__main__":
    # Test pretrained model
    model = VGG16(weights=None)
    model_checkpoint = ModelCheckpoint('./weights/vgg.hdf5', monitor='loss', save_best_only=True)
    model.compile(optimizer=Adam(lr=1.0e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    trainfile = np.load('../preprocess/train.npz')
    testfile = np.load('../preprocess/validate.npz')
    train_imgs, train_labels = prepare_data( trainfile['data'].astype(np.float32), trainfile['labels'] )
    test_imgs, test_labels   = prepare_data( testfile['data'].astype(np.float32), testfile['labels'] )
    print(train_imgs.shape,train_labels.shape)
    print(test_imgs.shape,test_labels.shape)

    train_imgs = train_imgs.transpose( 0, 2, 3, 1 )
    test_imgs  = test_imgs.transpose( 0, 2, 3, 1 )
    
    train( model, train_imgs, train_labels )
    score = model.evaluate( test_imgs, test_labels,batch_size=4 )
    print( 'Test results: score = ', score[0], '; accuracy = ', score[1] )

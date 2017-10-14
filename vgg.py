#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os, sys, time
import numpy as np
from scipy.misc import imread, imresize
try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))

"""
VGG-16 for ImageNet
Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper “Very Deep Convolutional Networks for
Large-Scale Image Recognition”  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.
Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow
Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping, as shown in the
following snippet:
>>> image_h, image_w, _ = np.shape(img)
>>> shorter_side = min(image_h, image_w)
>>> scale = 224. / shorter_side
>>> image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
>>> img = imresize(img, (image_h, image_w))
>>> crop_x = (image_w - 224) / 2
>>> crop_y = (image_h - 224) / 2
>>> img = img[crop_y:crop_y+224,crop_x:crop_x+224,:]
"""


def conv_layers(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
    """ conv1 """
    network = Conv2dLayer(net_in, act = tf.nn.relu, shape = [3, 3, 3, 64],      # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool1')
    """ conv2 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool2')
    """ conv3 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool3')
    """ conv4 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool4')
    """ conv5 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool5')
    return network


def conv_layers_simple_api(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool5')
    return network

def fc_layers(net , class_num):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    network = DenseLayer(network, n_units=class_num, act=tf.identity, name='fc3_relu')
    return network

def vgg_16(x , class_num=1000):
#sess = tf.InteractiveSession()

#x = tf.placeholder(tf.float32, [None, 64, 64, 1])
# y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    net_in = InputLayer(x, name='input')
    # net_cnn = conv_layers(net_in)               # professional CNN APIs
    net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
    network = fc_layers(net_cnn , class_num)

    #y = network.outputs
    return network

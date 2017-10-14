#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import tarfile
import sys
import os

import numpy as np
from six.moves import urllib, range
from six.moves import cPickle as pickle
from scipy.io import loadmat


def standardize(data_train, data_test):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0)
    std[std == 0] = 1
    mean = np.mean(data_train, 0)
    data_train_standardized \
        = (data_train - np.full(data_train.shape, mean, dtype='float32')) / \
        np.full(data_train.shape, std, dtype='float32')
    data_test_standardized \
        = (data_test - np.full(data_test.shape, mean, dtype='float32')) / \
        np.full(data_test.shape, std, dtype='float32')
    return data_train_standardized, data_test_standardized, mean, std


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.

    :param x: 1-D Numpy array of type int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth), dtype=np.int32)
    ret[np.arange(x.shape[0]), x] = 1

def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def delete_all_zero_columns(C_ ):
    shape = list(C_.shape)[:-1]
    shape.append(123)
    shape = tuple(shape)
    C = np.zeros(shape)
    cnt = 0

    if len(shape) == 3:
        for i in range(123):
            if np.max(C_[:, :, i]) != 0:
                C[:, :, cnt] = C_[:, :, i]
                cnt += 1
        C = C[:, :, 0:cnt]
        return C
    elif len(shape) == 2:
        for i in range(123):
            if np.max(C_[:, i]) != 0:
                C[:, cnt] = C_[:, i]
                cnt += 1
            C = C[:, 0:cnt]
            return C
        else:
            print('error')
            return None


    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y




def hccr_onehot_standard_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_2939x200x64x64_stand.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_sogou_64(ny, sample_num = 1000, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_500x1000x64x64_sogou.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_casia_online_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-online.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_casia_offline_reverse_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-offline-reverse.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_casia_offline_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-offline.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_casia_offline_reverse_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_test_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-offline.npy'))
    X = X[:ny , :sample_num , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]

    test_X = X[:,train_num: , : , :]
    test_Y = Y[:,train_num: ]

    return train_X ,  test_X , train_Y , test_Y

def hccr_onehot_hand_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_val_test_rate=[0.8, 0.1, 0.1]):
    X = np.load(os.path.join(data_dir , 'image_100x1000x64x64_hand.npy'))
    X = X[:ny , : , : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]
    val_X = X[: , train_num:train_num+val_num, : , :]
    val_Y = Y[: , train_num:train_num+val_num ]

    test_X = X[:,train_num+val_num: , : , :]
    test_Y = Y[:,train_num+val_num: ]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y

def hccr_code_stand_28(ny, sample_num = 100, data_dir='/home/danyang/mfs/data/hccr', train_val_test_rate=[0.8, 0.1, 0.1]):
    X = np.load(os.path.join(data_dir , 'image_100x100x28x28_stand.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_100x1000x123.npy'))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]
    val_X = X[: , train_num:train_num+val_num, : , : ]
    val_Y = Y[: , train_num:train_num+val_num , : ]

    test_X = X[:,train_num+val_num: , : ,:]
    test_Y = Y[:,train_num+val_num: , : ]

    code_len = Y.shape[-1]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y , code_len

def hccr_code_standard_64(ny, sample_num = 100, data_dir='/home/danyang/mfs/data/hccr', train_test_rate=[0.8,  0.2]):
    X = np.load(os.path.join(data_dir , 'image_2939x200x64x64_stand.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_2939x123.npy'))
    Y = np.tile(np.expand_dims(Y , 1) , (1,200,1))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]

    test_X = X[:,train_num: , : ,:]
    test_Y = Y[:,train_num: , : ]

    code_len = Y.shape[-1]

    return train_X  ,  test_X , train_Y  , test_Y , code_len

def hccr_code_sogou_64(ny, sample_num = 1000, data_dir='/home/danyang/mfs/data/hccr', train_test_rate=[0.8,  0.2]):
    X = np.load(os.path.join(data_dir , 'image_500x1000x64x64_stand.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_2939x123.npy'))   #TODO
    Y = np.tile(np.expand_dims(Y , 1) , (1,200,1))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]

    test_X = X[:,train_num: , : ,:]
    test_Y = Y[:,train_num: , : ]

    code_len = Y.shape[-1]

    return train_X  ,  test_X , train_Y  , test_Y , code_len

def hccr_code_casia_online_64(ny, sample_num = 100, data_dir='/home/danyang/mfs/data/hccr', train_test_rate=[0.8,  0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-online.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_1000x123.npy'))
    Y = np.tile(np.expand_dims(Y , 1) , (1,300,1))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]

    test_X = X[:,train_num: , : ,:]
    test_Y = Y[:,train_num: , : ]

    code_len = Y.shape[-1]

    return train_X  ,  test_X , train_Y  , test_Y , code_len

def hccr_code_casia_offline_64(ny, sample_num = 100, data_dir='/home/danyang/mfs/data/hccr', train_test_rate=[0.8,  0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-offline.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_1000x123.npy'))
    Y = np.tile(np.expand_dims(Y , 1) , (1,300,1))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]

    test_X = X[:,train_num: , : ,:]
    test_Y = Y[:,train_num: , : ]

    code_len = Y.shape[-1]

    return train_X  ,  test_X , train_Y  , test_Y , code_len

def hccr_code_casia_offline_reverse_64(ny, sample_num = 100, data_dir='/home/danyang/mfs/data/hccr', train_test_rate=[0.8,  0.2]):
    X = np.load(os.path.join(data_dir , 'image_1000x300x64x64_casia-offline-reverse.npy'))
    X = X[:ny , :sample_num , : , :]      #ny*sample_num*28*28
    Y = np.load(os.path.join(data_dir , 'code_1000x123.npy'))
    Y = np.tile(np.expand_dims(Y , 1) , (1,300,1))
    Y = Y[:ny , :sample_num , :]
    Y = delete_all_zero_columns(Y)

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_test_rate[0])
    test_num = int(np.shape(X)[1] * train_test_rate[1])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num,: ]

    test_X = X[:,train_num: , : ,:]
    test_Y = Y[:,train_num: , : ]

    code_len = Y.shape[-1]

    return train_X  ,  test_X , train_Y  , test_Y , code_len


def hccr_onehot_casia_offline_64(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_rate=[0.8, 0.2]):
    X = np.load(os.path.join(data_dir , 'image_500x300x64x64_casia-offline.npy'))
    X = X[:ny , : sample_num, : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]
    val_X = X[: , train_num:train_num+val_num, : , :]
    val_Y = Y[: , train_num:train_num+val_num ]

    test_X = X[:,train_num+val_num: , : , :]
    test_Y = Y[:,train_num+val_num: ]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y

def hccr_onehot_competition_64(ny, sample_num = 100, data_dir ='/home/tongzheng/mfs/data', train_val_test_rate=[0.8, 0.1, 0.1]):
    X = np.load(os.path.join(data_dir , 'image_26x163x64x64_competition.npy'))
    X = X[:ny , : sample_num, : , :] #ny*1000*64*64
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]
    val_X = X[: , train_num:train_num+val_num, : , :]
    val_Y = Y[: , train_num:train_num+val_num ]

    test_X = X[:,train_num+val_num: , : , :]
    test_Y = Y[:,train_num+val_num: ]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y

def hccr_onehot_standard_256(ny, sample_num = 100, data_dir ='/home/danyang/mfs/data/hccr', train_val_test_rate=[0.8, 0.1, 0.1]):
    X = np.load(os.path.join(data_dir , 'image_100x163x256x256_stand.npy'))
    X = X[:ny , : sample_num , : , :]
    Y = np.ones((ny,sample_num) , dtype=np.int)
    for i in range(ny):
        Y[i,:] = np.ones(sample_num)*i #ny*1000

    assert np.shape(X)[1] == np.shape(Y)[1]
    total_num = np.shape(X)[1]
    train_num = int(np.shape(X)[1] * train_val_test_rate[0])
    val_num = int(np.shape(X)[1] * train_val_test_rate[1])
    test_num = int(np.shape(X)[1] * train_val_test_rate[2])

    train_X = X[:,0:train_num,: , :]
    train_Y = Y[:,0:train_num]
    val_X = X[: , train_num:train_num+val_num, : , :]
    val_Y = Y[: , train_num:train_num+val_num ]

    test_X = X[:,train_num+val_num: , : , :]
    test_Y = Y[:,train_num+val_num: ]

    return train_X ,  val_X ,  test_X , train_Y , val_Y , test_Y

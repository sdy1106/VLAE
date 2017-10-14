#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import os
import random

import tensorflow as tf
from tensorflow.python.training import optimizer
import numpy as np
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity
from six.moves import range


class AdamaxOptimizer(optimizer.Optimizer):
    """
    Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

def to_onehot(a , max_num):
    shape_a = list(a.shape)
    shape_a.append(max_num)
    b = np.zeros(tuple(shape_a) , dtype=np.float32)
    b = np.reshape(b , [-1 , max_num])
    cnt = 0
    for i in a.flat:
        b[cnt , i-1] = 1
        cnt += 1
    b = np.reshape(b , shape_a)
    return b

def shuffle(a , b):
    assert a.shape[0] == b.shape[0]
    all = np.concatenate((a , b) , axis=1)
    np.random.shuffle(all)
    shuffle_a , shuffle_b = all[: , :a.shape[1]] , all[: , a.shape[1]:]
    return shuffle_a , shuffle_b

def random_select(a,b):
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1]
    all = np.concatenate((a,b),axis = 2)
    select = np.array([all[random.randint(0, a.shape[0]-1), i,:] for i in range(a.shape[1])])
    return select[:,:a.shape[2]], select[:,a.shape[2]:]

def save_image_collections(x, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    """
    :param shape: tuple
        The shape of final big images.
    :param x: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :return: uint8 numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def save_contrast_image_collections(x1, x2, filename, shape=(10, 10),
                                    scale_each=False, transpose=False,
                                    along_col=True):
    """
    :param x1: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param x2: uint8 numpy array
        Input image collections. (number_of_images, rows, columns, channels) or
        (number_of_images, channels, rows, columns)
    :param shape: tuple
        The shape of final big images.
    :param scale_each: bool
        If true, rescale intensity for each image.
    :param transpose: bool
        If true, transpose x to (number_of_images, rows, columns, channels),
        i.e., put channels behind.
    :param along_col: bool
        If true, the contrastive images are placed one by one along the column.
        If False, they are placed one by one along the row.
    :return: uint8 numpy array
        The output image.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x1.shape[0]
    if transpose:
        x1 = x1.transpose(0, 2, 3, 1)
        x2 = x2.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x1[i] = rescale_intensity(x1[i], out_range=(0, 1))
            x2[i] = rescale_intensity(x2[i], out_range=(0, 1))
    n_channels = x1.shape[3]
    x1 = img_as_ubyte(x1)
    x2 = img_as_ubyte(x2)
    r, c = shape
    if r * c < 2 * n:
        print('Shape too small to contain all images')
    h, w = x1.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < 2 * n:
                if along_col:
                    if j % 2 == 0:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x1[
                            int(i * c / 2 + j / 2)]
                    else:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x2[
                            int(i * c / 2 + (j - 1) / 2)]
                else:
                    if i % 2 == 0:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x1[
                            int(j * r / 2 + i / 2)]
                    else:
                        ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x2[
                            int(j * r / 2 + (i - 1) / 2)]

    ret = ret.squeeze()
    io.imsave(filename, ret)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs
from tensorflow.python.ops import init_ops

from examples import conf
from examples.utils import dataset

n_xl = 28
n_x = 28 ** 2
ngf = 64

def lrelu(input_tensor, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_tensor + f2 * abs(input_tensor)

@zs.reuse('model')
def VLAE(observed, n, n_x, n_z_0, n_z_1, n_z_2, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as model:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
    #     z_mean = tf.zeros([n, n_z])
    #     z = zs.Normal('z', z_mean, std=1., n_samples=n_particles,
    #                   group_event_ndims=1)
    #     lx_z = layers.fully_connected(
    #         z, 500, normalizer_fn=layers.batch_norm,
    #         normalizer_params=normalizer_params)
    #     lx_z = layers.fully_connected(
    #         lx_z, 500, normalizer_fn=layers.batch_norm,
    #         normalizer_params=normalizer_params)
    #     x_logits = layers.fully_connected(lx_z, n_x, activation_fn=None)
    #     x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    # return model
        z_2_mean = tf.zeros([n, n_z_2])
        z_2 = zs.Normal('z_2', z_2_mean, std=1., n_samples=1, group_event_ndims=1)
        z_2 = tf.reshape(z_2, [-1, n_z_2])
        z_1_mean = tf.zeros([n, n_z_1])
        z_1 = zs.Normal('z_1', z_1_mean, std=1., n_samples=1, group_event_ndims=1)
        z_1 = tf.reshape(z_1, [-1, n_z_1])
        z_0_mean = tf.zeros([n, n_z_0])
        z_0 = zs.Normal('z_0', z_0_mean, std=1., n_samples=1, group_event_ndims=1)
        z_0 = tf.reshape(z_0, [-1, n_z_0])
        latent2 = z_2
        latent1 = z_1
        latent0 = z_0

        ladder2 = layers.fully_connected(latent2, ngf * 16 * 1 * 1, activation_fn=tf.nn.relu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder2 = tf.reshape(ladder2, [-1, 4, 4, ngf * 16])
        ladder2 = layers.conv2d_transpose(ladder2, ngf * 16, 4, stride=1, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder2 = layers.conv2d_transpose(ladder2, ngf * 4, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        inference1 = layers.conv2d_transpose(ladder2, ngf * 4, 4, stride=1, activation_fn=tf.nn.relu,
                                             normalizer_fn=layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder1 = layers.fully_connected(latent1, ngf * 4 * 7 * 7, activation_fn=tf.nn.relu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder1 = tf.reshape(ladder1, [-1, 8, 8, ngf * 4])

        ladder1 = tf.concat([ladder1, inference1], 3)
        ladder1 = layers.conv2d_transpose(ladder1, ngf * 2, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder1 = layers.conv2d_transpose(ladder1, ngf * 2, 4, stride=1, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        inference0 = layers.conv2d_transpose(ladder1, ngf, 4, stride=2, activation_fn=tf.nn.relu,
                                             normalizer_fn=layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder0 = layers.fully_connected(latent0, ngf * 1 * 14 * 14, activation_fn=tf.nn.relu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = tf.reshape(ladder0, [-1, 14, 14, ngf * 1])
        ladder0 = tf.concat([ladder0, inference0], 3)

        # ladder0 = layers.conv2d_transpose(inference0, ngf, 4, stride=1, activation_fn=tf.nn.relu,
        #                                   normalizer_fn=layers.batch_norm,
        #                                   normalizer_params=normalizer_params,
        #                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))

        x_logits = layers.conv2d_transpose(ladder0, 1, 4, stride=2, activation_fn=tf.identity,
                                           weights_initializer=init_ops.RandomNormal(stddev=0.02))

        x_logits = tf.reshape(x_logits, [-1, n_x])
        x = zs.Bernoulli('x', x_logits, n_samples=n_particles, group_event_ndims=1)
    return model, x_logits


@zs.reuse('variational')
def q_net(observed, x, n_z_0, n_z_1 , n_z_2, n_particles, is_training):
    with zs.BayesianNet(observed=observed) as variational:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
    #     lz_x = layers.fully_connected(
    #         tf.to_float(x), 500, normalizer_fn=layers.batch_norm,
    #         normalizer_params=normalizer_params)
    #     lz_x = layers.fully_connected(
    #         lz_x, 500, normalizer_fn=layers.batch_norm,
    #         normalizer_params=normalizer_params)
    #     z_mean = layers.fully_connected(lz_x, n_z, activation_fn=None)
    #     z_logstd = layers.fully_connected(lz_x, n_z, activation_fn=None)
    #     z = zs.Normal('z', z_mean, logstd=z_logstd, n_samples=n_particles,
    #                   group_event_ndims=1)
    # return variational
        x = tf.reshape(tf.to_float(x), [-1, n_xl, n_xl, 1])
        ladder0 = layers.conv2d(x, ngf, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf, 4, stride=1, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.flatten(ladder0)
        latent0_mean = layers.fully_connected(ladder0, 2, activation_fn=tf.identity)
        latent0_std = layers.fully_connected(ladder0, 2, activation_fn=tf.sigmoid)

        inference0 = layers.conv2d(x, ngf, 4, stride=2, activation_fn=lrelu,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))
        inference0 = layers.conv2d(inference0, ngf, 4, stride=1, activation_fn=lrelu,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder1 = layers.conv2d(inference0, ngf * 2, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder1 = layers.conv2d(ladder1, ngf * 2, 4, stride=1, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder1 = layers.conv2d(ladder1, ngf * 4, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder1 = layers.flatten(ladder1)
        latent1_mean = layers.fully_connected(ladder1, 2, activation_fn=tf.identity)
        latent1_std = layers.fully_connected(ladder1, 2, activation_fn=tf.sigmoid)

        inference1 = layers.conv2d(inference0, ngf * 2, 4, stride=2, activation_fn=lrelu,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))
        inference1 = layers.conv2d(inference1, ngf * 2, 4, stride=1, activation_fn=lrelu,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))
        inference1 = layers.conv2d(inference1, ngf * 4, 4, stride=2, activation_fn=lrelu,
                                   normalizer_fn=layers.batch_norm,
                                   normalizer_params=normalizer_params,
                                   weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder2 = layers.conv2d(inference1, ngf * 4, 4, stride=1, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder2 = layers.conv2d(ladder2, ngf * 4, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder2 = layers.conv2d(ladder2, ngf * 8, 4, stride=1, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder2 = layers.flatten(ladder2)
        latent2_mean = layers.fully_connected(ladder2, 3, activation_fn=tf.identity)
        latent2_std = layers.fully_connected(ladder2, 3, activation_fn=tf.sigmoid)

        z_0 = zs.Normal('z_0', mean=latent0_mean, std=latent0_std, n_samples=n_particles, group_event_ndims=1)
        z_1 = zs.Normal('z_1', mean=latent1_mean, std=latent1_std, n_samples=n_particles, group_event_ndims=1)
        z_2 = zs.Normal('z_2', mean=latent2_mean, std=latent2_std, n_samples=n_particles, group_event_ndims=1)

    return variational, z_0, z_1, z_2


if __name__ == "__main__":
    tf.set_random_seed(1237)

    # Load MNIST
    data_path = os.path.join(conf.data_dir, 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    np.random.seed(1234)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype('float32')
    n_x = x_train.shape[1]

    # Define model parameters
    n_z_0 = 2
    n_z_1 = 2
    n_z_2 = 3

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 1000
    epochs = 3000
    batch_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    save_freq = 100
    result_path = "results/vae"

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1])
    n = tf.shape(x)[0]

    def log_joint(observed):
        model,_ = VLAE(observed, n, n_x, n_z_0, n_z_1, n_z_2, n_particles, is_training)
        log_pz0 ,log_pz1, log_pz2 ,log_px_z = model.local_log_prob(['z_0','z_1','z_2', 'x'])
        return log_pz0 + log_pz1 + log_pz2 + log_px_z

    variational,_,_,_ = q_net({}, x, n_z_0, n_z_1, n_z_2, n_particles, is_training)
    qz_samples0, log_qz0 = variational.query('z_0', outputs=True,
                                             local_log_prob=True)
    qz_samples1, log_qz1 = variational.query('z_1', outputs=True,
                                            local_log_prob=True)
    qz_samples2, log_qz2 = variational.query('z_2', outputs=True,
                                            local_log_prob=True)
    lower_bound = tf.reduce_mean(
        zs.sgvb(log_joint, {'x': x_obs}, {'z_0': [qz_samples0, log_qz0],
                'z_1': [qz_samples1, log_qz1],'z_2': [qz_samples2, log_qz2]}, axis=0))

    # Importance sampling estimates of marginal log likelihood
    is_log_likelihood = tf.reduce_mean(
        zs.is_loglikelihood(log_joint, {'x': x_obs}, {'z_0': [qz_samples0, log_qz0],
                'z_1': [qz_samples1, log_qz1] , 'z_2': [qz_samples2, log_qz2]}, axis=0))

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads = optimizer.compute_gradients(-lower_bound)
    infer = optimizer.apply_gradients(grads)

    params = tf.trainable_variables()
    for i in params:
        print(i.name, i.get_shape())

    saver = tf.train.Saver(max_to_keep=10)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
                _, lb = sess.run([infer, lower_bound],
                                 feed_dict={x: x_batch_bin,
                                            learning_rate_ph: learning_rate,
                                            n_particles: lb_samples,
                                            is_training: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:
                                          (t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: lb_samples,
                                                  is_training: False})
                    test_ll = sess.run(is_log_likelihood,
                                       feed_dict={x: test_x_batch,
                                                  n_particles: ll_samples,
                                                  is_training: False})
                    test_lbs.append(test_lb)
                    test_lls.append(test_ll)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))
                print('>> Test log likelihood (IS) = {}'.format(
                    np.mean(test_lls)))

            if epoch % save_freq == 0:
                print('Saving model...')
                save_path = os.path.join(result_path,
                                         "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                print('Done')

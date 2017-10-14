# -*- coding: utf-8 -*-

import argparse
import os
import time

import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops

import multi_gpu
import utils
from multi_gpu import FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--result_path', type=str, default='./result/VAE')
parser.add_argument('--dataset', type=str, default='standard')
parser.add_argument('--char_num', type=int, default=1000, help='The number of characters')
parser.add_argument('--sample_num', type=int, default=100, help='the number of each character sampling')
parser.add_argument('--mode', type=str, default='font')
parser.add_argument('--pairwise', default=False, action='store_true')
parser.add_argument('--char_dim', type=int, default=100)
parser.add_argument('--font_dim', type=int, default=60)
args = parser.parse_args()

print ('gpus:', FLAGS.num_gpus)
print (args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

n_y = args.char_num
n_font = args.sample_num
n_xl = 64
n_x = 64 ** 2
ngf = 64
char_dim = args.char_dim
font_dim = args.font_dim
train_ratio = 0.9


def lrelu(input_tensor, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_tensor + f2 * abs(input_tensor)

def conv_cond_concat(x, y):
    y_ = y * tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(y)[3]])
    return tf.concat([x, y_], 3)

@zs.reuse('encoder_font')
def q_net_font(observed, x, is_training):
    with zs.BayesianNet(observed=observed) as encoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        x = tf.reshape(tf.to_float(x), [-1, n_xl, n_xl, 1])
        ladder0 = layers.conv2d(x, ngf, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 2, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 4, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 8, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.flatten(ladder0)
        font_mean = layers.fully_connected(ladder0, font_dim, activation_fn=tf.identity)
        font_std = layers.fully_connected(ladder0, font_dim, activation_fn=tf.sigmoid)

        z_font = zs.Normal('z_font', mean=font_mean, std=font_std, n_samples=1, group_event_ndims=1)
    return encoder, z_font

@zs.reuse('encoder_char')
def q_net_char(observed, x, is_training):
    with zs.BayesianNet(observed=observed) as encoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        x = tf.reshape(tf.to_float(x), [-1, n_xl, n_xl, 1])
        ladder0 = layers.conv2d(x, ngf, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 2, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 4, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 8, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 8, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.conv2d(ladder0, ngf * 8, 4, stride=2, activation_fn=lrelu,
                                normalizer_fn=layers.batch_norm,
                                normalizer_params=normalizer_params,
                                weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = layers.flatten(ladder0)
        char_mean = layers.fully_connected(ladder0, char_dim, activation_fn=tf.identity)
        char_std = layers.fully_connected(ladder0, char_dim, activation_fn=tf.sigmoid)

        z_char = zs.Normal('z_char', mean=char_mean, std=char_std, n_samples=1, group_event_ndims=1)
    return encoder, z_char

# @zs.reuse('decode_font')
# def VAE_font(observed, n, y, is_training):
#     with zs.BayesianNet(observed=observed) as decoder:
#         normalizer_params = {'is_training': is_training,
#                              'updates_collections': None}
#         z_mean = tf.zeros([n, font_dim])
#         z_font = zs.Normal('z_font', z_mean, std=1., n_samples=1, group_event_ndims=1)
#         z_font = tf.reshape(z_font,[-1 ,font_dim])
#         yb = tf.reshape(y, [-1, 1, 1, int(train_ratio * n_y)])
#
#         lx_z = layers.fully_connected(tf.concat([z_font, y], 1), ngf * 8 * 4 * 4, activation_fn=tf.nn.relu,
#                                        normalizer_fn=layers.batch_norm,
#                                        normalizer_params=normalizer_params,
#                                        weights_initializer=init_ops.RandomNormal(stddev=0.02))
#         lx_z = tf.reshape(lx_z, [-1, 4, 4, ngf * 8])
#
#         lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 4, 4, stride=2, activation_fn=tf.nn.relu,
#                                        normalizer_fn=layers.batch_norm,
#                                        normalizer_params=normalizer_params,
#                                        weights_initializer=init_ops.RandomNormal(stddev=0.02))
#         lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 2, 4, stride=2, activation_fn=tf.nn.relu,
#                                        normalizer_fn=layers.batch_norm,
#                                        normalizer_params=normalizer_params,
#                                        weights_initializer=init_ops.RandomNormal(stddev=0.02))
#         lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 1, 4, stride=2, activation_fn=tf.nn.relu,
#                                        normalizer_fn=layers.batch_norm,
#                                        normalizer_params=normalizer_params,
#                                        weights_initializer=init_ops.RandomNormal(stddev=0.02))
#         lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), 1, 4, stride=2, activation_fn=None,
#                                        weights_initializer=init_ops.RandomNormal(stddev=0.02))
#         x_logits = tf.reshape(lx_z, [1, -1, n_x])
#         x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
#     return decoder, x_logits

@zs.reuse('decoder')
def VAE(observed, n, is_training):
    with zs.BayesianNet(observed=observed) as decoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_char_mean = tf.zeros([n, char_dim])
        z_char = zs.Normal('z_char', z_char_mean, std=1., n_samples=1, group_event_ndims=1)
        z_char = tf.reshape(z_char, [-1, char_dim])
        z_font_mean = tf.zeros([n, font_dim])
        z_font = zs.Normal('z_font', z_font_mean, std=1., n_samples=1, group_event_ndims=1)
        z_font = tf.reshape(z_font, [-1, font_dim])
        latent1 = z_char
        latent0 = z_font
        ladder1 = layers.fully_connected(latent1, ngf * 8 * 1 * 1, activation_fn=tf.nn.relu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder1 = tf.reshape(ladder1, [-1, 1, 1, ngf * 8])

        ladder1 = layers.conv2d_transpose(ladder1, ngf * 8, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))
        inference0 = layers.conv2d_transpose(ladder1, ngf * 8, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder0 = layers.fully_connected(latent0, ngf * 8 * 4 * 4, activation_fn=tf.nn.relu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         weights_initializer=init_ops.RandomNormal(stddev=0.02))
        ladder0 = tf.reshape(ladder0, [-1, 4, 4, ngf * 8])
        ladder0 = tf.concat([ladder0, inference0], 3)


        ladder0 = layers.conv2d_transpose(ladder0, ngf * 4, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder0 = layers.conv2d_transpose(ladder0, ngf * 2, 4, stride=2, activation_fn=tf.nn.relu,
                                             normalizer_fn=layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             weights_initializer=init_ops.RandomNormal(stddev=0.02))

        ladder0 = layers.conv2d_transpose(ladder0, ngf, 4, stride=2, activation_fn=tf.nn.relu,
                                          normalizer_fn=layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          weights_initializer=init_ops.RandomNormal(stddev=0.02))

        x_logits = layers.conv2d_transpose(ladder0, 1, 4, stride=2, activation_fn=tf.identity,
                                           weights_initializer=init_ops.RandomNormal(stddev=0.02))

        x_logits = tf.reshape(x_logits, [1, -1, n_x])
        x = zs.Bernoulli('x', x_logits, n_samples=1, group_event_ndims=1)
    return decoder, x_logits


def main():
    if args.dataset == 'standard':
        X = np.load('/home/danyang/mfs/data/hccr/image_1000x163x64x64_stand.npy')
        if n_font > 100:
            print('too much fonts')
            os._exit(-1)
    elif args.dataset == 'casia-offline':
        X = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-offline.npy')
    elif args.dataset == 'casia-online':
        X = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-online.npy')
    else:
        print('Unknown Dataset!')
        os._exit(-1)
    train_x = X[:int(train_ratio * n_y), :int(train_ratio * n_font), :, :]
    code_x = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[0]))
    for i in range(train_x.shape[0]):
        code_x[i, :, i] = np.ones(train_x.shape[1])
    test_x_font = X[:int(train_ratio * n_y), int(train_ratio * n_font):n_font, :, :]
    code_test = np.zeros((test_x_font.shape[0], test_x_font.shape[1], test_x_font.shape[0]))
    for i in range(test_x_font.shape[0]):
        code_test[i, :, i] = np.ones(test_x_font.shape[1])
    test_x_char = X[int(train_ratio * n_y):n_y, :int(train_ratio * n_font), :, :]
    test_x = X[int(train_ratio * n_y):n_y, int(train_ratio * n_font):n_font, :, :]

    epochs = args.epoch
    train_batch_size = args.batch_size * FLAGS.num_gpus
    learning_rate = args.lr
    anneal_lr_freq = 200
    anneal_lr_rate = 0.75
    result_path = args.result_path
    train_iters = min(train_x.shape[0] * train_x.shape[1], 10000) // train_batch_size

    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    # font_source = tf.placeholder(tf.int32, shape=[None, n_x], name='font_source')
    # char_source = tf.placeholder(tf.int32, shape=[None, n_x], name='char_source')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    def build_tower_graph_font(id_):
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]
        x_obs = tf.tile(tf.expand_dims(tower_x, 0), [1, 1, 1])

        def log_joint(observed):
            decoder, _, = VAE(observed, n, is_training)
            log_pz_font, log_pz_char, log_px_z = decoder.local_log_prob(['z_font', 'z_char', 'x'])
            return log_pz_font + log_pz_char + log_px_z


        #train font
        encoder_font, qz_samples_font = q_net_font(None, tower_x, is_training)
        encoder_char, qz_samples_char = q_net_char(None, tower_x, is_training)

        char_mean = tf.tile(tf.reduce_mean(qz_samples_char , 0) , (tf.shape(qz_samples_font)[0] , 1))
        lower_bound = tf.reduce_mean(
            zs.sgvb(log_joint, {'x': tower_x},
                    {'z_font': [qz_samples_font, log_qz_font], 'z_char': [char_mean, log_qz_char]}, axis=0))
        average_loss = tf.reduce_mean(tf.square(qz_samples_char-char_mean))

        font_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='encoder_font') + \
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='decoder')
        char_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope='encoder_char')
        font_grads = optimizer.compute_gradients(-lower_bound , var_list=font_var_list)
        char_grads = optimizer.compute_gradients(average_loss , var_list=char_var_list)


        return font_grads,char_grads, lower_bound , average_loss

    def build_tower_graph_char(id_):
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]
        x_obs = tf.tile(tf.expand_dims(tower_x, 0), [1, 1, 1])

        def log_joint(observed):
            decoder, _, = VAE(observed, n, is_training)
            log_pz_font, log_pz_char, log_px_z = decoder.local_log_prob(['z_font', 'z_char', 'x'])
            return log_pz_font + log_pz_char + log_px_z

        # train char
        encoder_font, _ = q_net_font(None, tower_x, is_training)
        qz_samples_font, log_qz_font = encoder_font.query('z_font', outputs=True,
                                                          local_log_prob=True)
        encoder_char, _ = q_net_char(None, tower_x, is_training)
        qz_samples_char, log_qz_char = encoder_char.query('z_char', outputs=True,
                                                          local_log_prob=True)

        font_mean = tf.tile(tf.reduce_mean(qz_samples_font, 0), (tf.shape(qz_samples_char)[0], 1))
        # lower_bound = tf.reduce_mean(
        #     zs.sgvb(log_joint, {'x': tower_x},
        #             {'z_font': [font_mean, log_qz_font], 'z_char': [qz_samples_char, log_qz_char]}, axis=0))
        _ , x_recon = VAE({'z_font': font_mean, 'z_char': z_char}, tf.shape(tower_x)[0], is_training)
        loss_recon = tf.reduce_mean(tf.square(x_recon - tower_x))


        font_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='encoder_font') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='decoder')
        char_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='encoder_char') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='decoder')
        average_loss = tf.reduce_mean(tf.square(qz_samples_font - font_mean))

        font_grads = optimizer.compute_gradients(average_loss, var_list=font_var_list)
        char_grads = optimizer.compute_gradients(-lower_bound, var_list=char_var_list)

        return font_grads, char_grads, lower_bound, average_loss , font_mean


    f_font_grads, f_char_grads, f_lower_bound, f_average_loss = build_tower_graph_font(0)
    c_font_grads, c_char_grads, c_lower_bound, c_average_loss,font_mean = build_tower_graph_char(0)
    f_infer_font = optimizer.apply_gradients(f_font_grads)
    f_infer_char = optimizer.apply_gradients(f_char_grads)
    c_infer_font = optimizer.apply_gradients(c_font_grads)
    c_infer_char = optimizer.apply_gradients(c_char_grads)

    params = tf.trainable_variables()

    for i in params:
        print(i.name, i.get_shape())
    saver = tf.train.Saver(max_to_keep=10,
                           var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='encoder_font') + \
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='encoder_char') + \
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='decoder'))

    # if args.mode == 'font':
    #     _, z_font = q_net_font(None, font_source, is_training)
    #     _, x_gen = VAE_font({'z_font': z_font}, tf.shape(font_source)[0], code, is_training)
    #     x_gen = tf.reshape(tf.sigmoid(x_gen), [-1, n_xl, n_xl, 1])
    #     _, _ = q_net_char(None, char_source, is_training)
    #     _, _ = VAE({'z_font': z_font, 'z_char': z_char}, tf.shape(char_source)[0], is_training)
    #
    # else:
    #     _, z_font = q_net_font(None, font_source, is_training)
    #     _, z_char = q_net_char(None, char_source, is_training)
    #     _, x_gen = VAE({'z_font': z_font, 'z_char': z_char}, tf.shape(char_source)[0], is_training)
    #     x_gen = tf.reshape(tf.sigmoid(x_gen), [-1, n_xl, n_xl, 1])


    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate

            time_train = -time.time()
            lower_bounds = []
            average_losses = []
            x_train = train_x.reshape(-1, n_x)
            np.random.shuffle(x_train)
            x_train = x_train[:min(train_x.shape[0] * train_x.shape[1], 10000)]
            np.random.shuffle(x_train)
            print ('train_x.shape:', train_x.shape)
            if epoch % 2 == 0:
                print ('enter font train')
                x_font_train = np.array([X[np.random.randint(0, train_x.shape[0] - 1), i, :, :] for i in range(train_x.shape[1])])
                np.random.shuffle(x_font_train)

                #print 'x_font shape:' , x_font_train.shape
                for i in range(train_x.shape[1] // train_batch_size):
                    _, _,  lb , al, fm = sess.run([f_infer_char,f_infer_char  , f_lower_bound , f_average_loss , font_mean],
                                     feed_dict={x: x_font_train[i * train_batch_size: (i + 1) * train_batch_size, :n_x].reshape(-1, 4096),
                                           learning_rate_ph: learning_rate,
                                           is_training: True})
                    lower_bounds.append(lb)
                    average_losses.append(al)
                #print (fm.shape)
            else:
                print ('enter char train')
                print ('iteras:%d' % (train_x.shape[0] // train_batch_size))
                x_char_train = np.array([X[ i ,np.random.randint(0, train_x.shape[1] - 1), :, :] for i in range(train_x.shape[0])])
                #print 'x_char shape:', x_char_train.shape
                np.random.shuffle(x_char_train)
                for i in range(train_x.shape[0] // train_batch_size):
                    _, _, lb, al = sess.run([c_infer_char, c_infer_char, c_lower_bound, f_average_loss],
                                            feed_dict={x: x_char_train[i * train_batch_size: (i + 1) * train_batch_size, :n_x].reshape(-1, 4096),
                                                       learning_rate_ph: learning_rate,
                                                       is_training: True})
                    lower_bounds.append(lb)
                    average_losses.append(al)
            print('Epoch={} ({:.3f}s/epoch): '
                  'Lower Bound = {} , Average loss = {}'.
                  format(epoch, (time.time() + time_train),
                         np.mean(lower_bounds) , np.mean(average_losses)))

            # if args.mode != 'font':
            #     # train reconstruction
            #     gen_images = sess.run(x_gen, feed_dict={char_source: train_x[:10, :10, :, :].reshape(-1, n_x),
            #                                             font_source: train_x[:10, :10, :, :].reshape(-1, n_x),
            #                                             is_training: False})
            #
            #     name = "train_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_contrast_image_collections(train_x[:10, :10, :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
            #                                           name, shape=(10, 20),
            #                                           scale_each=True)
            #
            #     # new font reconstruction
            #     char_index = np.arange(test_x_font.shape[0])
            #     font_index = np.arange(test_x_font.shape[1])
            #     np.random.shuffle(char_index)
            #     np.random.shuffle(font_index)
            #     gen_images = sess.run(x_gen, feed_dict={char_source: test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                             font_source: test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                             is_training: False})
            #     name = "test_font_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_contrast_image_collections(test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
            #                                           name, shape=(10, 20),
            #                                           scale_each=True)
            #
            #     # new char reconstruction
            #     char_index = np.arange(test_x_char.shape[0])
            #     font_index = np.arange(test_x_char.shape[1])
            #     np.random.shuffle(char_index)
            #     np.random.shuffle(font_index)
            #     gen_images = sess.run(x_gen, feed_dict={char_source: test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                             font_source: test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                             is_training: False})
            #
            #     name = "test_char_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_contrast_image_collections(test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
            #                                           name, shape=(10, 20),
            #                                           scale_each=True)
            #
            #     # never seen reconstruction
            #     char_index = np.arange(test_x.shape[0])
            #     font_index = np.arange(test_x.shape[1])
            #     np.random.shuffle(char_index)
            #     np.random.shuffle(font_index)
            #     gen_images = sess.run(x_gen,
            #                           feed_dict={char_source: test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                      font_source: test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
            #                                      is_training: False})
            #
            #     name = "test_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_contrast_image_collections(test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
            #                                           name, shape=(10, 20),
            #                                           scale_each=True)
            #
            #     # one shot font generation
            #     font_index = np.arange(test_x_font.shape[1])
            #     np.random.shuffle(font_index)
            #     test_x_font_feed = np.tile(np.expand_dims(
            #         np.array([test_x_font[np.random.randint(test_x_font.shape[0] - 1), font_index[i], :, :] for i in range(10)]), 0),
            #         (10, 1, 1, 1))
            #     gen_images = sess.run(x_gen, feed_dict={char_source: train_x[:10, :10, :, :].reshape(-1, n_x),
            #                                             font_source: test_x_font_feed[:10, :10, :, :].reshape(-1, n_x),
            #                                             is_training: False})
            #     images = np.concatenate([test_x_font_feed[0].reshape(-1, n_xl, n_xl, 1), gen_images], 0)
            #
            #     name = "one_shot_font_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_image_collections(images, name, shape=(11, 10),
            #                                  scale_each=True)
            #
            #     # one shot char generation
            #     char_index = np.arange(test_x_char.shape[0])
            #     np.random.shuffle(char_index)
            #     test_x_char_feed = np.tile(np.expand_dims(
            #         np.array([test_x_char[char_index[i], np.random.randint(test_x_char.shape[1] - 1), :, :] for i in range(10)]), 1),
            #         (1, 10, 1, 1))
            #     gen_images = sess.run(x_gen, feed_dict={char_source: test_x_char_feed[:10, :10, :, :].reshape(-1, n_x),
            #                                             font_source: train_x[:10, :10, :, :].reshape(-1, n_x),
            #                                             is_training: False})
            #     name = "one_shot_char_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     images = np.zeros((110, 64, 64, 1))
            #     for i in range(10):
            #         images[i * 11] = np.expand_dims(test_x_char_feed[i, 0, :, :], 2)
            #         images[i * 11 + 1:(i + 1) * 11] = gen_images[i * 10:(i + 1) * 10]
            #     utils.save_image_collections(images, name, shape=(10, 11),
            #                                  scale_each=True)
            # else:
            #     gen_images = sess.run(x_gen, feed_dict={
            #         font_source: test_x_font[:10, :10, :, :].reshape(-1, n_x),
            #         code: code_test[:10, :10, :].reshape(-1, int(train_ratio * n_y)),
            #         is_training: False})
            #     name = "test_font_{}/VAE_hccr.epoch.{}.png".format(n_y, epoch)
            #     name = os.path.join(result_path, name)
            #     utils.save_contrast_image_collections(
            #         test_x_font[:10, :10, :, :].reshape(-1, n_xl, n_xl, 1),
            #         gen_images,
            #         name, shape=(10, 20),
            #         scale_each=True)
            save_path = "VAE.epoch.{}.ckpt".format(epoch)
            save_path = os.path.join(result_path, save_path)
            saver.save(sess, save_path)


if __name__ == '__main__':
    main()

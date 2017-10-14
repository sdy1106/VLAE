import argparse
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops
import vgg
import dataset
import multi_gpu
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epoches', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--code', action='store_true', default=False, help='if use , it will use code,\
                                                                     else , it will use onehot')
parser.add_argument('--result_path', type=str, default='./result/char1000_classifier')
parser.add_argument('--dataset', type=str, default='standard')
parser.add_argument('--char_num', type=int, default=1000, help='The number of characters')
parser.add_argument('--sample_num', type=int, default=100, help='the number of each character sampling')
parser.add_argument('--one_shot', action='store_true', default=True, help='if false , combine the train and test ')
args = parser.parse_args()

print (args)

n_z = 100
n_y = args.char_num
sample_num = args.sample_num
n_code = n_y
n_channels = 1
n_x = 4096
n_xl = 64
ngf = 128
train_test_rate = [0.8 , 0.2]
display_each_character = int(train_test_rate[1] * sample_num)
test_ny = 10
anneal_lr_freq = 200
anneal_lr_rate = 0.9
lr = args.learning_rate
batch_size = args.batch_size
train_iters = int(n_y * sample_num * train_test_rate[0]  / batch_size)
print_freq = train_iters
test_iters = int(n_y * sample_num * train_test_rate[1] / batch_size)
result_path = args.result_path
epoches = args.epoches

if not args.code:
    if args.dataset == 'hand':
        x_train, x_test, t_train, t_test = dataset.hccr_onehot_hand_64(n_y, sample_num)
        t_train, t_test = \
            utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
    elif args.dataset == 'standard':
        x_train, x_test, t_train, t_test = dataset.hccr_onehot_standard_64(n_y, sample_num)
        t_train, t_test = \
            utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
    elif args.dataset == 'casia-online':
        x_train, x_test, t_train, t_test = dataset.hccr_onehot_casia_online_64(n_y, sample_num)
        t_train, t_test = \
            utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
    elif args.dataset == 'casia-offline':
        x_train, x_test, t_train, t_test = dataset.hccr_onehot_casia_offline_64(n_y, sample_num)
        t_train, t_test = \
            utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
    elif args.dataset == 'fusion':
        x_train_s, x_test_s, t_train_s, t_test_s = dataset.hccr_onehot_standard_64(ny, sample_num,
                                                                                   train_test_rate=train_test_rate)
        t_train_s, t_test_s = utils.to_onehot(t_train_s, n_y), utils.to_onehot(t_test_s, n_y)
        x_train_con, x_test_con, t_train_con, t_test_con = dataset.hccr_onehot_casia_online_64(ny, casia_online,
                                                                                               train_test_rate=train_test_rate)
        t_train_con, t_test_con = utils.to_onehot(t_train_con, n_y), utils.to_onehot(t_test_con, n_y)
        x_train_cofr, x_test_cofr, t_train_cofr, t_test_cofr = dataset.hccr_onehot_casia_offline_reverse_64(ny,
                                                                                                            casia_offline_reverse,
                                                                                                            train_test_rate=train_test_rate)
        t_train_cofr, t_test_cofr = utils.to_onehot(t_train_cofr, n_y), utils.to_onehot(t_test_cofr, n_y)
        sample_num = sample_num + casia_offline_reverse + casia_online
        x_train = np.concatenate((x_train_s, x_train_con, x_train_cofr), axis=1)
        t_train = np.concatenate((t_train_s, t_train_con, t_train_cofr), axis=1)
        x_test = np.concatenate((x_test_s, x_test_con, x_test_cofr), axis=1)
        t_test = np.concatenate((t_test_s, t_test_con, t_test_cofr), axis=1)
    else:
        raise ValueError('Only have dataset: hand, standard, casia')
else:
    if args.dataset == 'hand':
        x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_hand_64(n_y, sample_num)
    elif args.dataset == 'standard':
        x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_standard_64(n_y, sample_num,
                                                                                 train_test_rate=train_test_rate)
    elif args.dataset == 'casia-online':
        x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_casia_online_64(n_y, sample_num,
                                                                                     train_test_rate=train_test_rate)
    elif args.dataset == 'casia-offline':
        x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_casia_offline_64(n_y, sample_num,
                                                                                      train_test_rate=train_test_rate)
    elif args.dataset == 'fusion':
        x_train_s, x_test_s, t_train_s, t_test_s, n_code = dataset.hccr_code_standard_64(n_y, sample_num,
                                                                                         train_test_rate=train_test_rate)
        x_train_con, x_test_con, t_train_con, t_test_con, n_code = dataset.hccr_code_casia_online_64(n_y, casia_online,
                                                                                                     train_test_rate=train_test_rate)
        x_train_cofr, x_test_cofr, t_train_cofr, t_test_cofr, n_code = dataset.hccr_code_casia_offline_reverse_64(
            n_y, casia_offline_reverse, train_test_rate=train_test_rate)
        sample_num = sample_num + casia_offline_reverse + casia_online
        x_train = np.concatenate((x_train_s, x_train_con, x_train_cofr), axis=1)
        t_train = np.concatenate((t_train_s, t_train_con, t_train_cofr), axis=1)
        x_test = np.concatenate((x_test_s, x_test_con, x_test_cofr), axis=1)
        t_test = np.concatenate((t_test_s, t_test_con, t_test_cofr), axis=1)
    else:
        raise ValueError('Only have dataset: hand, standard, casia')

if not args.one_shot:
    x_train = np.concatenate((x_train, x_test), axis=1)
    t_train = np.concatenate((t_train, t_test), axis=1)

x_train = x_train.reshape([-1 , n_xl , n_xl , n_channels]).astype(np.float32)
x_test = x_test.reshape([-1 , n_xl , n_xl , n_channels]).astype(np.float32)
t_train = t_train.reshape([-1 , n_y])
t_test = t_test.reshape([-1 , n_y])

#is_training = tf.placeholder(tf.bool, shape=[])
x = tf.placeholder(tf.float32 , shape=[None , n_xl , n_xl , n_channels])
y = tf.placeholder(tf.float32 , shape=[None , n_y])
optimizer = tf.train.AdamOptimizer(lr , beta1=0.5)
net = vgg.vgg_16(x , n_y)
y_ = net.outputs
net.print_paras()
net.print_layers()

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y , logits=y_)
#infer = optimizer.minimize(loss)
grads = optimizer.compute_gradients(loss)
infer = optimizer.apply_gradients(grads)
saver = tf.train.Saver(max_to_keep=10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_file = tf.train.latest_checkpoint(result_path)
    begin_epoch = 1
    if ckpt_file is not None:
        print('Restoring model from {}...'.format(ckpt_file))
        begin_epoch = int(ckpt_file.split('.')[-4]) + 1
        saver.restore(sess, ckpt_file)
    for epoch in range(begin_epoch, epoches + 1):
        if epoch % anneal_lr_freq == 0:
            learning_rate *= anneal_lr_rate
        losses = []
        accurracys = []
        x_train , t_train = utils.shuffle(x_train.reshape(-1 , n_x) , t_train)
        x_train = x_train.reshape([-1 , n_xl , n_xl , n_channels])
        x_test , t_test = utils.shuffle(x_test.reshape(-1 , n_x) , t_test)
        x_test = x_test.reshape([-1 , n_xl , n_xl , n_channels])
        for t in range(train_iters):
            iter = t + 1
            x_batch = x_train[t * batch_size : (t+1) * batch_size]
            t_batch = t_train[t * batch_size : (t+1) * batch_size]
            _ , l , yy , g = sess.run([infer , loss , y_ , grads] ,
                             feed_dict={x:x_batch , y : t_batch })
            losses.append(l)
            #print g
        for t in range(test_iters):
            iter = t + 1
            x_batch = x_test[t * batch_size: (t + 1) * batch_size]
            t_batch = t_test[t * batch_size: (t + 1) * batch_size]
            [yy] = sess.run([y_], feed_dict={x: x_batch})
            #print np.argmax(yy , axis=1) , np.argmax(t_batch , axis=1)
            accu = np.sum(np.argmax(yy , axis=1)  == np.argmax(t_batch , axis=1)) / float(yy.shape[0])
            accurracys.append(accu)

        print ('Epoch={} loss={} accuracy={}'.format(epoch, np.mean(losses) ,np.mean(accurracys)))
        losses = []
        accurracys = []






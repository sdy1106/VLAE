# -*- coding: utf-8 -*-

import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops

import dataset
import multi_gpu
import utils
from multi_gpu import FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epoches', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--code', action='store_true', default=True, help='if use , it will use code,\
                                                                     else , it will use onehot')
parser.add_argument('--result_path', type=str, default='./result/iwae-64-standard-test')
parser.add_argument('--dataset', type=str, default='standard')
parser.add_argument('--char_num', type=int, default=2939, help='The number of characters')
parser.add_argument('--sample_num', type=int, default=200, help='the number of each character sampling')
parser.add_argument('--one_shot', action='store_true', default=True, help='if false , combine the train and test ')
args = parser.parse_args()

print ('gpus:', FLAGS.num_gpus)
print (args)

# Define model parameters
n_z = 100
n_y = args.char_num
sample_num = args.sample_num
n_code = n_y
n_channels = 1
n_x = 4096
n_xl = 64
ngf = 128
train_test_rate = [0.8 , 0.2]
display_each_character = 20 #int(train_test_rate[1] * sample_num)

test_ny = 5

print_threhold = 0.5



def conv_cond_concat(x, y):
    y_ = y * tf.ones([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(y)[3]])
    return tf.concat([
        x, y_], 3)

def lrelu(input_tensor, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * input_tensor + f2 * abs(input_tensor)


@zs.reuse('decoder')
def vae(observed, n, y, is_training):
    with zs.BayesianNet(observed=observed) as decoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., n_samples=1, group_event_ndims=1)
        z = tf.reshape(z,[-1 ,n_z])
        yb = tf.reshape(y, [-1, 1, 1, n_code])

        lx_z = layers.fully_connected(tf.concat([z, y], 1), 1024, activation_fn=tf.nn.relu,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params,
                                      weights_initializer=init_ops.RandomNormal(stddev=0.02)
                                      )
        lx_z = layers.fully_connected(tf.concat([lx_z, y], 1), ngf * 8 * 4 * 4, activation_fn=tf.nn.relu,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params,
                                      weights_initializer=init_ops.RandomNormal(stddev=0.02)
                                      )
        lx_z = tf.reshape(lx_z, [-1, 4, 4, ngf * 8])

        # assert tf.shape(lx_z)[0] == n

        lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 4, 5, stride=2, activation_fn=tf.nn.relu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params,
                                       weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 2, 5, stride=2, activation_fn=tf.nn.relu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params,
                                       weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), ngf * 1, 5, stride=2, activation_fn=tf.nn.relu,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params,
                                       weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lx_z = layers.conv2d_transpose(conv_cond_concat(lx_z, yb), 1, 5, stride=2, activation_fn=None,
                                       weights_initializer=init_ops.RandomNormal(stddev=0.02))
        x_logits = tf.reshape(lx_z, [1, -1, n_x])
        x = zs.Bernoulli('x', x_logits, group_event_ndims=1)
    return decoder, x_logits

@zs.reuse('encoder')
def q_net(observed, x, y, is_training):
    with zs.BayesianNet(observed=observed) as encoder:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        x = tf.reshape(tf.to_float(x), [-1, n_xl, n_xl, 1])
        yb = tf.reshape(y, [-1, 1, 1, n_code])
        lz_x = layers.conv2d(conv_cond_concat(x, yb), ngf, 5, stride=2, activation_fn=tf.nn.relu,
                              normalizer_fn=layers.batch_norm,
                              normalizer_params=normalizer_params,
                              weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lz_x = layers.conv2d(conv_cond_concat(lz_x, yb), ngf * 2, 5, stride=2, activation_fn=tf.nn.relu,
                             normalizer_fn=layers.batch_norm,
                             normalizer_params=normalizer_params,
                             weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lz_x = layers.conv2d(conv_cond_concat(lz_x, yb), ngf * 4, 5, stride=2, activation_fn=tf.nn.relu,
                              normalizer_fn=layers.batch_norm,
                              normalizer_params=normalizer_params,
                             weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lz_x = layers.conv2d(conv_cond_concat(lz_x, yb), ngf * 8, 5, stride=2, activation_fn=tf.nn.relu,
                             normalizer_fn=layers.batch_norm,
                             normalizer_params=normalizer_params,
                             weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lz_x = layers.flatten(lz_x)
        # assert tf.shape(lz_x)[0] == tf.shape(y)[0]
        lz_x = layers.fully_connected(tf.concat([lz_x, y], 1), 1024, activation_fn=tf.nn.relu,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params,
                                      weights_initializer=init_ops.RandomNormal(stddev=0.02))
        lz_x = layers.fully_connected(tf.concat([lz_x, y], 1), 2 * n_z, activation_fn=None,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params,
                                      weights_initializer=init_ops.RandomNormal(stddev=0.02))

        mu, logstd = lz_x[:, :n_z], lz_x[:, n_z:]
        lz_x = zs.Normal('z', mu, logstd, n_samples=1, group_event_ndims=1)
    return encoder, lz_x,


if __name__ == "__main__":
    tf.set_random_seed(1234)
    np.random.seed(1234)
    if not args.code:
        if args.dataset == 'hand':
            x_train, x_test, t_train,  t_test = dataset.hccr_onehot_hand_64(n_y, sample_num)
            t_train,t_test = \
                utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
        elif args.dataset == 'standard':
            x_train, x_test, t_train, t_test = dataset.hccr_onehot_standard_64(n_y, sample_num)
            t_train, t_test = \
                utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
        elif args.dataset == 'casia':
            x_train, x_test, t_train, t_test = dataset.hccr_onehot_casia_64(n_y, sample_num)
            t_train, t_test = \
                utils.to_onehot(t_train, n_y), utils.to_onehot(t_test, n_y)
        else:
            raise ValueError('Only have dataset: hand, standard, casia')
    else:
        if args.dataset == 'hand':
            x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_hand_64(n_y, sample_num)
        elif args.dataset == 'standard':
            x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_standard_64(n_y, sample_num , train_test_rate=train_test_rate)
        elif args.dataset == 'casia-online':
            x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_casia_online_64(n_y, sample_num, train_test_rate=train_test_rate)
        elif args.dataset == 'casia-offline':
            x_train , x_test , t_train  , t_test , n_code = dataset.hccr_code_casia_offline_64(n_y , sample_num, train_test_rate=train_test_rate)
        elif args.dataset == 'casia-offline-reverse':
            x_train, x_test, t_train, t_test, n_code = dataset.hccr_code_casia_offline_reverse_64(n_y, sample_num, train_test_rate=train_test_rate)
        else:
            raise ValueError('Only have dataset: hand, standard, casia')

    if not args.one_shot:
        x_train = np.concatenate((x_train, x_test), axis=1)
        t_train = np.concatenate((t_train, t_test), axis=1)



    x_oneshot_test = np.zeros((display_each_character, n_xl, n_xl, n_channels))
    t_oneshot_test = np.zeros((display_each_character, n_code))
    t_oneshot_gen_test = t_test[:test_ny, 0, :]
    for i in range(display_each_character):
        char_index = random.randint(0, n_y - 1)
        x_oneshot_test[i, :, :, 0] = x_test[char_index, i, :, :]
        t_oneshot_test[i, :] = t_test[char_index, i, :]


    name1_index = [795, 2859, 1677]  # 刘禹锡
    name2_index = [752, 2032]  # 孔丘
    poem1_index = [220, 4, 3, 77, 6, 2057, 253, 1667, 75, 4, 3, 400, 6, 814, 253, 994]  # 山不在高
    poem2_index = [56, 50, 20, 415, 64, 4, 1088, 2325, 835, 6, 1325, 78, 545, 51, 21, 4, 1088, 862, 835]  # 学而时习之

    display1_index = [1 , 38]
    display2_index = [0 , 34]

    poem_index= poem2_index
    name_index = name2_index
    display_index = display2_index

    name_len = len(name_index)
    poem_len = len(poem_index)
    display_len = len(display_index)

    x_oneshot_poem = np.zeros((display_len, name_len ,  n_xl, n_xl, n_channels))
    t_oneshot_poem = np.zeros((display_len, name_len , n_code))
    t_oneshot_gen_peom = t_test[poem_index , 0 , :]
    for i,index in enumerate(display_index):
        x_oneshot_poem[i , : , : , : , 0] = x_test[name_index , index , : , :]
        t_oneshot_poem[i , : , :] = t_test[name_index , index , :]
    oneshot_ground_test =  x_test[poem_index , : , : , :]

    x_train_recon = x_train[:, :display_each_character, :, :]
    t_train_recon = t_train[:, :display_each_character, :]
    x_train_interp = np.concatenate((x_train[6:6+test_ny, 41, :, :], x_train[:test_ny, 5, :, :]), axis=1)
    t_train_interp = np.concatenate((t_train[6:6+test_ny, 41, :], t_train[:test_ny, 5, :]), axis=1)
    x_test = x_test[:display_each_character, :display_each_character, :, :]
    t_test = t_test[:display_each_character, :display_each_character, :]
    # x_train = np.reshape(x_train, [-1, n_xl, n_xl, n_channels])
    # t_train = np.reshape(t_train, [-1, n_code])
    x_test = np.reshape(x_test, [-1, n_xl, n_xl, n_channels])
    t_test = np.reshape(t_test, [-1, n_code])
    x_train_interp = np.reshape(x_train_interp, [-1, n_x])
    t_train_interp = np.reshape(t_train_interp, [-1, n_code])
    x_train_recon = np.reshape(x_train_recon, [-1, n_xl, n_xl, n_channels])
    t_train_recon = np.reshape(t_train_recon, [-1, n_code])

    # Define training/evaluation parameters
    epoches = args.epoches
    train_batch_size = args.batch_size * FLAGS.num_gpus
    gen_size = test_ny * display_each_character
    recon_size = test_ny * display_each_character
    train_iters = x_train.shape[0] * x_train.shape[1] // train_batch_size
    print_freq = train_iters
    test_freq = train_iters
    save_freq = train_iters
    learning_rate = args.learning_rate
    anneal_lr_freq = 200
    anneal_lr_rate = 0.9

    result_path = args.result_path
    if args.code:
        result_path += '_code'
    else:
        result_path += '_onehot'

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x_orig')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
                    tf.int32)
    x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
    x_source = tf.placeholder(tf.int32, shape=[None, n_x], name='x_source')
    oneshot_z = tf.placeholder(tf.float32, shape=(1, None, n_z), name='oneshot_z')
    interp_z = tf.placeholder(tf.float32, shape=(1, gen_size, n_z), name='oneshot_z')
    tf_ny = tf.placeholder(tf.int32, name='ny')
    code = tf.placeholder(tf.float32, shape=(None, n_code), name='code')
    code_source = tf.placeholder(tf.float32, shape=(None, n_code), name='code_source')
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)


    def build_tower_graph(x, id_):
        tower_x_orig = x_orig[id_ * tf.shape(x_orig)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(x_orig)[0] // FLAGS.num_gpus]
        tower_x_source = x_source[id_ * tf.shape(x_orig)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(x_source)[0] // FLAGS.num_gpus]
        tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
        tower_code = code[id_ * tf.shape(code)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(code)[0] // FLAGS.num_gpus]
        tower_code_source = code_source[id_ * tf.shape(code_source)[0] // FLAGS.num_gpus:
        (id_ + 1) * tf.shape(code_source)[0] // FLAGS.num_gpus]
        n = tf.shape(tower_x)[0]
        x_obs = tf.tile(tf.expand_dims(tower_x, 0), [1, 1, 1])

        def log_joint(observed):
            decoder, _ = vae(observed, n, tower_code, is_training)
            log_pz, log_px_z = decoder.local_log_prob(['z', 'x'])
            return log_pz + log_px_z

        encoder, _ = q_net(None, tower_x_source, tower_code_source, is_training)
        qz_samples, log_qz = encoder.query('z', outputs=True,
                                      local_log_prob=True)

        lower_bound = tf.reduce_mean(
            zs.iwae(log_joint, {'x': x_obs}, {'z': [qz_samples, log_qz]}, axis=0))

        grads = optimizer.compute_gradients(-lower_bound)
        return grads, lower_bound


    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                grads, lower_bound = build_tower_graph(x, i)
                tower_losses.append([lower_bound])
                tower_grads.append(grads)
    lower_bound = multi_gpu.average_losses(tower_losses)
    grads = multi_gpu.average_gradients(tower_grads)
    infer = optimizer.apply_gradients(grads)

    # eval generation
    _, eval_x_gen = vae(None, gen_size, code, is_training)
    eval_x_gen = tf.reshape(tf.sigmoid(eval_x_gen), [-1, n_xl, n_xl, n_channels])
    # eval reconstruction
    _, eval_z_gen = q_net(None, x, code, is_training)
    _, eval_x_recon = vae({'z': eval_z_gen},
                                 tf.shape(x)[0], code, is_training)
    eval_x_recon = tf.reshape(tf.sigmoid(eval_x_recon), [-1, n_xl, n_xl, n_channels])
    # eval disentangle
    disentange_z = tf.placeholder(tf.float32, shape=(None, n_z), name='disentangle_z')
    _, disentangle_x = vae({'z': disentange_z}, recon_size,
                                  code, is_training)
    disentangle_x = tf.reshape(tf.sigmoid(disentangle_x), [-1, n_xl, n_xl, n_channels])
    # eval one-shot generation
    _, eval_z_oneshot = q_net(None, x, code, is_training)
    _, eval_x_oneshot = vae({'z': oneshot_z}, tf_ny, code, is_training)
    eval_x_oneshot = tf.reshape(tf.sigmoid(eval_x_oneshot), [-1, n_xl, n_xl, n_channels])

    # eval interpolation
    _, eval_x_interp = vae({'z': interp_z}, gen_size, code, is_training)
    eval_x_interp = tf.reshape(tf.sigmoid(eval_x_interp), [-1, n_xl, n_xl, n_channels])
    params = tf.trainable_variables()

    for i in params:
        print(i.name, i.get_shape())

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='decoder') + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='encoder')

    saver = tf.train.Saver(max_to_keep=10, var_list=var_list)

    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-4]) + 1
            saver.restore(sess, ckpt_file)
            epoch  = begin_epoch


            # # train interpolation
            x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_train_interp})
            t_batch = t_train_interp
            eval_zs, _ = \
                sess.run([eval_z_gen.tensor, eval_x_recon],
                         feed_dict={x: x_batch_bin, is_training: False, code: t_batch})
            epsilon = np.linspace(0, 1, display_each_character)
            eval_zs_interp = np.array(
                [eps * eval_zs[0, 2 * i, :] + (1 - eps) * eval_zs[0, 2 * i + 1, :] for i in range(test_ny) for eps
                 in epsilon]).reshape(1, -1, n_z)
            t_batch = np.tile([t_batch[2 * i, :] for i in range(test_ny)], (1, display_each_character)).reshape(-1,
                                                                                                            n_code)
            recon_images = \
                sess.run(eval_x_interp, feed_dict={interp_z: eval_zs_interp, is_training: False, code: t_batch})
            recon_images = (recon_images > print_threhold).astype(np.float32)
            name = "interp_{}/iwae_hccr.epoch.{}.iter.{}.png".format(n_y, epoch, iter)
            name = os.path.join(result_path, name)
            utils.save_image_collections(recon_images, name, shape=(test_ny, display_each_character),
                                         scale_each=True)

            # one-shot generation
            x_batch = x_oneshot_test.reshape(-1, n_x)  # display_number*nxl*nxl*nchannel
            x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
            t_batch = t_oneshot_test
            display_x_oneshot = np.zeros((display_each_character,  test_ny+ 1, n_xl, n_xl, n_channels))

            eval_zs_oneshot = sess.run(eval_z_oneshot.tensor,
                                       feed_dict={x: x_batch_bin, is_training: False, code: t_batch})
            for i in range(display_each_character):
                display_x_oneshot[i, 0, :, :, :] = x_batch[i, :].reshape(-1, n_xl, n_xl, n_channels)
                tmp_z = np.zeros((1, test_ny, n_z))
                for j in range(test_ny):
                    # print (np.shape(tmp_z) ,np.shape(eval_zs_oneshot))
                    tmp_z[0, j, :] = eval_zs_oneshot[0, i, :]
                tmp_x = sess.run(eval_x_oneshot,
                                 feed_dict={oneshot_z: tmp_z, tf_ny: test_ny, code: t_oneshot_gen_test,
                                            is_training: False})
                # print (np.shape(tmp_x))
                display_x_oneshot[i, 1:, :, :, :] = tmp_x
            display_x_oneshot = np.reshape(display_x_oneshot, (-1, n_xl, n_xl, n_channels))
            name = "oneshot_{}/iwae_hccr.epoch.{}.png".format(n_y,
                                                                      epoch)
            name = os.path.join(
                result_path, name)
            display_x_oneshot = (display_x_oneshot > print_threhold).astype(np.float32)
            utils.save_image_collections(display_x_oneshot,
                                         name, shape=(display_each_character, test_ny + 1),
                                         scale_each=True)

            # one-shot peom generation
            x_batch = x_oneshot_poem.reshape(-1 , n_x)   # [display , name_len , n_xl , n_xl]
            x_batch_bin = sess.run(x_bin , feed_dict={x_orig:x_batch})
            x_batch_bin = x_batch_bin.reshape(display_len , name_len , n_x)
            t_batch = t_oneshot_poem
            display_x_oneshot = np.zeros((display_len * 2 , poem_len + name_len , n_xl , n_xl , n_channels))

            #print ('######', np.shape(x_batch_bin), np.shape(t_batch))
            for i,index in enumerate(display_index):
                #偶数列生成的图
                #print ('*****' , np.shape(x_batch_bin[i , : , :]) , np.shape(t_oneshot_poem[: , 0 , :]))
                zs_oneshot = sess.run(eval_z_oneshot.tensor ,feed_dict={x: x_batch_bin[i , : , :] , is_training:False , code:t_oneshot_poem[0 , : , :]}) # z[name_len , nz]

                zs_oneshot = np.mean(zs_oneshot.reshape(-1 , n_z) , axis=0) #[n_z]
                zs_oneshot = np.tile(np.expand_dims(zs_oneshot , 0) , (poem_len , 1))
                #print ('@@@@' , np.shape(zs_oneshot) , np.shape(t_oneshot_gen_peom))
                tmp_x = sess.run(eval_x_oneshot ,
                                 feed_dict={oneshot_z:np.expand_dims(zs_oneshot , 0) , tf_ny:poem_len , code:t_oneshot_gen_peom,
                                            is_training:False})
                name = np.ones((name_len , n_xl , n_xl , n_channels))
                name = x_batch_bin[i].reshape(-1 , n_xl , n_xl , n_channels)
                display_x_oneshot[2*i , :name_len , : , : , :] = name
                display_x_oneshot[2*i , name_len: , : , : , :] = tmp_x.reshape(-1 , n_xl , n_xl , n_channels)

                #奇数列是原图
                display_x_oneshot[2*i+1 , :name_len , : , : , :] = name
                display_x_oneshot[2*i+1 , name_len: , : , : , :] = oneshot_ground_test[: , index , : , :].reshape(-1 , n_xl , n_xl , n_channels)

            display_x_oneshot = np.reshape(display_x_oneshot, (-1, n_xl, n_xl, n_channels))
            name = "peom_{}/iwae_hccr.epoch.{}.poem.{}.png".format(n_y,
                                                                      epoch, poem_len)
            name = os.path.join(
                result_path, name)
            display_x_oneshot = (display_x_oneshot > print_threhold).astype(np.float32)
            utils.save_image_collections(display_x_oneshot,
                                         name, shape=(display_len*2, poem_len + name_len),
                                         scale_each=True)


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
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--result_path', type=str, default='./result/VLAE')
parser.add_argument('--dataset', type=str, default='standard')
parser.add_argument('--char_num', type=int, default=1000, help='The number of characters')
parser.add_argument('--sample_num', type=int, default=100, help='the number of each character sampling')
parser.add_argument('--pairwise', default=False, action='store_true')
parser.add_argument('--char_dim', type=int, default=20)
parser.add_argument('--font_dim', type=int, default=20)
args = parser.parse_args()

print ('gpus:', FLAGS.num_gpus)
print (args)

n_y = args.char_num
n_font = args.sample_num
n_xl = 64
n_x = 64 ** 2
ngf = 64
char_dim = args.char_dim
font_dim = args.font_dim
n_z = char_dim + font_dim
train_ratio = 0.9


def lrelu(input_tensor, leak=0.1, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * input_tensor + f2 * abs(input_tensor)


@zs.reuse('encoder')
def q_net(observed, x, is_training):
	with zs.BayesianNet(observed=observed) as encoder:
		normalizer_params = {'is_training': is_training,
							 'updates_collections': None}
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
		latent0_mean = layers.fully_connected(ladder0, font_dim / 2, activation_fn=tf.identity)
		latent0_std = layers.fully_connected(ladder0, font_dim / 2, activation_fn=tf.sigmoid)

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
		latent1_mean = layers.fully_connected(ladder1, font_dim / 2, activation_fn=tf.identity)
		latent1_std = layers.fully_connected(ladder1, font_dim / 2, activation_fn=tf.sigmoid)

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
		latent2_mean = layers.fully_connected(ladder2, char_dim / 2, activation_fn=tf.identity)
		latent2_std = layers.fully_connected(ladder2, char_dim / 2, activation_fn=tf.sigmoid)

		inference2 = layers.conv2d(inference1, ngf * 4, 4, stride=1, activation_fn=lrelu,
								   normalizer_fn=layers.batch_norm,
								   normalizer_params=normalizer_params,
								   weights_initializer=init_ops.RandomNormal(stddev=0.02))
		inference2 = layers.conv2d(inference2, ngf * 4, 4, stride=2, activation_fn=lrelu,
								   normalizer_fn=layers.batch_norm,
								   normalizer_params=normalizer_params,
								   weights_initializer=init_ops.RandomNormal(stddev=0.02))
		inference2 = layers.conv2d(inference2, ngf * 8, 4, stride=1, activation_fn=lrelu,
								   normalizer_fn=layers.batch_norm,
								   normalizer_params=normalizer_params,
								   weights_initializer=init_ops.RandomNormal(stddev=0.02))

		ladder3 = layers.conv2d(inference2, ngf * 8, 4, stride=2, activation_fn=lrelu,
								normalizer_fn=layers.batch_norm,
								normalizer_params=normalizer_params,
								weights_initializer=init_ops.RandomNormal(stddev=0.02))

		ladder3 = layers.conv2d(ladder3, ngf * 8, 4, stride=1, activation_fn=lrelu,
								normalizer_fn=layers.batch_norm,
								normalizer_params=normalizer_params,
								weights_initializer=init_ops.RandomNormal(stddev=0.02))

		ladder3 = layers.conv2d(ladder3, ngf * 16, 4, stride=2, activation_fn=lrelu,
								normalizer_fn=layers.batch_norm,
								normalizer_params=normalizer_params,
								weights_initializer=init_ops.RandomNormal(stddev=0.02))
		ladder3 = layers.flatten(ladder3)
		latent3_mean = layers.fully_connected(ladder3, char_dim / 2, activation_fn=tf.identity)
		latent3_std = layers.fully_connected(ladder3, char_dim / 2, activation_fn=tf.sigmoid)
		mu_font = tf.concat([latent0_mean, latent1_mean], 1)
		std_font = tf.concat([latent0_std, latent1_std], 1)
		mu_char = tf.concat([latent2_mean, latent3_mean], 1)
		std_char = tf.concat([latent2_std, latent3_std], 1)
		z_char = zs.Normal('z_char', mean=mu_char, std=std_char, n_samples=1, group_event_ndims=1)
		z_font = zs.Normal('z_font', mean=mu_font, std=std_font, n_samples=1, group_event_ndims=1)
		return encoder, z_font, z_char


@zs.reuse('decoder')
def VLAE(observed, n, is_training):
	with zs.BayesianNet(observed=observed) as decoder:
		normalizer_params = {'is_training': is_training,
							 'updates_collections': None}
		z_char_mean = tf.zeros([n, char_dim])
		z_char = zs.Normal('z_char', z_char_mean, std=1., n_samples=1, group_event_ndims=1)
		z_char = tf.reshape(z_char, [-1, char_dim])
		z_font_mean = tf.zeros([n, font_dim])
		z_font = zs.Normal('z_font', z_font_mean, std=1., n_samples=1, group_event_ndims=1)
		z_font = tf.reshape(z_font, [-1, font_dim])
		latent3 = z_char[:, :char_dim / 2]
		latent2 = z_char[:, char_dim / 2:]
		latent1 = z_font[:, font_dim / 2:]
		latent0 = z_font[:, :font_dim / 2]
		ladder3 = layers.fully_connected(latent3, ngf * 16 * 1 * 1, activation_fn=tf.nn.relu,
										 normalizer_fn=layers.batch_norm,
										 normalizer_params=normalizer_params,
										 weights_initializer=init_ops.RandomNormal(stddev=0.02))
		ladder3 = tf.reshape(ladder3, [-1, 1, 1, ngf * 16])
		ladder3 = layers.conv2d_transpose(ladder3, ngf * 8, 4, stride=2, activation_fn=tf.nn.relu,
										  normalizer_fn=layers.batch_norm,
										  normalizer_params=normalizer_params,
										  weights_initializer=init_ops.RandomNormal(stddev=0.02))
		ladder3 = layers.conv2d_transpose(ladder3, ngf * 8, 4, stride=1, activation_fn=tf.nn.relu,
										  normalizer_fn=layers.batch_norm,
										  normalizer_params=normalizer_params,
										  weights_initializer=init_ops.RandomNormal(stddev=0.02))
		inference2 = layers.conv2d_transpose(ladder3, ngf * 4, 4, stride=2, activation_fn=tf.nn.relu,
										  normalizer_fn=layers.batch_norm,
										  normalizer_params=normalizer_params,
										  weights_initializer=init_ops.RandomNormal(stddev=0.02))

		ladder2 = layers.fully_connected(latent2, ngf * 8 * 4 * 4, activation_fn=tf.nn.relu,
										 normalizer_fn=layers.batch_norm,
										 normalizer_params=normalizer_params,
										 weights_initializer=init_ops.RandomNormal(stddev=0.02))
		ladder2 = tf.reshape(ladder2, [-1, 4, 4, ngf * 8])

		ladder2 = tf.concat([ladder2, inference2], 3)
		ladder2 = layers.conv2d_transpose(ladder2, ngf * 8, 4, stride=1, activation_fn=tf.nn.relu,
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

		ladder1 = layers.fully_connected(latent1, ngf * 4 * 8 * 8, activation_fn=tf.nn.relu,
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

		ladder0 = layers.fully_connected(latent0, ngf * 1 * 32 * 32, activation_fn=tf.nn.relu,
										 normalizer_fn=layers.batch_norm,
										 normalizer_params=normalizer_params,
										 weights_initializer=init_ops.RandomNormal(stddev=0.02))
		ladder0 = tf.reshape(ladder0, [-1, 32, 32, ngf * 1])
		ladder0 = tf.concat([ladder0, inference0], 3)

		ladder0 = layers.conv2d_transpose(ladder0, ngf, 4, stride=1, activation_fn=tf.nn.relu,
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
		X = np.load('/home/danyang/mfs/data/hccr/image_1000x20x64x64_stand.npy')
	elif args.dataset == 'casia-offline':
		X = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-offline.npy')
	elif args.dataset == 'casia-online':
		X = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-online.npy')
	else:
		print('Unknown Dataset!')
		os._exit(-1)
	train_x = X[:int(train_ratio * n_y), :int(train_ratio * n_font), :, :]
	test_x_font = X[:int(train_ratio * n_y), int(train_ratio * n_font):n_font, :, :]
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
	x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
	x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig),
					tf.int32)
	x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
	font_source = tf.placeholder(tf.int32, shape=[None, n_x], name='font_source')
	char_source = tf.placeholder(tf.int32, shape=[None, n_x], name='char_source')
	learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
	optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

	def build_tower_graph(id_):
		tower_x = x[id_ * tf.shape(x)[0] // FLAGS.num_gpus:
		(id_ + 1) * tf.shape(x)[0] // FLAGS.num_gpus]
		tower_font_source = font_source[id_ * tf.shape(font_source)[0] // FLAGS.num_gpus:
		(id_ + 1) * tf.shape(font_source)[0] // FLAGS.num_gpus]
		tower_char_source = char_source[id_ * tf.shape(char_source)[0] // FLAGS.num_gpus:
		(id_ + 1) * tf.shape(char_source)[0] // FLAGS.num_gpus]
		n = tf.shape(tower_x)[0]
		x_obs = tf.tile(tf.expand_dims(tower_x, 0), [1, 1, 1])

		def log_joint(observed):
			decoder, _, = VLAE(observed, n, is_training)
			log_pz_font, log_pz_char, log_px_z = decoder.local_log_prob(['z_font', 'z_char', 'x'])
			return log_pz_font + log_pz_char + log_px_z

		encoder, _, _ = q_net(None, tower_font_source, is_training)
		qz_samples_font, log_qz_font = encoder.query('z_font', outputs=True,
													 local_log_prob=True)
		encoder, _, _ = q_net(None, tower_char_source, is_training)
		qz_samples_char, log_qz_char = encoder.query('z_char', outputs=True,
													 local_log_prob=True)
		lower_bound = tf.reduce_mean(
			zs.iwae(log_joint, {'x': x_obs},
					{'z_font': [qz_samples_font, log_qz_font], 'z_char': [qz_samples_char, log_qz_char]}, axis=0))

		grads = optimizer.compute_gradients(-lower_bound)
		return grads, lower_bound

	tower_losses = []
	tower_grads = []
	for i in range(FLAGS.num_gpus):
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i):
				grads, lower_bound = build_tower_graph(i)
				tower_losses.append([lower_bound])
				tower_grads.append(grads)
	lower_bound = multi_gpu.average_losses(tower_losses)
	grads = multi_gpu.average_gradients(tower_grads)
	infer = optimizer.apply_gradients(grads)

	params = tf.trainable_variables()

	for i in params:
		print(i.name, i.get_shape())
	saver = tf.train.Saver(max_to_keep=10,
						   var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
													  scope='encoder') + \
									tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
													  scope='decoder'))

	_, z_font, _ = q_net(None, font_source, is_training)
	_, _, z_char = q_net(None, char_source, is_training)
	_, x_gen = VLAE({'z_font': z_font, 'z_char': z_char}, tf.shape(char_source)[0], is_training)
	x_gen = tf.reshape(tf.sigmoid(x_gen), [-1, n_xl, n_xl, 1])

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
			x_train = train_x.reshape(-1, n_x)
			np.random.shuffle(x_train)
			x_train = x_train[:min(train_x.shape[0] * train_x.shape[1], 10000)]
			x_train = sess.run(x_bin, feed_dict={x_orig: x_train})
			if args.pairwise:
				x_font_train = np.tile(np.expand_dims(
					np.array([X[np.random.randint(0, train_x.shape[0] - 1), i, :, :] for i in range(train_x.shape[1])]),
					0),
					(train_x.shape[0], 1, 1, 1))
				x_char_train = np.tile(np.expand_dims(
					np.array([X[i, np.random.randint(0, train_x.shape[1] - 1), :, :] for i in range(train_x.shape[0])]),
					1),
					(1, train_x.shape[1], 1, 1))
				x_pair = np.concatenate(
					(train_x.reshape(-1, n_x),
					 x_char_train.reshape(-1, n_x),
					 x_font_train.reshape(-1, n_x)), 1
				)
				np.random.shuffle(x_pair)
				x_train = x_pair[:min(train_x.shape[0] * train_x.shape[1], 10000)]
				#x_train = sess.run(x_bin, feed_dict={x_orig: x_train})
			np.random.shuffle(x_train)
			for i in range(train_iters):
				if args.pairwise:
					_, lb = sess.run(
						[infer, lower_bound],
						feed_dict={x: x_train[i * train_batch_size: (i + 1) * train_batch_size, :n_x],
								   char_source: x_train[i * train_batch_size: (i + 1) * train_batch_size, n_x:2 * n_x],
								   font_source: x_train[i * train_batch_size: (i + 1) * train_batch_size, 2 * n_x:],
								   learning_rate_ph: learning_rate,
								   is_training: True})
				else:
					_, lb = sess.run(
						[infer, lower_bound],
						feed_dict={x: x_train[i * train_batch_size: (i + 1) * train_batch_size],
								   char_source: x_train[i * train_batch_size: (i + 1) * train_batch_size],
								   font_source: x_train[i * train_batch_size: (i + 1) * train_batch_size],
								   learning_rate_ph: learning_rate,
								   is_training: True})
				lower_bounds.append(lb)
			print('Epoch={} ({:.3f}s/epoch): '
				  'Lower Bound = {}'.
				  format(epoch, (time.time() + time_train),
						 np.mean(lower_bounds)))

			# train reconstruction
			gen_images = sess.run(x_gen, feed_dict={char_source: train_x[:10, :10, :, :].reshape(-1, n_x),
													font_source: train_x[:10, :10, :, :].reshape(-1, n_x),
													is_training: False})

			name = "train_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			name = os.path.join(result_path, name)
			utils.save_contrast_image_collections(train_x[:10, :10, :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
												  name, shape=(10, 20),
												  scale_each=True)

			# new font reconstruction
			char_index = np.arange(test_x_font.shape[0])
			font_index = np.arange(test_x_font.shape[1])
			np.random.shuffle(char_index)
			np.random.shuffle(font_index)
			gen_images = sess.run(x_gen, feed_dict={
				char_source: test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
				font_source: test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
				is_training: False})
			name = "test_font_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			name = os.path.join(result_path, name)
			utils.save_contrast_image_collections(
				test_x_font[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
				name, shape=(10, 20),
				scale_each=True)

			# new char reconstruction
			char_index = np.arange(test_x_char.shape[0])
			font_index = np.arange(test_x_char.shape[1])
			np.random.shuffle(char_index)
			np.random.shuffle(font_index)
			gen_images = sess.run(x_gen, feed_dict={
				char_source: test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
				font_source: test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
				is_training: False})

			name = "test_char_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			name = os.path.join(result_path, name)
			utils.save_contrast_image_collections(
				test_x_char[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
				name, shape=(10, 20),
				scale_each=True)

			# never seen reconstruction
			char_index = np.arange(test_x.shape[0])
			font_index = np.arange(test_x.shape[1])
			np.random.shuffle(char_index)
			np.random.shuffle(font_index)
			gen_images = sess.run(x_gen,
								  feed_dict={
									  char_source: test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
									  font_source: test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_x),
									  is_training: False})

			name = "test_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			name = os.path.join(result_path, name)
			utils.save_contrast_image_collections(
				test_x[char_index[:10], :, :, :][:, font_index[:10], :, :].reshape(-1, n_xl, n_xl, 1), gen_images,
				name, shape=(10, 20),
				scale_each=True)

			# one shot font generation
			# font_index = np.arange(test_x_font.shape[1])
			# np.random.shuffle(font_index)
			# test_x_font_feed = np.tile(np.expand_dims(
			# 	np.array(
			# 		[test_x_font[np.random.randint(test_x_font.shape[0] - 1), font_index[i], :, :] for i in range(10)]),
			# 	0),
			# 	(10, 1, 1, 1))
			# gen_images = sess.run(x_gen, feed_dict={char_source: train_x[:10, :10, :, :].reshape(-1, n_x),
			# 										font_source: test_x_font_feed[:10, :10, :, :].reshape(-1, n_x),
			# 										is_training: False})
			# images = np.concatenate([test_x_font_feed[0].reshape(-1, n_xl, n_xl, 1), gen_images], 0)
            #
			# name = "one_shot_font_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			# name = os.path.join(result_path, name)
			# utils.save_image_collections(images, name, shape=(11, 10),
			# 							 scale_each=True)

			# one shot char generation
			char_index = np.arange(test_x_char.shape[0])
			np.random.shuffle(char_index)
			test_x_char_feed = np.tile(np.expand_dims(
				np.array(
					[test_x_char[char_index[i], np.random.randint(test_x_char.shape[1] - 1), :, :] for i in range(10)]),
				1),
				(1, 10, 1, 1))
			gen_images = sess.run(x_gen, feed_dict={char_source: test_x_char_feed[:10, :10, :, :].reshape(-1, n_x),
													font_source: train_x[:10, :10, :, :].reshape(-1, n_x),
													is_training: False})
			name = "one_shot_char_{}/VLAE_hccr.epoch.{}.png".format(n_y, epoch)
			name = os.path.join(result_path, name)
			images = np.zeros((110, 64, 64, 1))
			for i in range(10):
				images[i * 11] = np.expand_dims(test_x_char_feed[i, 0, :, :], 2)
				images[i * 11 + 1:(i + 1) * 11] = gen_images[i * 10:(i + 1) * 10]
			utils.save_image_collections(images, name, shape=(10, 11),
										 scale_each=True)

			save_path = "VLAE.epoch.{}.ckpt".format(epoch)
			save_path = os.path.join(result_path, save_path)
			saver.save(sess, save_path)


if __name__ == '__main__':
	main()

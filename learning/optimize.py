import tensorflow as tf

from config import *


def optimize(global_step,total_loss):

	num_batches_per_epoch = cfg.nee / cfg.batchsize
	decay_steps = int(num_batches_per_epoch * cfg.ned)

	lr = tf.train.exponential_decay(cfg.lr, global_step,decay_steps,cfg.lrdf,staircase=True)

	opt = tf.train.GradientDescentOptimizer(lr)

	grads = opt.compute_gradients(total_loss)

	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	with tf.control_dependencies([apply_gradient_op]):
		train_op = tf.no_op(name='train')
		return train_op
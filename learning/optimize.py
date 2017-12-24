import tensorflow as tf

from config import *


def optimize(global_step,total_loss):

	num_batches_per_epoch = cfg.nee / cfg.batchsize
	decay_steps = int(num_batches_per_epoch * cfg.ned)

	lr = tf.train.exponential_decay(cfg.lr, global_step,decay_steps,cfg.lrdf,staircase=True)

	opt = tf.train.AdamOptimizer()

	grads = opt.compute_gradients(total_loss)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)
			
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	variable_averages = tf.train.ExponentialMovingAverage(cfg.mad, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())


	with tf.control_dependencies([apply_gradient_op,variables_averages_op]):
		train_op = tf.no_op(name='train')
		return train_op
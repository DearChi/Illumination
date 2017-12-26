import tensorflow as tf
from config import *

class LinearNetwork:
	def __init__(self, input_value, is_training):
		self.value = input_value
		self.is_training = is_training
		self.layer_index = 0
		self.layer_name = 'root'

	def input(self,x):
		self.value = x


	def scope_name(self):
		return self.layer_name+'_%02d'%self.layer_index

	def output(self):
		return self.value

	def variable(self, name, shape, initializer=None, stddev=None, wdecay=None, trainable = True):

		if initializer == None:
			initializer=tf.truncated_normal_initializer(stddev=stddev)

		var = tf.get_variable(name, shape, initializer=initializer)

		if wdecay != None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def layer(self,name):
		self.layer_index += 1
		self.layer_name = name
		return self

	def conv2d(self, kshape, strides = [1,2,2,1],padding='SAME'):
		with tf.variable_scope(self.scope_name()):
			kernel = self.variable('kernel', kshape, stddev=0.05)
			biases = self.variable('biases', kshape[-1:-1], tf.constant_initializer(0.01))
		self.value = tf.add(tf.nn.conv2d(self.value, kernel,strides,padding), biases)
		return self

	def matmul(self, shape):
		if len(self.value.get_shape()) > 2:
			self.value = tf.reshape(self.value,[int(self.value.get_shape()[0]),-1])
		wshape = [int(self.value.get_shape()[1])] + shape

		with tf.variable_scope(self.scope_name()):
			weight = self.variable('weight', wshape, stddev=0.03)
			biases = self.variable('biases',  shape, tf.constant_initializer(0.01))
		self.value = tf.add(tf.matmul(self.value, weight), biases)
		return self

	def bn(self):
		shape = self.value.get_shape()  

		params_shape = shape[-1:]
		axis = list(range(len(shape) - 1))  

		with tf.variable_scope(self.scope_name()):

			beta  = self.variable( 'beta', params_shape, tf.zeros_initializer())  
			gamma = self.variable('gamma', params_shape, tf.ones_initializer())  

			moving_mean = self.variable('moving_mean', params_shape, tf.zeros_initializer(), trainable=False)  
			moving_var  = self.variable( 'moving_var', params_shape,  tf.ones_initializer(), trainable=False)  

			mean, variance = tf.nn.moments(self.value, axis)  
			ema = tf.train.ExponentialMovingAverage(cfg.bnmad)
			vop = ema.apply([mean,variance])
			if self.is_training:
				with tf.control_dependencies([vop]):
					self.value = tf.nn.batch_normalization(self.value, mean, variance, beta, gamma, 0.01)
			else:
				average_mean 	= ema.average(mean)
				average_variance = ema.average(variance)
				self.value = tf.nn.batch_normalization(self.value, average_mean, average_variance, beta, gamma, 0.01)
		return self

	def relu(self):
		self.value = tf.nn.relu(self.value)
		return self

def inference(image, is_training=True):
	net = LinearNetwork(image,is_training)

	net.layer('conv').conv2d([7,7,3,64]
		).bn().relu()

	net.layer('conv').conv2d([5,5,64,128]
		).bn().relu()

	net.layer('conv').conv2d([3,3,128,256]
		).bn().relu()

	net.layer('conv').conv2d([3,3,256,256],[1,1,1,1]
		).bn().relu()

	net.layer('conv').conv2d([3,3,256,256]
		).bn().relu()

	net.layer('conv').conv2d([3,3,256,256],[1,1,1,1]
		).bn().relu()

	net.layer('fcon').matmul([2048]
		).bn().relu()

	net.layer('fcon').matmul([512]
		).bn().relu()

	net.layer('fcon').matmul([ 128]
		).relu()

	net.layer('fcon').matmul([  27]
		)

	return net.output()


def get_total_loss(reference, prediction):
	mse = tf.reduce_mean(
		tf.square(
			tf.subtract(prediction,reference)
		)
	)
	tf.summary.histogram('00SH_COEFFICIENT' + '/reference', reference)
	tf.summary.histogram('00SH_COEFFICIENT' + '/prediction', prediction)

	return mse

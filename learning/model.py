import tensorflow as tf

from config import *

def layer(name,layers,input,shape,strides=[1,2,2,1],padding='SAME',outshape=None,summary=False):

	with tf.variable_scope(name):
		result = input
		for layer in layers:
			if layer == 'c':
				kernel = tf.get_variable('kernel', shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
				biases = tf.get_variable('biases', shape[-1:-1], initializer=tf.constant_initializer(0.1))
				result = tf.add(tf.nn.conv2d(result,kernel,strides=strides,padding=padding),biases)
				
			elif layer == 'r':
				result = tf.nn.relu(result)

			elif layer == 't':
				result = tf.nn.tanh(result)
			
			elif layer == 'b':
				mean, var = tf.nn.moments(result,[0,1,2])
				result = tf.nn.batch_normalization(result,mean,var,0,1,0.01)
			
			elif layer == 'm':
				weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.03))
				biases = tf.get_variable('biases', shape[-1:-1], initializer=tf.constant_initializer(0))
				result = tf.add(tf.matmul(result,weight),biases)

			elif layer == 'd':
				kernel = tf.get_variable('filter',shape,initializer = tf.truncated_normal_initializer(stddev=0.05))
				biases = tf.get_variable('biases',outshape[-1:-1],initializer = tf.constant_initializer(0.1))
				result = tf.add(tf.nn.conv2d_transpose(result,kernel,outshape,strides),biases)

			elif layer == 'e':
				result = tf.reshape(result,outshape)
				shape[0] = int(result.get_shape()[1])

			elif layer == 'a':
				shape = result.get_shape()
				h,w,c = int(shape[1]),int(shape[2]),int(shape[3])
				kernel = tf.get_variable('kforloss', shape=[3,3,c,3],initializer=tf.truncated_normal_initializer(stddev=0.02))
				biases = tf.get_variable('bforloss', [3], initializer=tf.constant_initializer(0))
				output = tf.add(tf.nn.conv2d(result,kernel,strides=[1,1,1,1],padding='SAME'),biases)
				tf.add_to_collection('pathout',output)

		return result

def inference_former(images):
	
	conv_1 = layer('layer-1', 'cbr', images, [7,7,3,64])
	conv_2 = layer('layer-2', 'cbr', conv_1, [5,5,64,128])
	conv_3 = layer('layer-3', 'cbr', conv_2, [3,3,128,256])
	conv_4 = layer('layer-4', 'cbr', conv_3, [3,3,256,256], strides=[1,1,1,1])
	conv_5 = layer('layer-5', 'cbr', conv_4, [3,3,256,256])
	conv_6 = layer('layer-6', 'cbr', conv_5, [3,3,256,256], strides=[1,1,1,1])
	conv_7 = layer('layer-7', 'cbr', conv_6, [3,3,256,256])

	fcon_8 = layer('layer-8', 'emr', conv_7, [-1,  2048], outshape=[cfg.batchsize,-1])
	fcon_9 = layer('layer-9', 'mbr',  fcon_8, [2048, 512])
	fcon_a = layer('layer-a', 'mt',  fcon_9, [ 512, 128])
	result = layer('layer-b', 'mt',  fcon_a, [ 128,  27])
	
	return result

def get_total_loss(reference, prediction):
	mse = tf.reduce_mean(
		tf.square(
			tf.subtract(prediction,reference)
		)
	)
	tf.summary.histogram('00SH_COEFFICIENT' + '/reference', reference)
	tf.summary.histogram('00SH_COEFFICIENT' + '/prediction', prediction)

	return mse

def variable(name, shape, stddev, wd = None):
  var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

  return var

def inference(image):
	with tf.variable_scope('conv_1') as scope:
		kernel = variable('kernel',[7,7,3,64],0.05)
		biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
		_conv1 = tf.add(tf.nn.conv2d(image,kernel, strides=[1,2,2,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv1,[0,1,2])
		conv_1 = tf.nn.relu(tf.nn.batch_normalization(_conv1,mean,var,0,1,0.01))

	with tf.variable_scope('conv2') as scope:
		kernel = variable('kernel',[5,5,64,128],0.05)
		biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1))
		_conv2 = tf.add(tf.nn.conv2d(conv_1,kernel, strides=[1,2,2,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv2,[0,1,2])
		conv_2 = tf.nn.relu(tf.nn.batch_normalization(_conv2,mean,var,0,1,0.01))

	with tf.variable_scope('conv3') as scope:
		kernel = variable('kernel',[3,3,128,256],0.05)
		biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
		_conv3 = tf.add(tf.nn.conv2d(conv_2,kernel, strides=[1,2,2,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv3,[0,1,2])
		conv_3 = tf.nn.relu(tf.nn.batch_normalization(_conv3,mean,var,0,1,0.01))

	with tf.variable_scope('conv4') as scope:
		kernel = variable('kernel',[3,3,256,256],0.05)
		biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
		_conv4 = tf.add(tf.nn.conv2d(conv_3,kernel, strides=[1,1,1,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv4,[0,1,2])
		conv_4 = tf.nn.relu(tf.nn.batch_normalization(_conv4,mean,var,0,1,0.01))

	with tf.variable_scope('conv5') as scope:
		kernel = variable('kernel',[3,3,256,256],0.05)
		biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
		_conv5 = tf.add(tf.nn.conv2d(conv_4,kernel, strides=[1,2,2,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv5,[0,1,2])
		conv_5 = tf.nn.relu(tf.nn.batch_normalization(_conv5,mean,var,0,1,0.01))

	with tf.variable_scope('conv6') as scope:
		kernel = variable('kernel',[3,3,256,256],0.05)
		biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
		_conv6 = tf.add(tf.nn.conv2d(conv_5,kernel, strides=[1,1,1,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv6,[0,1,2])
		conv_6 = tf.nn.relu(tf.nn.batch_normalization(_conv6,mean,var,0,1,0.01))

	with tf.variable_scope('conv7') as scope:
		kernel = variable('kernel',[3,3,256,256],0.05)
		biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1))
		_conv7 = tf.add(tf.nn.conv2d(conv_6,kernel, strides=[1,2,2,1], padding='SAME'), biases)
		mean, var = tf.nn.moments(_conv7,[0,1,2])
		conv_7 = tf.nn.relu(tf.nn.batch_normalization(_conv7,mean,var,0,1,0.01))

	reshape = tf.reshape(conv_7,[cfg.batchsize,-1])
	dim = int(reshape.get_shape()[1])

	with tf.variable_scope('fcon8') as scope:
		weight = tf.get_variable('weight', [dim,2048], initializer=tf.truncated_normal_initializer(stddev=0.03))
		biases = tf.get_variable('biases', [2048], initializer=tf.constant_initializer(0))
		_fcon8 = tf.add(tf.matmul(reshape,weight),biases)
		fcon_8 = tf.nn.relu(_fcon8)

	with tf.variable_scope('fcon9') as scope:
		weight = tf.get_variable('weight', [2048,512], initializer=tf.truncated_normal_initializer(stddev=0.03))
		biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0))
		_fcon9 = tf.add(tf.matmul(fcon_8,weight),biases)
		mean, var = tf.nn.moments(_fcon9,[0])
		fcon_9 = tf.nn.relu(tf.nn.batch_normalization(_fcon9,mean,var,0,1,0.01))

	with tf.variable_scope('fcona') as scope:
		weight = tf.get_variable('weight', [512,128], initializer=tf.truncated_normal_initializer(stddev=0.03))
		biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0))
		_fcona = tf.add(tf.matmul(fcon_9,weight),biases)
		fcon_a = tf.nn.relu(_fcona)

	with tf.variable_scope('fconb') as scope:
		weight = tf.get_variable('weight', [128,27], initializer=tf.truncated_normal_initializer(stddev=0.03))
		biases = tf.get_variable('biases', [27], initializer=tf.constant_initializer(0))
		result = tf.add(tf.matmul(fcon_a,weight),biases)

	return result
"""
------operator with shape------
  operator  shape
c:    conv  shape of kernerl
d:  deconv  shape of kernerl
m:  matmul  shape of kernerl
p: pooling  shape of kernerl
a: pathout  shape of 

------operator with outshape------
  operator  outshape
e: reshape  shape after reshapping (always using 'em')

------appendix operator------
b: batch_norm
r: relu
t: tanh

"""
import tensorflow as tf

from config import *

def layer(name,layers,input,shape,strides=[1,2,2,1],padding='SAME',outshape=None,summary=False):

	with tf.variable_scope(name):
		result = input
		for layer in layers:
			if layer == 'c':
				kernel = tf.get_variable('kernel', shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
				biases = tf.get_variable('biases', shape[-1:-1], initializer=tf.constant_initializer(0.001))
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
				biases = tf.get_variable('biases',outshape[-1:-1],initializer = tf.constant_initializer(0.001))
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

def inference(images):
	
	conv_1 = layer('layer-1', 'cbr', images, [7,7,3,64])
	conv_2 = layer('layer-2', 'cbr', conv_1, [5,5,64,128])
	conv_3 = layer('layer-3', 'cbr', conv_2, [3,3,128,256])
	conv_4 = layer('layer-4', 'cbr', conv_3, [3,3,256,256], strides=[1,1,1,1])
	conv_5 = layer('layer-5', 'cbr', conv_4, [3,3,256,256])
	conv_6 = layer('layer-6', 'cbr', conv_5, [3,3,256,256], strides=[1,1,1,1])
	conv_7 = layer('layer-7', 'cbr', conv_6, [3,3,256,256])

	fcon_8 = layer('layer-8', 'emr', conv_7, [-1,  2048], outshape=[cfg.batchsize,-1])
	fcon_9 = layer('layer-9', 'mr',  fcon_8, [2048, 512])
	fcon_a = layer('layer-a', 'mr',  fcon_9, [ 512, 128])
	result = layer('layer-b', 'mr',  fcon_a, [ 128,  27])
	
	return result

def get_total_loss(reference, prediction):
	mse = tf.reduce_mean(
		tf.square(
			tf.subtract(tf.divide(prediction,256.0),tf.divide(reference,256.0))
		)
	)
	return mse
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
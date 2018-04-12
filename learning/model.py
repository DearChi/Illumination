import tensorflow as tf
from config import *
from pretrained import vgg
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  

class LinearNetwork:
	def __init__(self, input_value, is_training, net_name="",start_index = 0):
		self.input(input_value)
		self.is_training = is_training
		self.layer_index = start_index
		self.layer_name = 'root'
		self.prefix_name = net_name

	def input(self,x):
		self.value = [x]

	def scope_name(self):
		return self.prefix_name+'_'+self.layer_name+'_%03d'%self.layer_index

	def output(self):
		return self.value[-1]

	def variable(self, name, shape, initializer=None, stddev=None, wdecay=None, trainable = True):

		if initializer == None:
			initializer=tf.truncated_normal_initializer(stddev=stddev)

		var = tf.get_variable(name, shape, initializer=initializer)

		if wdecay != None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def layer(self,layer_name):
		self.layer_index += 1
		self.layer_name = layer_name
		return self

	def conv2d(self, kshape, strides = [1,2,2,1],padding='SAME'):
		with tf.variable_scope(self.scope_name()):
			kernel = self.variable('kernel', kshape, stddev=0.05)
			biases = self.variable('biases', kshape[-1:-1], tf.constant_initializer(0.01))
		self.value.append(tf.add(tf.nn.conv2d(self.value[-1], kernel,strides,padding), biases))
		return self

	def matmul(self, shape):
		if len(self.value[-1].get_shape()) > 2:
			self.value.append(tf.reshape(self.value[-1],[int(self.value[-1].get_shape()[0]),-1]))
		wshape = [int(self.value[-1].get_shape()[1])] + shape

		with tf.variable_scope(self.scope_name()):
			weight = self.variable('weight', wshape, stddev=0.03)
			biases = self.variable('biases',  shape, tf.constant_initializer(0.01))
		self.value.append(tf.add(tf.matmul(self.value[-1], weight), biases))
		return self

	def bn(self): # some bugs on it
		shape = self.value[-1].get_shape()  

		params_shape = shape[-1:]
		axis = list(range(len(shape) - 1))  

		with tf.variable_scope(self.scope_name()):

			beta  = self.variable( 'beta', params_shape, tf.zeros_initializer())  
			gamma	 = self.variable('gamma', params_shape, tf.ones_initializer())  

			moving_mean = self.variable('moving_mean', params_shape, tf.zeros_initializer(), trainable=False)  
			moving_variance = self.variable('moving_variance', params_shape,  tf.ones_initializer(), trainable=False)  

			mean, variance = tf.nn.moments(self.value[-1], axis)  

			update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, cfg.bnmad)  
			update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, cfg.bnmad) 

			if not self.is_training:
				mean, variance = moving_mean, moving_variance
			self.value.append(tf.nn.batch_normalization(self.value[-1], mean, variance, beta, gamma, 0.01) )

			return self

	def relu(self):
		self.value.append(tf.nn.relu(self.value[-1]))
		return self

def build_branch(input_value,branch_name,is_training):

	net = LinearNetwork(
		input_value = input_value,
		net_name = branch_name,
		is_training = is_training
	)
	net.layer('conv').conv2d([3,3,3,64]
		).relu()
	net.layer('conv').conv2d([3,3,64,128]
		).relu()
	net.layer('conv').conv2d([3,3,128,128]
		).relu()
	net.layer('conv').conv2d([3,3,128,256]
		).relu()
	net.layer('conv').conv2d([3,3,256,256]
		).relu()
	net.layer('fconv').matmul([2048]
		).relu()
	net.layer('fconv').matmul([1024]
		).relu()

	return net.output()

def inference(fronts,baccks, is_training=True):

	front_feature = build_branch(fronts,'branch_front',is_training)
	bacck_feature = build_branch(fronts,'branch_bacck',is_training)

	net = LinearNetwork(
		input_value = tf.concat([front_feature,bacck_feature],axis=1),
		net_name = 'concat_branch',
		is_training = is_training
	)

	net.layer('fcon').matmul([1024]
		).relu()

	net.layer('fcon').matmul([ 512]
		).relu()

	net.layer('fcon').matmul([coeff_number]
		)

	return net.output()

def visualize(name,x):
	with tf.variable_scope(name):
		#tf.summary.histogram(x.op.name + '/hist', x)
		tf.summary.scalar(name+'/mean',tf.reduce_mean(x))
		tf.summary.scalar(name+'/sum',tf.reduce_sum(x))
		tf.summary.scalar(name+'/min',tf.reduce_min(x))
		tf.summary.scalar(name+'/max',tf.reduce_max(x))


def shmap_render(coeff):
	file = open(os.path.join(data_path,'shmap.data'),'r')
	lines = file.readlines()
	file.close()

	shmap = []
	for line in lines:
		shmap.append(float(line))

	visualize('shmap',shmap)

	shmap = tf.reshape(shmap,[1,image_height,image_width,coeff_order * coeff_order,1]) #[ 1,224,224,9,1]
	coeff = tf.slice(coeff,[0,0],[cfg.batchsize,coeff_order * coeff_order * image_channel])
	coeff = tf.reshape(coeff,[cfg.batchsize,1,1,coeff_order * coeff_order,image_channel])          #[batchsize,1,1,9,3]

	render = tf.multiply(shmap,coeff)

	render = tf.maximum(render,0)

	return tf.reduce_sum(render,3)

def get_render_loss(reference,prediction):

	r1 = shmap_render(reference)
	r2 = shmap_render(prediction)

	if cfg.istraining:
		tf.summary.image('render_reeal',r1)
		tf.summary.image('render_guess',r2)

	render_loss = tf.reduce_mean(
		tf.square(
			tf.subtract(r1,r2)
		)
	)

	visualize('reference',reference)
	visualize('prediction',prediction)

	visualize('render_ref',r1)
	visualize('render_pre',r2)

	return render_loss

def get_mse_loss(reference,prediction):
	mse = tf.reduce_mean(
		tf.square(
			tf.subtract(reference,prediction)
		)
	)
	return mse

def get_total_loss(reference,prediction):

	mloss = get_mse_loss(reference,prediction)
	return mloss

	#below are removed
	rloss = get_render_loss(reference,prediction)
	return mloss, rloss, tf.add(
		tf.multiply(mloss,0.3),
		tf.multiply(rloss,0.7)
	)


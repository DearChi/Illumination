import tensorflow as tf

import os

"""static parameters"""

data_path = '/data/illumination/baseline'
train_path = '/record/illumination/baseline/train'
test_path  = '/record/illumination/baseline/test'

image_height  = 256

image_width   = 256

image_channel = 3

coeff_order = 3

"""training"""
tf.app.flags.DEFINE_integer('batchsize',32,
	"""batchsize""")

tf.app.flags.DEFINE_float('lr',0.01,
	"""orginal learning rate""")

tf.app.flags.DEFINE_integer('nee',31000,
	"""number of examples per epoch""")

tf.app.flags.DEFINE_integer('ned',4,
	"""number of epoches per decay""")

tf.app.flags.DEFINE_float('lrdf',1.0,
	"""decay factor of learning rate""")

tf.app.flags.DEFINE_boolean('restart',False,
	"""if restart toogled, the training result will be removed""")

tf.app.flags.DEFINE_integer('port',6006,
	"""port id for tensorboard""")

tf.app.flags.DEFINE_integer('logfreq',10,
	"""frequency of print log (steps)""")

tf.app.flags.DEFINE_integer('maxstep',10000000,
	"""max steps of training""")

tf.app.flags.DEFINE_integer('savefreq',60,
	"""frequency of saving checkpoint (seconds)""")

tf.app.flags.DEFINE_float('gpufrac',0.7,
	"""fraction of gpu placed""")#todo 

tf.app.flags.DEFINE_float('mad',0.9,
	"""move decay factor""")

"""testing"""
tf.app.flags.DEFINE_float('evalfreq',600,
	"""frequency of testing (seconds)""" )

cfg = tf.app.flags.FLAGS 

_this_variable_is_just_for_output_command_options_list_ = cfg.batchsize
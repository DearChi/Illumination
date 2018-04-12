import tensorflow as tf
import os

"""static parameters"""

data_path = '/document/data/illumination/experiment-10x32/'
train_path = '/record/experiment/train'
test_path  = '/record/experiment/test'
best_path  = '/record/experiment/best'

image_width  = 256

image_height  = 256

image_channel = 3

coeff_order = 3

coeff_number = (coeff_order + 1) * (coeff_order + 1) * image_channel

"""flexible parameters"""
tf.app.flags.DEFINE_integer('batchsize',32,
	"""size of per batch""")

tf.app.flags.DEFINE_float('lr',0.1,
	"""orginal learning rate""")

tf.app.flags.DEFINE_integer('nee',5040,
	"""number of examples per epoch""")

tf.app.flags.DEFINE_integer('ned',10,
	"""number of epoches per decay""")

tf.app.flags.DEFINE_float('lrdf',0.9,
	"""decay factor of learning rate""")

tf.app.flags.DEFINE_boolean('restart',False,
	"""if restart toogled, the training result will be removed""")

tf.app.flags.DEFINE_boolean('istraining',True,
	"""indciate the training process""")

tf.app.flags.DEFINE_integer('port',6006,
	"""port id for tensorboard""")

tf.app.flags.DEFINE_integer('logfreq',10,
	"""frequency of print log (steps)""")

tf.app.flags.DEFINE_integer('maxstep',10000000,
	"""max steps of training""")

tf.app.flags.DEFINE_integer('savefreq',60,
	"""frequency of saving checkpoint (seconds)""")

tf.app.flags.DEFINE_float('gpufrac',0.6,
	"""fraction of gpu placed""")#todo 

tf.app.flags.DEFINE_float('mad',0.9999,
	"""moving decay factor""")

tf.app.flags.DEFINE_float('bnmad',0.999,
	"""moving decay factor of batch norm""")

"""testing"""
tf.app.flags.DEFINE_float('evalfreq',60,
	"""frequency of testing (seconds)""" )

cfg = tf.app.flags.FLAGS 

_this_variable_is_just_set_for_output_command_options_list_ = cfg.batchsize
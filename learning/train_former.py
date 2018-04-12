import tensorflow as tf

from config   import *
from datetime import datetime

import time

import inputs
import model
import optimize
import recorder
	
def train():

	global_step = tf.train.get_or_create_global_step()

	fronts,baccks,coeffs = inputs.fetch_batches(eval_flag = False)

	guess = model.inference(fronts,baccks)

	loss  = model.get_total_loss(coeffs, guess)
	
	tf.summary.scalar('loss',loss)

	operator = optimize.optimize(global_step, loss)

	class _LoggerHook(tf.train.SessionRunHook):
		def begin(self):
			self._step = recorder.get_step()
			self._start_time = time.time()

		def before_run(self, run_context):
			self._step += 1
			return tf.train.SessionRunArgs(loss)

		def after_run(self, run_context, run_values):
			if self._step % cfg.logfreq == 0:
				duration = time.time() - self._start_time
				self._start_time = time.time()

				#seconds_per_batchs
				spb = duration / cfg.logfreq

				#examples_per_second
				eps = cfg.batchsize // spb

				#loss value
				lv = run_values.results

				print('%s: step %d loss=%.4lf %d examples/sec; %.3lf sec/batch'%(
					str(datetime.now())[:19], self._step, lv, eps, spb
				))
				recorder.record_loss(self._step,lv, 0)

	var_list = [var for var in tf.global_variables() if True or var not in tf.trainable_variables()]

	scaffold = tf.train.Scaffold(
		saver=tf.train.Saver(tf.trainable_variables())
		init_op = tf.variables_initializer([var for var in tf.global_variables() if var not in tf.trainable_variables()])
	)

	with tf.train.MonitoredTrainingSession(

		checkpoint_dir=os.path.join(train_path,'save'),

		hooks=[tf.train.StopAtStepHook(last_step=cfg.maxstep),
				tf.train.NanTensorHook(loss),
				_LoggerHook()
		],

		scaffold = scaffold,

    	save_summaries_steps = cfg.logfreq,

    	save_checkpoint_secs = cfg.savefreq,

		config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpufrac))

	)as mon_sess:
		while not mon_sess.should_stop():
			mon_sess.run(operator)

def work_main():

	if not os.path.exists(data_path):
		exit('Error when loading input data - Path does not exists')

	if cfg.restart and tf.gfile.Exists(train_path):
		tf.gfile.DeleteRecursively(train_path)

	if not os.path.exists(os.path.join(train_path,'save')):
		os.makedirs(os.path.join(train_path,'save'))
	
	train()

work_main()
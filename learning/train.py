import tensorflow as tf

from config   import *
from datetime import datetime

import time

import inputs
import model
import optimize

def read_step():
	loss_file = os.path.join(train_path,'tloss.txt') 
	if tf.gfile.Exists(loss_file):
		txt = open(loss_file,'r')
		lines = txt.readlines()
		txt.close()
		return int(lines[-1].split(',')[0])
	return 0
def write_step(step,loss):
	loss_file = os.path.join(train_path,'tloss.txt') 

	txt = open(loss_file,'a')
	txt.write('%10d,%f\n'%(step,loss))
	txt.close()
def train():

	global_step = tf.train.get_or_create_global_step()

	image, coeff = inputs.fetch_batches(eval_flag = False)

	guess = model.inference(image)

	loss  = model.get_total_loss(coeff,guess)
	
	tf.summary.scalar('loss',loss)

	operator = optimize.optimize(global_step, loss)

	class _LoggerHook(tf.train.SessionRunHook):
		def begin(self):
			self._step = read_step()
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

				loss_value = run_values.results

				print('%s: step %d loss=%.4lf %d examples/sec; %.3lf sec/batch'%(
					str(datetime.now())[:19],
					self._step,
					loss_value,
					eps, spb
				))
				write_step(self._step,loss_value)

	with tf.train.MonitoredTrainingSession(

		checkpoint_dir=train_path,

		hooks=[tf.train.StopAtStepHook(last_step=cfg.maxstep),
				tf.train.NanTensorHook(loss),
				_LoggerHook()
		],

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

	if not os.path.exists(train_path):
		os.makedirs(train_path)
	
	train()
		

work_main()
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

	front, bacck, coeff = inputs.fetch_batches(eval_flag = False)

	guess = model.inference(front,bacck)

	mse_loss = model.get_total_loss(coeff, guess)
	
	tf.summary.scalar('mse_loss',mse_loss)

	operator = optimize.optimize(global_step, mse_loss)

	with tf.Session(
		config = tf.ConfigProto(
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpufrac)
		)
	) as sess:
		
		# fetch input reader thread
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		# merged all summary we have defined
		summary_merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(os.path.join(train_path,'event'))

		# restore the ckpt if it exist, otherwise restart training
		ckpt = tf.train.get_checkpoint_state(os.path.join(train_path,'model'))
		saver = tf.train.Saver(tf.global_variables())

		if ckpt and ckpt.model_checkpoint_path:
			sess.run(tf.variables_initializer([var for var in tf.global_variables() if var not in tf.trainable_variables()]))
			saver.restore(sess, ckpt.model_checkpoint_path)
			step = recorder.get_step()
		else:
			print('No check point file, training will be restart')
			step = 0
			sess.run(tf.variables_initializer(tf.global_variables()))

		#time recorder for computing speed of training, and controlling the save frequency.
		last_save_time = time.time()
		last_log_time = time.time()

		while step < cfg.maxstep:

			step += 1  

			sess.run([operator])

			# log the loss
			if step % cfg.logfreq == 0:
				duration = time.time() - last_log_time
				last_log_time = time.time()

				seconds_per_batchs = duration / cfg.logfreq

				examples_per_second = cfg.batchsize // seconds_per_batchs

				mlv, summary = sess.run([mse_loss ,summary_merged])

				# print('%s: step %d mse=%.4lf rloss=%.4lf loss=%.4lf %d examples/sec; %.3lf sec/batch'%(
				# 	str(datetime.now())[:19], step, mlv,rlv,lv, examples_per_second, seconds_per_batchs
				# ))

				print('%s: step %d epoch_count=%d mse_loss=%.4lf %d examples/sec; %.3lf sec/batch'%(
					str(datetime.now())[:19], step, step * cfg.batchsize // cfg.nee + 1, mlv, examples_per_second, seconds_per_batchs
				))

				summary_writer.add_summary(summary, step)

				#record loss to a txt file, because losses record in the summary are not sufficient to visualize by other program.
				recorder.record_loss(step,mlv,mlv,mlv,0)

			# save the model
			if time.time() - last_save_time >= cfg.savefreq:
				#print('saving to ckpt....',end='')
				saver.save(sess,os.path.join(train_path,'model','model'),global_step=step)
				#print('done')
				last_save_time = time.time()

		# stop the reader queue thread
		coord.request_stop()
		coord.join(threads)

		sess.close()

def work_main():

	if not os.path.exists(data_path):
		exit('Error when loading input data - Path does not exists')

	if cfg.restart and tf.gfile.Exists(train_path):
		tf.gfile.DeleteRecursively(train_path)

	if not os.path.exists(os.path.join(train_path,'model')):
		os.makedirs(os.path.join(train_path,'model'))

	if not os.path.exists(os.path.join(train_path,'event')):
		os.makedirs(os.path.join(train_path,'event'))
	
	train()
		
if __name__=="__main__":
	work_main()
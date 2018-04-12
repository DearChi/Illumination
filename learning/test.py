import tensorflow as tf

from config   import *
from datetime import datetime

import time

import inputs
import model
import recorder

def eval():

	cfg.istraining = False
	
	global_step = tf.train.get_or_create_global_step()

	front, bacck, coeff = inputs.fetch_batches(eval_flag = True)

	guess = model.inference(front,bacck,is_training=False)


	mse_loss = model.get_total_loss(coeff,guess)
	saver = tf.train.Saver(tf.global_variables())

	mloss_summary = tf.placeholder(dtype = tf.float32)

	tf.summary.scalar('mse_loss',mloss_summary)

	with tf.Session() as sess:
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		summary_merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(os.path.join(test_path,'event'))


		while True:
			
			NUM_PER_EVAL = cfg.nee // cfg.batchsize

			ckpt = tf.train.get_checkpoint_state(os.path.join(train_path,'model'))

			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			else:
				print('No checkpoint file found')
				return

			avg_loss = [0.0]
			for i in range(NUM_PER_EVAL):
				loss_value = sess.run([mse_loss])
				for index in range(1):
					avg_loss[index] += loss_value[index]

			for index in range(1):
				avg_loss[index] /= NUM_PER_EVAL

			t1 = datetime.now()

			print(t1,'step =',step,', mloss =',avg_loss[0])

			recorder.record_loss(step,avg_loss[0],avg_loss[0],avg_loss[0],1)


			for index in range(1):
				min_loss = recorder.get_min_loss(index)

				if avg_loss[index] < min_loss:
					best_model_path = os.path.join(best_path,'model_for_loss_%d'%index,'best_model')
					if tf.gfile.Exists(best_model_path):
						tf.gfile.DeleteRecursively(best_model_path)
					saver.save(sess,best_model_path,global_step=step)
					recorder.record_loss(step,avg_loss[0],avg_loss[0],avg_loss[0],2)

			summary = sess.run(summary_merged,feed_dict={mloss_summary:avg_loss[0]})

			summary_writer.add_summary(summary, step)
			
			time.sleep(cfg.evalfreq)

		coord.request_stop()
		coord.join(threads)
		sess.close()

def work_main():

	if not os.path.exists(data_path):
		exit('Error when loading input data - Path does not exists')

	if cfg.restart and tf.gfile.Exists(test_path):
		tf.gfile.DeleteRecursively(test_path)

	if not os.path.exists(os.path.join(test_path,'event')):
		os.makedirs(os.path.join(test_path,'event'))
	
	best_model_path = os.path.join(best_path,'model_for_loss_0')
	if not os.path.exists(best_model_path):
		os.makedirs(best_model_path)

	best_model_path = os.path.join(best_path,'model_for_loss_1')
	if not os.path.exists(best_model_path):
		os.makedirs(best_model_path)
		
	best_model_path = os.path.join(best_path,'model_for_loss_2')
	if not os.path.exists(best_model_path):
		os.makedirs(best_model_path)

	eval()
		

work_main()
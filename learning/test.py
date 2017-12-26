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
	loss_file = os.path.join(test_path,'eloss.txt') 

	txt = open(loss_file,'a')
	txt.write('%10d,%f\n'%(step,loss))
	txt.close()

def eval():

	lobal_step = tf.contrib.framework.get_or_create_global_step()

	image, coeff = inputs.fetch_batches(eval_flag = True)

	guess = model.inference(image,is_training=False)

	loss  = model.get_total_loss(coeff,guess)

	saver = tf.train.Saver(tf.global_variables())

	total_loss = tf.placeholder(dtype = tf.float32)

	tf.summary.scalar('loss', total_loss)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		summary_merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(test_path)

		step = read_step()

		while True:

			NUM_PER_EVAL = cfg.nee // cfg.batchsize

			ckpt = tf.train.get_checkpoint_state(train_path)

			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			else:
				print('No checkpoint file found')
				return

			total = 0.0
			for i in range(NUM_PER_EVAL):
				total += sess.run(loss)

			total /= NUM_PER_EVAL

			t1 = datetime.now()

			print(t1,'step =',step,'loss =',total)

			write_step(step,total)

			summary = sess.run(summary_merged,feed_dict={total_loss : total})

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

	if not os.path.exists(test_path):
		os.makedirs(test_path)
	
	eval()
		

work_main()
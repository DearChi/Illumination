import tensorflow as tf

from config import *
import model

from PIL import Image as im
import matplotlib.image as plimg

best_path = '/record/fbvgg-nobn/best'
data_path = '/document/rawdata/fandb-fov60-256x256/data'
predict_path = '/document/share/eval/sh_fbvgg'

test_set  = [16,35,42,57,60,62,68,77,84,85,90,91,106,113,127,142,158,161,174,186,191,193,200,213,229,231,245,254,266,271]


def sh_write(filename,shlist):
	if not os.path.exists(os.path.dirname(filename)):
		os.makedirs(os.path.dirname(filename))
	sh = open(filename,'w')
	sh.write('<?xml version="1.0"?>\n')
	sh.write('<opencv_storage><SH type_id="opencv-matrix"><rows>%d</rows><cols>1</cols><dt>"3f"</dt><data>\n'%(coeff_number//image_channel))
	for d in shlist:
		sh.write('%e '%d)
	sh.write('</data></SH></opencv_storage>\n')
	sh.close()

def read_image(filename):
	return plimg.pil_to_array(im.open(filename).resize((image_height,image_width))).tolist()

def inference(files_path, model_path):
	queue = files_path

	cfg.batchsize = min(32,len(queue))

	front = tf.placeholder(tf.uint8,shape=(cfg.batchsize,image_height,image_width,image_channel))
	bacck = tf.placeholder(tf.uint8, shape=(cfg.batchsize,image_height,image_width,image_channel))

	front_norm = tf.divide(tf.cast(front,tf.float32), 1.0)
	bacck_norm = tf.divide(tf.cast(bacck,tf.float32), 1.0)

	coeff = model.inference(front_norm, bacck_norm ,is_training=False)

	saver = tf.train.Saver(tf.trainable_variables())

	with tf.Session() as sess:

		ckpt = tf.train.get_checkpoint_state(model_path)

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			sess.run(tf.variables_initializer([var for var in tf.global_variables() if var not in tf.trainable_variables()]))
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return None

		for batch in range(len(queue)//cfg.batchsize + 1):
			#ensure that size of this batch is cfg.batchsize
			end   = min( (batch + 1) * cfg.batchsize, len(queue) )
			begin = min( batch * cfg.batchsize, end - cfg.batchsize)

			print(begin,end,len(queue))

			images = (
				[read_image(queue[index][0]) for index in range(begin,end)],
				[read_image(queue[index][1]) for index in range(begin,end)]
			)

			coeffes = sess.run(coeff,feed_dict={front:images[0],bacck:images[1]})

			for index in range(begin,end):
				name = queue[index][2]
				sh_write(queue[index][2],coeffes[index-begin])

		sess.close()

def fetch_filename_queue():
	#return [(image1_path,image2_path,shsaved_path)...]

	for index in test_set:
		for sample in range(60):
			queue.append((
				os.path.join(data_path,'im','%03d'%index,'%03d_0.png'%sample),
				os.path.join(data_path,'im','%03d'%index,'%03d_1.png'%sample),
				os.path.join(predict_path,'%03d'%index,'%03d_vgg.xml'%sample)
			))
	return queue

if __name__ == "__main__"
	inference(
		fetch_filename_queue(),
		os.path.join(best_path,'model_for_loss_1')
	)


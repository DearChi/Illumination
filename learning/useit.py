import tensorflow as tf

from config import *
import model

from PIL import Image as im
import matplotlib.image as plimg



def inference(images_data):

	cfg.batchsize = len(images_data)

	image = tf.placeholder(tf.float32,shape=(cfg.batchsize,256,256,3))

	coeff = model.inference(image,is_training=False)

	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:

		ckpt = tf.train.get_checkpoint_state(train_path)

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return None

		shs = sess.run(coeff,feed_dict={image:images_data})

		sess.close()

	return shs

def read_images(filenames):
	dataes = [ 
		plimg.pil_to_array(im.open(filename)).tolist() 
		for filename in filenames
	]
	dataes = [
		[[[dataes[i][row][col][cha] / 256.0 for cha in range(3)]
		for col in range(256) ] for row in range(256) ] for i in range(len(dataes))
	]
	return dataes

def work_main():
	filenames = ['/home/dearchi/workspace/cgi/simulate_video/output/%02d/00_000.png'%(index+1) for index in range(1)]
	coeffes = inference(read_images(filenames))
	print(coeffes)

work_main()

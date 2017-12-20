import tensorflow as tf

from config import *

def read_data_info(eval_flag):

	return 1,100
	info = open(os.path.join(data_path,'data.info'),'r')
	lines = info.readlines()
	info.close()

	if not eval_flag:
		num_files,num_examples = int(lines[0]),int(lines[1])  
	else:
		num_files,num_examples = int(lines[2]),int(lines[3])

	cfg.nee = num_examples
	
	return num_files, num_examples

def fetch_filename_queue(eval_flag):

	num_files, num_examples = read_data_info(eval_flag)
	
	if eval_flag:
		filenames = [os.path.join(data_path,'test_%03d.bin'%_) for _ in range(num_files)]
	else:
		filenames = [os.path.join(data_path,'train_%03d.bin'%_) for _ in range(num_files)]
	
	return tf.train.string_input_producer(filenames)

def read(eval_flag):

	filename_queue = fetch_filename_queue(eval_flag)

	image_size = image_height * image_width * image_channel
	coeff_size = coeff_order * coeff_order * image_channel
	example_bytes = (image_size + image_size) * 4 # size of tf.float32

	reader = tf.FixedLengthRecordReader(record_bytes = example_bytes)
	
	key, raw_data = reader.read(filename_queue)

	record = tf.decode_raw(raw_data,tf.float32)
	
	image = tf.reshape(
		tf.strided_slice(record,[0],[image_size]),
		[image_height, image_width, image_channel]
	)

	coeff = tf.reshape(
		tf.strided_slice(record, [image_size], [image_size + coeff_size]),
		[coeff_size]
	)

	return image, coeff

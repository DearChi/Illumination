import tensorflow as tf
from config import *

def read_data_info(eval_flag):

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

	coeff_bytes = coeff_number
	image_bytes = image_height * image_width * image_channel

	example_bytes = (image_bytes * 2 + coeff_bytes) * 4 # They are all float32 data - 4 bytes

	reader = tf.FixedLengthRecordReader(record_bytes = example_bytes)
	
	key, raw_data = reader.read(filename_queue)

	record = tf.decode_raw(raw_data,tf.float32)

	front = tf.cast(
		tf.reshape(
			tf.strided_slice(record,[0],[image_bytes]),
			[image_height, image_width, image_channel]
		),
		tf.float32
	)
	bacck  = tf.cast(
		tf.reshape(
			tf.strided_slice(record,[image_bytes],[image_bytes+image_bytes]),
			[image_height, image_width, image_channel]
		),
		tf.float32
	)

	coeff = tf.cast(
		tf.reshape(
			tf.strided_slice(record, [image_bytes + image_bytes], [image_bytes + image_bytes + coeff_bytes]),
			[coeff_number]
		),
		tf.float32
	)

	return front, bacck, coeff

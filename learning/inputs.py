import tensorflow as tf

from config import *
from decode import *

def generate_batch(example, min_queue_examples, batch_size, shuffle):

	num_preprocess_threads = 8

	if shuffle:
		examples = tf.train.shuffle_batch(
			example,
			batch_size = batch_size,
			num_threads = num_preprocess_threads,
			capacity = min_queue_examples + 3 * batch_size,
			min_after_dequeue = min_queue_examples + batch_size
		)
	else:
		examples = tf.train.batch(
		    example,
		    batch_size = batch_size,
		    num_threads = num_preprocess_threads,
		    capacity = min_queue_examples + 3 * batch_size
		)
		
	return examples


def fetch_batches(eval_flag):

	data = read(eval_flag)

	min_queue_examples = int(cfg.nee * 0.05)

	images, coeffs = generate_batch(
		data,
		min_queue_examples,
		batch_size = cfg.batchsize,
		shuffle = not eval_flag
	)
	return images, coeffs


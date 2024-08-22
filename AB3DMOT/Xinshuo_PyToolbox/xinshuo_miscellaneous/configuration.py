# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, numpy as np, random
# import tensorflow as tf

def suppress_caffe_terminal_log():
	'''
	prevent caffe log to terminal
	0 - debug
	1 - info (still a LOT of outputs)
	2 - warnings
	3 - errors
	'''

	os.environ["GLOG_minloglevel"] = "2"


# def prepare_seed(rand_seed):
# 	np.random.seed(rand_seed)
# 	random.seed(rand_seed)
# 	torch.manual_seed(rand_seed)
# 	torch.cuda.manual_seed_all(rand_seed)


# def get_tf_good_gpu_session():
# 	'''
# 	return a tf session which takes proper GPU usage instead of all memory
# 	'''
# 	num_threads = os.environ.get('OMP_NUM_THREADS')

# 	if num_threads:
# 		config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
# 		config.gpu_options.allow_growth=True
# 	else:
# 		config = tf.ConfigProto()
# 		config.gpu_options.allow_growth=True
	    
# 	return tf.Session(config=config)  

def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
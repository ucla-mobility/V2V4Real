# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, numpy as np, random, torch

def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)
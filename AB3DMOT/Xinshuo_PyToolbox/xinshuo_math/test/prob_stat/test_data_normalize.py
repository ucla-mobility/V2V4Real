# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from prob_stat import data_normalize
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_data_normalize():
	print('testing 1-d data with data range')
	random_data = [1, 2, 3, 4, 6]
	normalized_data = data_normalize(random_data)
	assert CHECK_EQ_NUMPY(normalized_data, np.array([0, 0.2, 0.4, 0.6, 1]))

	print('testing 1-d data with given data range')
	random_data = [1, 2, 3, 4, 6]
	normalized_data = data_normalize(random_data, data_range=(0, 100))
	assert CHECK_EQ_NUMPY(normalized_data, np.array([0.01, 0.02, 0.03, 0.04, 0.06]))

	print('testing 3-d data with data range')
	random_data = np.random.rand(2, 3, 4) * 100
	normalized_data = data_normalize(random_data)
	assert np.max(normalized_data) == 1 and np.min(normalized_data) == 0

	print('testing 1-d data with sum to 1')
	random_data = [1, 2, 3, 4, 6]
	normalized_data = data_normalize(random_data, method='sum')
	assert np.sum(normalized_data) == 1

	print('testing 3-d data with sum to a value')
	random_data = np.random.rand(2, 3, 4) * 100
	normalized_data = data_normalize(random_data, method='sum', sum=100)
	assert_almost_equal(np.sum(normalized_data), 100)
	assert normalized_data.shape == random_data.shape

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_data_normalize()
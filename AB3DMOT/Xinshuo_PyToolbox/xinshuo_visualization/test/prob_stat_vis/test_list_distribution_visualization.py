# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# from xinshuo_visualization import visualize_bar
# from xinshuo_miscellaneous import construct_dict_from_lists, scalar_list2str_list

# a = [3, 4, 5, 6, 7]
# visualize_bar(a, save_path='/home/xinshuo/test.png')
# # visualize_distribution(a, bin_size=0.001)
# # b = ['1', '2', '3', '4', '5']
# # b = scalar_list2str_list(range(len(a)))
# # print(bin_size)
# b = ['aa', 'ff', 'cc', 'dd', 'wee']
# test_dict = construct_dict_from_lists(b, a)
# visualize_bar_graph(test_dict)

import numpy as np
import matplotlib.pyplot as plt



alphab = ['1', '2', '3', '4', '5', '6']
frequencies = [23, 44, 12, 11, 2, 10]

pos = np.arange(len(alphab))
print pos
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks([1,3,5])
# ax.set_xticklabels([])

plt.bar(pos, frequencies, width, color='r')
plt.show()
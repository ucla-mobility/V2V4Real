# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import init_paths
from xinshuo_miscellaneous import print_log

def test_print_log():
	str_to_print = '123'
	log_file = 'log.txt'
	log_file = open(log_file, 'w')
	print_log(str_to_print, log=log_file, same_line=True)
	print_log(str_to_print, log=log_file, same_line=True)
	print_log(str_to_print, log=log_file, same_line=True)
	print_log(str_to_print, log=log_file, same_line=True)
	log_file.close()

if __name__ == '__main__':
	test_print_log()
# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import copy, os

from .type_check import islist, isstring

################################################################## conversion ##################################################################
def safe_list(input_data, warning=True, debug=True):
	'''
	copy a list to the buffer for use

	parameters:
		input_data:		a list

	outputs:
		safe_data:		a copy of input data
	'''
	if debug: assert islist(input_data), 'the input data is not a list'
	safe_data = copy.copy(input_data)
	return safe_data

def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data
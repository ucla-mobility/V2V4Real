# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions checking the datatype and equality of input variables
import numpy as np
from .type_check import islist, isdict, isnparray

############################################################# equality check
def CHECK_EQ_LIST_SELF(input_list, debug=True):
	'''
	check all elements in a list are equal
	'''
	if debug: assert islist(input_list), 'input is not a list'
	return input_list[1:] == input_list[:-1]

def CHECK_EQ_DICT(input_dict1, input_dict2, debug=True):
    '''
    check all elements in a list are equal
    '''
    if debug:
        assert isdict(input_dict1) and isdict(input_dict2), 'input is not a dictionary'
        assert len(input_dict1) == len(input_dict2), 'length of input dictionary is not equal'

    for key, value in input_dict1.items():
        if input_dict2.has_key(key) and input_dict2[key] == value: continue
        else: return False
    return True

def CHECK_EQ_LIST_ORDERED(input_list1, input_list2, debug=True):
    '''
    check two lists are equal in ordered way
    '''
    if debug: assert islist(input_list1) and islist(input_list2), 'input lists are not correct'
    return input_list1 == input_list2

def CHECK_EQ_LIST_UNORDERED(input_list1, input_list2, debug=True):
    '''
    check two lists are equal in ordered way
    '''
    if debug: assert islist(input_list1) and islist(input_list2), 'input lists are not correct'
    return set(input_list1) == set(input_list2)

def CHECK_EQ_NUMPY(np_data1, np_data2, debug=True):
    '''
    check two numpy data are equal
    '''
    if debug:
        assert isnparray(np_data1) and isnparray(np_data2), 'the input numpy data is not correct'
        assert np_data2.shape == np_data1.shape, 'the shapes of two data blob are not equal'

    return np.all(np_data1 == np_data2)
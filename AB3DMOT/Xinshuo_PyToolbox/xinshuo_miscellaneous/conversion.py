# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file define a set of functions which converting data type
import struct, numpy as np
from itertools import islice

from .private import safe_list
from .type_check import isstring, isinteger, isnparray, islist, isext, islistoflist, isrange, isscalar, isfloat
from .counter import Timer

######################################################### list related #########################################################
def remove_list_from_list(input_list, list_toremove_src, warning=True, debug=True):
	'''
	remove a list "list_toremove_src" from a list "list_src" if found, skip if not found

	parameteters:
		input_list:				a list to be removed from
		list_toremove_src:		a list to be removed

	outputs:
		list_remained:			a list of remaining elements after removal
		list_removed:			a list of elements to be successfully removed 
								(as some elements in list_toremove_src may not found in list_src, where the removal fails)
	'''
	list_remained = safe_list(input_list, warning=warning, debug=debug)
	list_toremove = safe_list(list_toremove_src, warning=warning, debug=debug)
	list_removed = []
	for item in list_toremove:
		try:
			list_remained.remove(item)
			list_removed.append(item)
		except ValueError:
			if warning: print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

	return list_remained, list_removed

def remove_unique_item_from_list(input_list, item, warning=True, debug=True):
	'''
	remove all instances of a single item from a list

	parameters:
		input_list:				a list to be removed from

	outputs:
		list_remained:			a list of remaining elements after removal
		count_removal:			number of times the requested item to be removed
	'''
	list_remained = safe_list(input_list, warning=warning, debug=debug)
	count_removal = 0
	while 1:
		try: 
			list_remained.remove(item)
			count_removal += 1
		except ValueError:
			if warning and count_removal == 0: print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')
			break

	return list_remained, count_removal

def find_unique_common_from_lists(input_list1, input_list2, only_com=False, warning=True, debug=True):
	'''
	find common items from 2 lists, the returned elements are unique. repetitive items will be ignored
	if the common items in two elements are not in the same order, the outputs follows the order in the first list

	parameters:
		input_list1, input_list2:		two input lists
		only_com:		True if only need the common list, i.e., the first output, saving computational time

	outputs:
		list_common:	a list of elements existing both in list_src1 and list_src2	
		index_list1:	a list of index that list 1 has common items
		index_list2:	a list of index that list 2 has common items
	'''

	input_list1 = safe_list(input_list1, warning=warning, debug=debug)
	input_list2 = safe_list(input_list2, warning=warning, debug=debug)

	common_list = list(set(input_list1).intersection(input_list2))

	if only_com: return common_list

	# find index
	index_list1 = []
	for index in range(len(input_list1)):
		item = input_list1[index]
		if item in common_list:
			index_list1.append(index)

	index_list2 = []
	for index in range(len(input_list2)):
		item = input_list2[index]
		if item in common_list:
			index_list2.append(index)

	return common_list, index_list1, index_list2

def reverse_list(input_list, warning=True, debug=True):
	'''
	reverse the order of a list

	parameters:
		input_list:		a list

	outputs:
		reversed_list:	the list in a reverse order
	'''
	input_list = safe_list(input_list, warning=warning, debug=debug)
	reversed_list = input_list[::-1]
	return reversed_list

def list_reorder(input_list, order_index, debug=True):
	'''
	reorder a list based on a list of index
	'''
	if debug:
		assert islist(input_list) and islist(order_index), 'inputs are not two lists'
		assert len(input_list) == len(order_index), 'length of input lists is not equal'
		assert all(isscalar(index_tmp) for index_tmp in order_index), 'the list of order is not correct'

	reordered_list = [ordered for whatever, ordered in sorted(zip(order_index, input_list))]
	return reordered_list

def merge_listoflist(listoflist, unique=False, debug=True):
	'''
	merge a list of list

	parameters:
		unique: 	boolean

	outputs:
		if unique false:	a combination of lists in original order
		if unique true:		a combination of lists with only unique items, the resulting list is not in original order
	'''
	if debug: assert islistoflist(listoflist), 'the input is not a list of list'
	merged = list()
	for individual in listoflist:
		merged = merged + individual

	if unique:
		merged = list(set(merged))
		merged.sort()

	return merged

def remove_list_from_index(list_src, list_index_src, warning=True, debug=True):
	'''
	remove a list "list_to_remove" from a list "list_all_src" based on value
	'''
	if debug: assert islist(list_src) and islist(list_index_src), 'input lists are not valid'

	input_list = safe_list(list_src, warning=warning, debug=debug)
	index_list = safe_list(list_index_src, warning=warning, debug=debug)
	index_list.sort(reverse=True)
	for item_index in index_list:
		try: del input_list[item_index]
		except ValueError: 
			if warning: print('Warning!!!!!! Item to remove is not in the list. Remove operation is not done.')

	return input_list

def remove_empty_item_from_list(list_to_remove, debug=True):
	'''
	remove an empty string from a list
	'''
	if debug: assert islist(list_to_remove), 'input list is not a list'
	return remove_unique_item_from_list(list_to_remove, '', debug=debug)[0]

def scalarlist2strlist(scalar_list, debug=True):
	'''
	convert a list of scalar to a list of string
	'''	
	if debug: assert islist(scalar_list) and all(isscalar(scalar_tmp) for scalar_tmp in scalar_list), 'input list is not a scalar list'
	
	str_list = list()
	for item in scalar_list: str_list.append(str(item))
	return str_list

def scalarlist2floatlist(scalar_list, debug=True):
	'''
	convert a list of scalar to a list of floating number
	'''
	if debug: assert islist(scalar_list) and all(isscalar(scalar_tmp) for scalar_tmp in scalar_list), 'input list is not a scalar list'
	
	float_list = list()
	for item in scalar_list: float_list.append(float(item))
	return float_list

def strlist2floatlist(str_list, warning=True, debug=True):
	'''
	convert a list of string to a list of floating number
	'''
	if debug:
		assert islist(str_list), 'input is not a list'
		assert all(isstring(str_tmp) for str_tmp in str_list), 'input is not a list of string'
	if any(len(str_tmp) == 0 for str_tmp in str_list):
		if warning: print('warning: the list of string contains empty element which will be removed before converting to floating number')
		str_list = filter(None, str_list)
	return [float(str_tmp) for str_tmp in str_list]

def strlist2intlist(str_list, warning=True, debug=True):
	'''
	convert a list of string to a list of integer number
	'''
	if debug:
		assert islist(str_list), 'input is not a list'
		assert all(isstring(str_tmp) for str_tmp in str_list), 'input is not a list of string'
	if any(len(str_tmp) == 0 for str_tmp in str_list):
		if warning: print('warning: the list of string contains empty element which will be removed before converting to integer number')
		str_list = filter(None, str_list)
	return [int(str_tmp) for str_tmp in str_list]

def floatlist2bytes(float_list, debug=True):
	'''
	convert a list of floating number to bytes
	'''
	if debug: assert isfloat(float_list) or (islist(float_list) and all(isfloat(float_tmp) for float_tmp in float_list)), 'input is not a floating number or a list of floating number'

	# convert a single floating number to a list with one item
	if isfloat(float_list): float_list = [float_list]
	try: binary = struct.pack('%sf' % len(float_list), *float_list)
	except ValueError: print('Warnings!!!! Failed to convert to bytes!!!!!')

	return binary

def list2tuple(input_list, debug=True):
	'''
	convert a list to a tuple
	'''
	if debug: assert islist(input_list), 'input is not a list'
	return tuple(input_list)

######################################################### string related #########################################################
def character2onehot(character, warning=True, debug=True):
	'''
	In this function you need to output a one hot encoding of the ASCII character.
	'''
	if debug: assert isinteger(character) or isstring(character), 'input data type is not correct'
	
	# convert string to integer number
	if isstring(character):
		if debug: assert len(character) == 1, 'character should be a string with length 1'
		character = ord(character)

	one_hot_vec = np.zeros([self.nFeats, ], dtype='float32')
	one_hot_vec[character] = 1
	return one_hot_vec

def string2onehot(string, warning=True, debug=True):
	'''
	convert a string to a set of 2d one hot tensor
	'''
	if debug: assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	one_hot_vec = [character2onehot(ord(string[0]))]
	for character in string[1:]: one_hot_vec = np.vstack((one_hot_vec, [character2onehot(ord(character))]))
	return one_hot_vec

def onehot2ord(onehot, warning=True, debug=True):
	'''
	convert one hot vector to a ord integer number
	'''
	if debug:
		assert isnparray(onehot) and onehot.ndim == 1, 'input should be 1-d numpy array'
		assert sum(onehot) == 1 and np.count_nonzero(onehot) == 1, 'input numpy array is not one hot vector'
	return np.argmax(onehot)

def onehot2character(onehot, warning=True, debug=True):
	'''
	convert one hot vector to a character
	'''
	return chr(onehot2ord(onehot))

def onehot2string(onehot, warning=True, debug=True):
	'''
	convert a set of one hot vector to a string
	'''
	if isnparray(onehot):
		onehot.ndim == 2, 'input should be 2-d numpy array'
		onehot = list(onehot)
	elif islist(onehot): assert CHECK_EQ_LIST([tmp.ndim for tmp in onehot]), 'input list of one hot vector should have same length'
	else: assert False, 'unknown error'

	if debug: assert all(sum(onehot_tmp) == 1 and np.count_nonzero(onehot_tmp) == 1 for onehot_tmp in onehot), 'input numpy array is not a set of one hot vector'
	ord_list = [onehot2ord(onehot_tmp) for onehot_tmp in onehot]
	return ord2string(ord_list)

def string2ord(string, warning=True, debug=True):
	'''
	convert a string to a list of ASCII character
	'''
	if debug: assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	
	ord_list = []
	for character in string: ord_list.append(ord(character))
	return ord_list

def ord2string(ord_list, debug=True):
	'''
	convert a list of ASCII character to a string
	'''
	if debug:
		assert islist(ord_list) and len(ord_list) > 0, 'input should be a list of ord with length larger than 0'
		assert all(isinteger(tmp) for tmp in ord_list), 'all elements in the list of ord should be integer'
	
	L = ''
	for o in ord_list: L += chr(o)
	return L

def string2ext_filter(string, debug=True):
	'''
	convert a string to an extension filter
	'''
	if debug: assert isstring(string), 'input should be a string'
	if isext(string): return string
	else: return '.' + string

def remove_str_from_str(src_str, substr, debug=True):
	'''
	remove a substring from a string
	'''
	if debug: assert isstring(src_str) and isstring(substr), 'the input string is not valid'
	valid = (src_str.find(substr)!=-1)
	removed = src_str.replace(substr, '')
	pre_part = src_str[0:valid-1] if (valid > -1) else ''
	pos_part = src_str[valid+len(substr):] if (valid < len(src_str) -1) else '' 

	return removed, valid, pre_part, pos_part

def str2num(string, debug=True):
	'''
	convert a string to float or int if possible
	'''
	if debug: assert isstring(string), 'the source string is not a string'
	try: return int(string)
	except ValueError: return float(string)

def path2str(path, debug=True):
	'''
	convert a string of path to a string
	'''
	if debug: isstring(path), 'the path is wrong'
	return '_'.join(path.split('/'))

def convert_secs2time(seconds):
    '''
    format second to human readable way
    '''
    assert isscalar(seconds), 'input should be a scalar to represent number of seconds'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '[%d:%02d:%02d]' % (h, m, s)

######################################################### dict related #########################################################
def get_subdict(dictionary, num, debug=True):	
	if debug:
		assert isdict(dictionary), 'dictionary is not correct'
		assert num > 0 and isinteger(num) and num <= len(dictionary), 'number of sub-dictionary is not correct'

	def take(num, iterable):
		return dict(islice(iterable, num))

	return take(num, dictionary.iteritems())

def sort_dict(dictionary, sort_base='value', order='descending', debug=True):
	'''
	sort a dictionary to a list
	'''
	if debug:
		assert isdict(dictionary), 'the input is not a dictionary'
		assert sort_base == 'value' or sort_base == 'key', 'the sorting is based on key or value'
		assert order == 'descending' or order == 'ascending', 'the sorting order is not descending or ascending'

	reverse = True if order == 'descending' else False

	# if sys.version_info[0] < 3:
	# if sort_base == 'value': return sorted(dictionary.iteritems(), key=lambda (k,v):(v,k), reverse=reverse)	# for python2
	if sort_base == 'value': return sorted(dictionary.iteritems(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)	# for python3
	else: return sorted(dictionary.iteritems(), reverse=reverse)

def construct_dict_from_lists(list_key, list_value, debug=True):
	'''
	construct a distionary from two lists
	'''
	if debug:
		assert islist(list_key) and islist(list_value), 'the input key list and value list are not correct'
		assert len(list_key) == len(list_value), 'the length of two input lists are not equal'

	return dict(zip(list_key, list_value))

######################################################### scalar related #########################################################
def float2percent(number, warning=True, debug=True):
	'''
	convert a floating number to a string representing percentage
	'''
	try: number = float(number)
	except ValueError: 
		if warning: print('could not convert to a floating number')
	return '{:.1%}'.format(number)

def number2onehot(number, ranges, debug=True):
	'''
	this function convert an integer number to a one hot vector
	parameters:
		number:			an integer
		ranges:			[min, max], inclusive, both are integers
	'''
	if debug:
		assert isinteger(number), 'input number is not an integer'
		assert isrange(ranges), 'input range is not correct'
		assert isinteger(ranges[0]) and isinteger(ranges[1]), 'the input range should be integer'

	num_integers = ranges[1] - ranges[0] + 1
	index = number - ranges[0]
	onehot = np.zeros([num_integers, ], dtype='float32')
	onehot[index] = 1
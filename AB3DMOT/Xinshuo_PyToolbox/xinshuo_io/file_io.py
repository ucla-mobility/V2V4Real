# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys, time, glob, glob2, numpy as np

from xinshuo_miscellaneous.private import safe_path
from xinshuo_miscellaneous import string2ext_filter, remove_empty_item_from_list, str2num, is_path_exists_or_creatable, is_path_exists, isfolder, isnparray, is2dptsarray, is2dptsarray_occlusion, islogical, isinteger, islist, isstring

def fileparts(input_path, warning=True, debug=True):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed

	parameters:
		input_path:     a string path

	outputs:
		directory:      the parent directory
		filename:       the file name without extension
		ext:            the extension
	'''
	good_path = safe_path(input_path, debug=debug)
	if len(good_path) == 0: return ('', '', '')
	if good_path[-1] == '/':
		if len(good_path) > 1: return (good_path[:-1], '', '')	# ignore the final '/'
		else: return (good_path, '', '')	                          # ignore the final '/'
	
	directory = os.path.dirname(os.path.abspath(good_path))
	filename = os.path.splitext(os.path.basename(good_path))[0]
	ext = os.path.splitext(good_path)[1]
	return (directory, filename, ext)

def mkdir_if_missing(input_path, warning=True, debug=True):
	'''
	create a directory if not existing:
		1. if the input is a path of file, then create the parent directory of this file
		2. if the root directory does not exists for the input, then create all the root directories recursively until the parent directory of input exists

	parameters:
		input_path:     a string path
	'''	
	good_path = safe_path(input_path, warning=warning, debug=debug)
	if debug: assert is_path_exists_or_creatable(good_path), 'input path is not valid or creatable: %s' % good_path
	dirname, _, _ = fileparts(good_path)
	if not is_path_exists(dirname): mkdir_if_missing(dirname)
	if isfolder(good_path) and not is_path_exists(good_path): os.mkdir(good_path)

######################################################### dict related #########################################################
def save_struct(struct_save, save_path, debug_mode):
    with open(save_path, 'w') as f:    
        for k, v in struct_save.__dict__.items(): f.write('%s    %s\n' % (k, v))

######################################################### txt IO #########################################################
def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safe_path(file_path)
    if debug: assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file: data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines

def combine_txt_file(file_path_list, save_path=None, debug=True):
    '''
    combine txt file line by line
    '''
    if debug: assert islist(file_path_list), 'input is not a list'
    
    data = []
    num_lines = 0
    for file_path in file_path_list:
        data_tmp, num_lines_tmp = load_txt_file(file_path, debug=debug)

        num_lines += num_lines_tmp
        data += data_tmp

    if save_path is not None:
        save_txt_file(data, save_path, debug=debug)

    return data, num_lines

def save_txt_file(data_list, save_path, debug=True):
    '''
    save a list of string to a file
    '''
    save_path = safe_path(save_path)
    if debug: assert is_path_exists_or_creatable(save_path), 'text file is not able to be created at path: %s!' % save_path

    first_line = True
    with open(save_path, 'w') as file:
        for item in data_list:
            if first_line:
                file.write('%s' % item)
                first_line = False
            else: file.write('\n%s' % item)
    file.close()

######################################################### list IO #########################################################
def load_list_from_file(file_path, debug=True):
    '''
    this function reads list from a txt file
    '''
    file_path = safe_path(file_path)
    _, _, extension = fileparts(file_path)

    if debug: assert extension == '.txt' or '.lst', 'File doesn''t have valid extension.'
    file = open(file_path, 'r')
    if debug: assert file != -1, 'datalist not found'

    fulllist = file.read().splitlines()
    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)
    file.close()

    return fulllist, num_elem

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    '''
    load a list of files or folders from a system path

    parameters:
        folder_path:    root to search 
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels till to the input depth

    outputs:
        fulllist:       a list of elements
        num_elem:       number of the elements
    '''
    folder_path = safe_path(folder_path)
    if debug: assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path): 
        print('the input folder does not exist %s\n' % folder_path)
        return [], 0
    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter): ext_filter = [ext_filter]                               # convert to a list
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist: file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

def load_list_from_folders(folder_path_list, ext_filter=None, depth=1, recursive=False, save_path=None, debug=True):
    '''
    load a list of files or folders from a list of system path
    '''
    if debug: assert islist(folder_path_list) or isstring(folder_path_list), 'input path list is not correct'
    if isstring(folder_path_list): folder_path_list = [folder_path_list]

    fulllist = list()
    num_elem = 0
    for folder_path_tmp in folder_path_list:
        fulllist_tmp, num_elem_tmp = load_list_from_folder(folder_path_tmp, ext_filter=ext_filter, depth=depth, recursive=recursive)
        fulllist += fulllist_tmp
        num_elem += num_elem_tmp

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist: file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

def generate_list_from_data(save_path, src_data, debug=True):
    '''
    generate a file which contains a 1-d numpy array data

    parameter:
        src_data:   a list of 1 element data, or a 1-d numpy array data
    '''
    save_path = safe_path(save_path)

    if debug:
        if isnparray(src_data): assert src_data.ndim == 1, 'source data is incorrect'
        elif islist(src_data): assert all(np.array(data_tmp).size == 1 for data_tmp in src_data), 'source data is in correct'
        assert isfolder(save_path) or isfile(save_path), 'save path is not correct'
        
    if isfolder(save_path): save_path = os.path.join(save_path, 'datalist.txt')
    if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
    with open(save_path, 'w') as file:
        for item in src_data: file.write('%f\n' % item)
    file.close()

######################################################### numpy IO #########################################################
def save_2dmatrix_to_file(data, save_path, formatting='%.1f', debug=True):
    save_path = safe_path(save_path)
    if debug:
        assert isnparray(data) and len(data.shape) == 2, 'input data is not 2d numpy array'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
        mkdir_if_missing(save_path)
        # assert isnparray(data) and len(data.shape) <= 2, 'the data is not correct'
        
    np.savetxt(save_path, data, delimiter=' ', fmt=formatting)

def load_2dmatrix_from_file(src_path, delimiter=' ', dtype='float32', debug=True):
    src_path = safe_path(src_path)
    if debug: assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path

    data = np.loadtxt(src_path, delimiter=delimiter, dtype=dtype)
    nrows = data.shape[0]
    return data, nrows
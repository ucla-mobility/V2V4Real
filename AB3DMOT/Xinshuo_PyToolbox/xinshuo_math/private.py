# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import copy, numpy as np
from xinshuo_miscellaneous import islist, isnparray, isbbox, islistoflist, iscenterbbox, isscalar, is2dptsarray
from xinshuo_miscellaneous import is2dptsarray_occlusion, is2dptsarray_homogeneous, is3dptsarray, is4dptsarray

################################################################## conversion ##################################################################
def safe_npdata(input_data, warning=True, debug=True):
	'''
	copy a list of data or a numpy data to the buffer for use

	parameters:
		input_data:		a list, a scalar or numpy data

	outputs:
		np_data:		a copy of numpy data
	'''
	if islist(input_data): 		np_data = np.array(input_data)
	elif isscalar(input_data): 	np_data = np.array(input_data).reshape((1, ))
	elif isnparray(input_data): np_data = input_data.copy()
	else: assert False, 'only list of data, scalar or numpy data are supported'

	return np_data

def safe_bbox(input_bbox, warning=True, debug=True):
	'''
	make sure to copy the bbox without modifying it and make the dimension to N x 4

	parameters:
		input_bbox: 	a list of 4 elements, a listoflist of 4 elements: e.g., [[1,2,3,4], [5,6,7,8]],
						a numpy array with shape or (N, 4) or (4, )

	outputs:
		np_bboxes:		N X 4 numpy array
	'''
	if islist(input_bbox):
		if islistoflist(input_bbox):
			if debug: assert all(len(list_tmp) == 4 for list_tmp in input_bbox), 'all sub-lists should have length of 4'
			np_bboxes = np.array(input_bbox)
		else:
			if debug: assert len(input_bbox) == 4, 'the input bbox list does not have a good shape'
			np_bboxes = np.array(input_bbox).reshape((1, 4))
	elif isnparray(input_bbox):
		input_bbox = input_bbox.copy()
		if input_bbox.shape == (4, ):
			np_bboxes = input_bbox.reshape((1, 4))
		else:
			if debug: assert isbbox(input_bbox), 'the input bbox numpy array does not have a good shape'
			np_bboxes = input_bbox
	else: assert False, 'only list and numpy array for bbox are supported'

	return np_bboxes

def safe_center_bbox(input_bbox, warning=True, debug=True):
	'''
	make sure to copy the center bbox without modifying it and make the dimension to N x 4 or N x 2

	parameters:
		input_bbox: 	a list of 4 (2) elements, a listoflist of 4 (2) elements: e.g., [[1,2,3,4], [5,6,7,8]],
						a numpy array with shape or (N, 4) or (4, ) or (N, 2) or (2, )

	outputs:
		np_bboxes:		N X 4 (2) numpy array
	'''
	if islist(input_bbox):
		if islistoflist(input_bbox):
			if debug:
				assert all(len(list_tmp) == 4 or len(list_tmp) == 2 for list_tmp in input_bbox), 'all sub-lists should have length of 4'
			np_bboxes = np.array(input_bbox)
		else:
			if debug:
				assert len(input_bbox) == 4 or len(input_bbox) == 2, 'the center bboxes list does not have a good shape'
			if len(input_bbox) == 4: np_bboxes = np.array(input_bbox).reshape((1, 4))
			else: np_bboxes = np.array(input_bbox).reshape((1, 2))
	elif isnparray(input_bbox):
		input_bbox = input_bbox.copy()
		if input_bbox.shape == (4, ): np_bboxes = input_bbox.reshape((1, 4))
		elif input_bbox.shape == (2, ): np_bboxes = input_bbox.reshape((1, 2))
		else:
			if debug: assert iscenterbbox(input_bbox), 'the input center bbox numpy array does not have a good shape'
			np_bboxes = input_bbox
	else:
		assert False, 'only list and numpy array for bbox are supported'

	return np_bboxes

def safe_angle(input_angle, radian=False, warning=True, debug=True):
	'''
	make ensure the rotation is in [-180, 180] in degree

	parameters:
		input_angle:	an angle which is supposed to be in degree
		radian:			if True, the unit is replaced to radian instead of degree

	outputs:
		angle:			an angle in degree within (-180, 180]
	'''
	angle = copy.copy(input_angle)
	if debug:
		assert isscalar(angle), 'the input angle should be a scalar'

	if isnparray(angle): angle = angle[0]		# single numpy scalar value
	if radian:
		while angle > np.pi: angle -= np.pi
		while angle <= -np.pi: angle += np.pi
	else:	
		while angle > 180: angle -= 360
		while angle <= -180: angle += 360

	return angle
	
def safe_2dptsarray(input_pts, homogeneous=False, dimen_add=0, warning=True, debug=True):
	'''
	make sure to copy the pts array without modifying it and make the dimension to 2(3 if homogenous) x N

	parameters:
		input_pts: 		a list of 2(3 if homogenous) elements, a listoflist of 2 elements: 
						e.g., [[1,2], [5,6]], a numpy array with shape or (2, N) or (2, )
		homogeneous:		the input points are in the homogenous coordinate
		dimen_add:		additional dimension, used to accommdate for higher dimensional array
	
	outputs:
		np_pts:			2 (3 if homogenous) X N numpy array
	'''
	if homogeneous: dimension = 3 + dimen_add
	else: dimension = 2 + dimen_add

	if islist(input_pts):
		if islistoflist(input_pts):
			if debug: assert all(len(list_tmp) == dimension for list_tmp in input_pts), 'all sub-lists should have length of %d' % dimension
			np_pts = np.array(input_pts).transpose()
		else:
			if debug: assert len(input_pts) == dimension, 'the input pts list does not have a good shape'
			np_pts = np.array(input_pts).reshape((dimension, 1))
	elif isnparray(input_pts):
		input_pts = input_pts.copy()
		if input_pts.shape == (dimension, ):
			np_pts = input_pts.reshape((dimension, 1))
		else:
			np_pts = input_pts
	else: assert False, 'only list and numpy array for pts are supported'

	if debug: 
		if homogeneous: assert is2dptsarray_homogeneous(np_pts), 'the input pts array does not have a good shape'
		else: assert is2dptsarray(np_pts), 'the input pts array does not have a good shape'

	return np_pts

def safe_3dptsarray(input_pts, homogeneous=False, warning=True, debug=True):
	'''
	make sure to copy the pts array without modifying it and make the dimension to 3 x N

	parameters:
		input_pts: 	a list of 3 elements, a listoflist of 3 elements: e.g., [[1,2], [5,6], [0, 1]],
						a numpy array with shape or (3, N) or (3, )

	outputs:
		np_pts:		3 X N numpy array
	'''
	np_pts = safe_2dptsarray(input_pts, homogeneous=False, dimen_add=1, warning=warning, debug=False)
	if debug: assert is3dptsarray(np_pts), 'the input pts array does not have a good shape'
	return np_pts	

def safe_4dptsarray(input_pts, homogeneous=False, warning=True, debug=True):
	'''
	make sure to copy the pts array without modifying it and make the dimension to 4 x N

	parameters:
		input_pts: 	a list of 3 elements, a listoflist of 3 elements: e.g., [[1,2], [5,6], [0, 1]],
						a numpy array with shape or (4, N) or (4, )

	outputs:
		np_pts:		4 X N numpy array
	'''
	np_pts = safe_2dptsarray(input_pts, homogeneous=False, dimen_add=2, warning=warning, debug=False)
	if debug: assert is4dptsarray(np_pts), 'the input pts array does not have a good shape'
	return np_pts	

def safe_2dptsarray_occlusion(input_pts, warning=True, debug=True):
	'''
	make sure to copy the pts array without modifying it and make the dimension to 3 x N
	the occlusion (3rd) row should contain 0, 1 or -1

	parameters:
		input_pts: 	a list of 3 elements, a listoflist of 3 elements: e.g., [[1,2], [5,6], [0, 1]],
						a numpy array with shape or (3, N) or (3, )

	outputs:
		np_pts:		3 X N numpy array, with the third row as the occlusion
	'''
	np_pts = safe_2dptsarray(input_pts, homogeneous=True, warning=warning, debug=debug)
	if debug: assert is2dptsarray_occlusion(np_pts), 'the input pts array does not have a good shape'
	return np_pts	

################################################################## sanity check ##################################################################
def bboxcheck_TLBR(input_bbox, warning=True, debug=True):
    '''
    check the input bounding box to be TLBR format

    parameters:
        input_bbox:   TLBR format, a list of 4 elements, a listoflist of 4 elements: e.g., [[1,2,3,4], [5,6,7,8]], a numpy array with shape or (N, 4) or (4, )
    
    outputs:
        True if the x2 > x1 and y2 > y1
    '''
    np_bboxes = safe_bbox(input_bbox, warning=warning, debug=debug)
    if debug: assert isbbox(np_bboxes), 'the input bboxes are not good'

    return (np_bboxes[:, 3] >= np_bboxes[:, 1]).all() and (np_bboxes[:, 2] >= np_bboxes[:, 0]).all()      # coordinate of bottom right point should be larger or equal than top left point

def bboxcheck_TLWH(input_bbox, warning=True, debug=True):
	'''
	check the input bounding box to be TLBR format

	parameters:
	    input_bbox:   TLBR format, a list of 4 elements, a listoflist of 4 elements: e.g., [[1,2,3,4], [5,6,7,8]], a numpy array with shape or (N, 4) or (4, )

	outputs:
	    True if the width and height are >= 0
	'''
	np_bboxes = safe_bbox(input_bbox, warning=warning, debug=debug)
	if debug: assert isbbox(np_bboxes), 'the input bboxes are not good'

	return (np_bboxes[:, 3] >= 0).all() and (np_bboxes[:, 2] >= 0).all()      # coordinate of bottom right point should be larger or equal than top left point
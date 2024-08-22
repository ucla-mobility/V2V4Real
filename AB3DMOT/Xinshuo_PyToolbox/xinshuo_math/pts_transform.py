# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes transformation for point clouds
import copy, numpy as np
from scipy.spatial import Delaunay

from .private import safe_3dptsarray, safe_4dptsarray
from xinshuo_miscellaneous import isnonnegativeinteger

################################################################## conversion ##################################################################
def point_sample(pts, sample_pts, shuffle=True, warning=True, debug=True):
	'''
	sample points from a point cloud data

	parameters:
		pts:			3 (or 4) x N, numpy array
		sample_pts:		integer

	outputs:
		pts_sampled:	3 (or 4) x M, numpy array
	'''
	try:
		pts = safe_3dptsarray(pts, warning=warning, debug=debug)
	except AssertionError:
		pts = safe_4dptsarray(pts, warning=warning, debug=debug)

	if debug: assert isnonnegativeinteger(sample_pts), 'the input pts to sample is not correct'

	if sample_pts > 0:
		num_pts = pts.shape[1]
		if num_pts <= 0: 
			if warning: print('warning: zero points. No sampling')
			return pts

		if num_pts >= sample_pts: 
			choice = np.random.choice(num_pts, sample_pts, replace=False)
			choice.sort()
		else:
			choice = np.random.choice(num_pts, sample_pts - num_pts, replace=True)
			choice = np.concatenate((np.arange(num_pts), choice))
			choice.sort()

		if shuffle: np.random.shuffle(choice)
		pts_return = pts[:, choice]
	else: 
		if warning: print('number of points to sample is negative. No sampling')
	return pts_return

def in_hull(p, hull):
	# p: 		3 x N
	if not isinstance(hull, Delaunay): hull = Delaunay(hull)
	return hull.find_simplex(p.transpose()) >= 0

def extract_pc_in_box3d(pts, box3d, warning=True, debug=True):
	''' 
	extract points within a 3D bounding box

	parameters:
		pts:			3 (or 4) x N, numpy array
		box3d:			8 x 3, numpy array, 8 corners

	outputs
		pc_extracted:	3 (or 4) x M, numpy array
		ids:			(M, ), numpy array

	'''
	try:
		pts = safe_3dptsarray(pts, warning=warning, debug=debug)
	except AssertionError:
		pts = safe_4dptsarray(pts, warning=warning, debug=debug)

	assert box3d.shape == (8, 3), '3D bounding box shape is incorrect'

	box3d_roi_inds = in_hull(pts[0:3, :], box3d)
	return pts[:, box3d_roi_inds], box3d_roi_inds

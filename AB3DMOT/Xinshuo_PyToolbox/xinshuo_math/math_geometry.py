# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic geometry in math
import math, numpy as np, warnings
# warnings.filterwarnings("error")

from .private import safe_2dptsarray, safe_npdata
from xinshuo_miscellaneous import is2dpts, is2dhomopts, is2dhomoline, is3dpts, isscalar

# homogeneous representation
# 2D line representation:           ax + by + c = 0,            vector representation: (a, b, c)
# 2D pts representation:            (x, y),                     vector representation: (x, y, z)
# 3D plane representation:          ax + by + cz + d = 0,       vector representation: (a, b, c, d)
# 3D pts representation:            (x, y, z),                  vector representation: (x, y, z, t)

################################################################## 2d planar geomemtry ##################################################################
def get_2dline_from_pts(input_pts1, input_pts2, warning=True, debug=True):
	'''
	get the homogeneous line representation from two 2d homogeneous points

	parameters:
		input_pts1:         a homogeneous 2D point, can be a list or tuple or numpy array: (x, y, z)
		input_pts2:         a homogeneous 2D point, can be a list or tuple or numpy array: (x, y, z)

	outputs:
		np_line:            a homogeneous 2D line,  can be a list or tuple or numpy array: 3 x 1, (a, b, c)
	'''
	np_pts1 = safe_2dptsarray(input_pts1, homogeneous=True, warning=warning, debug=debug)
	np_pts2 = safe_2dptsarray(input_pts2, homogeneous=True, warning=warning, debug=debug)
	if debug: assert is2dhomopts(np_pts1) and is2dhomopts(np_pts2), 'point is not correct'
	np_line = np.cross(np_pts1.transpose(), np_pts2.transpose()).transpose()

	return np_line

def get_2dpts_from_lines(input_line1, input_line2, warning=True, debug=True):
	'''
	get the homogeneous point representation from two 2d homogeneous lines

	parameters:
		input_line1:         a homogeneous 2D line, can be a list or tuple or numpy array: (a, b, c)
		input_line2:         a homogeneous 2D line, can be a list or tuple or numpy array: (a, b, c)

	outputs:
		np_pts:              a homogeneous 2D point,  can be a list or tuple or numpy array: 3 x 1, (a, b, c)
	'''
	np_pts = get_2dline_from_pts(input_line1, input_line2, warning=warning, debug=debug)
	return np_pts

def get_2dline_from_pts_slope(input_pts, slope, warning=True, debug=True):
	'''
	get the homogeneous line representation from two a homogeneous point and the slope in degree

	parameters:
		input_pts1:         a homogeneous 2D point, can be a list or tuple or numpy array: (x, y, z)
		slope:              a scalar in degree

	outputs:
		np_line:            a homogeneous 2D line,  can be a list or tuple or numpy array: 3 x 1, (a, b, c)
	'''
	np_pts1 = safe_2dptsarray(input_pts, homogeneous=True, warning=warning, debug=debug)
	if debug:
		assert is2dhomopts(np_pts1), 'point is not correct'
		assert isscalar(slope), 'the slope is not correct'

	y = math.sin(math.radians(slope))       # math.tan can handle 90 or -90
	x = math.cos(math.radians(slope))       # math.tan can handle 90 or -90
	np_pts2 = np.array([x, y, 0]).reshape((3, 1))       # this equation is obtained from slope
	np_line = get_2dline_from_pts(np_pts1, np_pts2, warning=warning, debug=debug)

	return np_line

def get_slope_from_pts(input_pts1, input_pts2, warning=True, debug=True):
	'''
	get the slope in degree from two 2d homogeneous points

	parameters:
		input_pts1:         a homogeneous 2D point, can be a list or tuple or numpy array: (x, y, z)
		input_pts2:         a homogeneous 2D point, can be a list or tuple or numpy array: (x, y, z)

	outputs:
		slope:            	a scalar in degree
	'''
	np_line = get_2dline_from_pts(input_pts1, input_pts2, warning=warning, debug=debug)

	try: 
		slope = - np_line[0] / np_line[1]
	except ZeroDivisionError: slope = float('inf')
	except RuntimeWarning: slope = float('inf')
	slope = math.degrees(np.arctan(slope))
	
	return slope

################################################################## 3d geometry ##################################################################
def generate_sphere(pts_3d, radius, debug=True):
    '''
    generate a boundary of a 3D shpere point cloud
    '''
    if debug:
        assert is3dpts(pts_3d), 'the input point is not a 3D point'

    num_pts = 100
    u = np.random.rand(num_pts, )
    v = np.random.rand(num_pts, )

    print(u.shape)
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    
    pts_shpere = np.zeros((3, num_pts), dtype='float32')
    pts_shpere[0, :] = pts_3d[0] + radius * math.sin(phi) * math.cos(theta)
    pts_sphere[1, :] = pts_3d[1] + radius * math.sin(phi) * math.sin(theta)
    pts_sphere[2, :] = pts_3d[2] + radius * math.cos(phi)

    return pts_sphere

def construct_3drotation_matrix_rodrigue(axis, theta):
	axis = axis / np.sqrt(np.dot(axis, axis))
	a = np.cos(theta / 2.)
	b, c, d = -axis * np.sin(theta / 2.)
	return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
					 [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
					 [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

# def generate_ply_

################################################################## homogeneous vs euclidean ##################################################################
def homogeneous2euclidean(homo_input, warning=True, debug=True):
	pass

def euclidean2homogeneous(input, warning=True, debug=True):
	pass
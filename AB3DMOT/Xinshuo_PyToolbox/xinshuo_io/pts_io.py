# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions about point file I/O stream
import numpy as np

from .file_io import load_txt_file
from xinshuo_miscellaneous import is_path_exists_or_creatable, is2dptsarray_occlusion, is2dptsarray_confidence, is2dptsarray, remove_empty_item_from_list, str2num

# note that, the top left point is (1, 1) in 300-W instead of zero-indexed
def anno_writer(pts_array, pts_savepath, num_pts=68, anno_version=1, debug=True):
    '''
    write the point array to a .pts file
    parameter:
        pts_array:      2 or 3 x num_pts numpy array
        
    '''
    if debug:
        assert is_path_exists_or_creatable(pts_savepath), 'the save path is not correct'
        assert (is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array) or is2dptsarray_confidence(pts_array)) and pts_array.shape[1] == num_pts, 'the input point is not correct'

    with open(pts_savepath, 'w') as file:
        file.write('version: %d\n' % anno_version)
        file.write('n_points: %d\n' % num_pts)
        file.write('{\n')

        # main content
        for pts_index in xrange(num_pts):
            if is2dptsarray(pts_array):
                file.write('%.3f %.3f %f\n' % (pts_array[0, pts_index], pts_array[1, pts_index], 1.0))      # all visible
            else:                           
                file.write('%.3f %.3f %f\n' % (pts_array[0, pts_index], pts_array[1, pts_index], pts_array[2, pts_index]))

        file.write('}')
        file.close()

def anno_parser(anno_path, num_pts=None, anno_version=None, warning=True, debug=True):
    '''
    parse the annotation for Keypoint file
    return:
        pts_array: 3 x num_pts (x, y, oculusion)          
    '''
    data, num_lines = load_txt_file(anno_path, debug=debug)
    # print(data)
    if debug:
        assert data[0].find('version: ') == 0, 'version is not correct: %s' % anno_path
        assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'
        assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'
    version = str2num(data[0][len('version: '):])
    n_points = int(data[1][len('n_points: '):])

    if debug:
        # print('version of annotation is %d' % version)
        # print('number of points is %d' % n_points)
        assert num_lines == n_points + 4, 'number of lines is not correct'      # 4 lines for general information: version, n_points, start and end symbol
        if anno_version is not None:
            assert version == anno_version, 'version of annotation is not correct: %d vs %d' % (version, anno_version)
        if num_pts is not None:
            assert num_pts == n_points, 'number of points is not correct: %d vs %d' % (num_pts, n_points)

    # print(anno_path)

    # read points coordinate
    pts_array = np.zeros((3, n_points), dtype='float32')
    line_offset = 3     # first point starts at fourth line
    for point_index in xrange(n_points):
        try:
            pts_list = data[point_index + line_offset].split(' ')           # x y format
            if len(pts_list) > 2 and pts_list[2] == '':     # handle edge case where additional whitespace exists after point coordinates
                pts_list = remove_empty_item_from_list(pts_list)

            # print(pts_list[0])
            pts_array[0, point_index] = float(pts_list[0])
            pts_array[1, point_index] = float(pts_list[1])
            if len(pts_list) == 3:
                pts_array[2, point_index] = float(pts_list[2])
            else:
                pts_array[2, point_index] = float(1)          # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by LS3D-W
        except ValueError:
            print('error in loading points in %s' % anno_path)
    return pts_array
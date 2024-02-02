# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, os, matplotlib.pyplot as plt, colorsys, random, matplotlib.patches as patches
import matplotlib.collections as plycollections
from matplotlib.patches import Ellipse
from skimage.measure import find_contours
# from scipy.stats import norm, chi2

# import matplotlib as mpl; mpl.use('Agg')
# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# from warnings import catch_warnings, simplefilter
# with catch_warnings(record=True):
    # simplefilter('ignore', FutureWarning)
from .private import save_vis_close_helper, get_fig_ax_helper
from xinshuo_math.private import safe_2dptsarray, safe_bbox
from xinshuo_math import pts_euclidean, bbox_TLBR2TLWH, bboxcheck_TLBR
from xinshuo_miscellaneous import islogical, islist, isstring, is2dptsarray_confidence, is2dptsarray_occlusion, is2dptsarray, isdict, list_reorder, list2tuple, islistofstring, ifconfscalar, isscalar, isnparray
from xinshuo_io import mkdir_if_missing, save_image

color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'lime', 'cyan', 'aqua']
color_set_big = ['aqua', 'azure', 'red', 'black', 'blue', 'brown', 'cyan', 'darkblue', 'fuchsia', 'gold', 'green', 'grey', 'indigo', 'magenta', 'lime', 'yellow', 'white', 'tomato', 'salmon']
marker_set = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
hatch_set = [None, 'o', '/', '\\', '|', '-', '+', '*', 'x', 'O', '.']
linestyle_set = ['-', '--', '-.', ':', None, ' ', 'solid', 'dashed']
dpi = 80

def visualize_bbox(input_bbox, linewidth=0.5, edge_color_index=15, scores=None, threshold=0.0, textsize=8,
    fig=None, ax=None, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    visualize a set of bounding box

    parameters:
        input_bbox:     a list of 4 elements, a listoflist of 4 elements: e.g., [[1,2,3,4], [5,6,7,8]],
                        a numpy array with shape or (N, 4) or (4, )
                        TLBR format
        scores:         a list of floating numbers representing the confidences
    '''
    if islist(input_bbox) and len(input_bbox) == 0: 
        return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig)
    elif isnparray(input_bbox) and input_bbox.size == 0:
        return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig)

    np_bboxes = safe_bbox(input_bbox, warning=warning, debug=debug)
    if debug: assert bboxcheck_TLBR(np_bboxes, warning=warning, debug=debug), 'input bounding boxes are not correct'
    edge_color = color_set_big[edge_color_index % len(color_set_big)]

    np_bboxes = bbox_TLBR2TLWH(np_bboxes, warning=warning, debug=debug)              # convert TLBR format to TLWH format
    for bbox_index in range(np_bboxes.shape[0]):
        bbox_tmp = np_bboxes[bbox_index, :]     
        if scores is not None:
            score = float(scores[bbox_index])
            if score < threshold: continue
            caption = '{:.2f}'.format(score)

            # score = str(scores[bbox_index])
            # caption = '%s' % (score)

            ax.text(bbox_tmp[0], bbox_tmp[1] + textsize, caption, color='r', size=textsize, backgroundcolor='none')

        ax.add_patch(plt.Rectangle((bbox_tmp[0], bbox_tmp[1]), bbox_tmp[2], bbox_tmp[3], fill=False, edgecolor=edge_color, linewidth=linewidth))
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig)

def visualize_pts_array(input_pts, color_index=0, pts_size=20, label=False, label_list=None, label_size=20, vis_threshold=0.3, 
    covariance=False, plot_occl=False, xlim=None, ylim=None, 
    fig=None, ax=None, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    plot keypoints with covariance ellipse

    parameters:
        pts_array:      2(3) x num_pts numpy array, the third channel could be confidence or occlusion
    '''
    # obtain the points
    try: pts_array = safe_2dptsarray(input_pts, homogeneous=True, warning=warning, debug=debug)
    except AssertionError: pts_array = safe_2dptsarray(input_pts, homogeneous=False, warning=warning, debug=debug)
    if debug: assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array) or is2dptsarray_confidence(pts_array), 'input points are not correct'
    num_pts = pts_array.shape[1]

    # obtain a label list if required but not provided
    if debug: assert islogical(label), 'label flag is not correct'
    if label and (label_list is None): label_list = [str(i) for i in xrange(num_pts)]
    if label_list is not None and debug: assert islistofstring(label_list), 'labels are not correct'

    # obtain the color index
    if islist(color_index):
        if debug: assert not (plot_occl or covariance) , 'the occlusion or covariance are not compatible with plotting different colors during scattering'
        color_tmp = [color_set_big[index_tmp] for index_tmp in color_index]
    else: color_tmp = color_set_big[color_index % len(color_set_big)]
    
    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)
    std, conf = None, 0.95
    if is2dptsarray(pts_array):             # only 2d points without third rows
        if debug and islist(color_tmp): assert len(color_tmp) == num_pts, 'number of points to plot is not equal to number of colors provided'
        ax.scatter(pts_array[0, :], pts_array[1, :], color=color_tmp, s=pts_size)
        pts_visible_index = range(pts_array.shape[1])
        pts_ignore_index = []
        pts_invisible_index = []
    else:
        # automatically justify if the third row is confidence or occlusion flag
        num_float_elements = np.where(np.logical_and(pts_array[2, :] != -1, np.logical_and(pts_array[2, :] != 0, pts_array[2, :] != 1)))[0].tolist()
        if len(num_float_elements) > 0: type_3row = 'conf'
        else: type_3row = 'occu'

        if type_3row == 'occu':
            pts_visible_index   = np.where(pts_array[2, :] == 1)[0].tolist()              # plot visible points in red color
            pts_ignore_index    = np.where(pts_array[2, :] == -1)[0].tolist()             # do not plot points with annotation, usually visible, but not annotated
            pts_invisible_index = np.where(pts_array[2, :] == 0)[0].tolist()              # plot invisible points in blue color
        else:
            pts_visible_index   = np.where(pts_array[2, :] > vis_threshold)[0].tolist()
            pts_invisible_index    = np.where(pts_array[2, :] <= vis_threshold)[0].tolist()
            pts_ignore_index = []

        if debug and islist(color_tmp): assert len(color_tmp) == len(pts_visible_index), 'number of points to plot is not equal to number of colors provided'
        ax.scatter(pts_array[0, pts_visible_index], pts_array[1, pts_visible_index], color=color_tmp, s=pts_size)
        if plot_occl: ax.scatter(pts_array[0, pts_invisible_index], pts_array[1, pts_invisible_index], color=color_set_big[(color_index+1) % len(color_set_big)], s=pts_size)
        if covariance: visualize_pts_covariance(pts_array[0:2, :], std=std, conf=conf, fig=fig, ax=ax, debug=debug, color=color_tmp)

    if plot_occl: not_plot_index = pts_ignore_index
    else: not_plot_index = pts_ignore_index + pts_invisible_index
    if label_list is not None:
        for pts_index in xrange(num_pts):
            label_tmp = label_list[pts_index]
            if pts_index in not_plot_index: continue
            else:
                # note that the annotation is based on the coordinate instead of the order of plotting the points, so the orider in pts_index does not matter
                if islist(color_index): plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index[pts_index]+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
                else: plt.annotate(label_tmp, xy=(pts_array[0, pts_index], pts_array[1, pts_index]), xytext=(-1, 1), color=color_set_big[(color_index+5) % len(color_set_big)], textcoords='offset points', ha='right', va='bottom', fontsize=label_size)
    
    # set axis
    if xlim is not None:
        if debug: assert islist(xlim) and len(xlim) == 2, 'the x lim is not correct'
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:    
        if debug: assert islist(ylim) and len(ylim) == 2, 'the y lim is not correct'
        plt.ylim(ylim[0], ylim[1])

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig, transparent=False)

def visualize_lines(lines_array, color_index=0, line_width=3, fig=None, ax=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    plot lines 

    parameters:
        lines_array:            4 x num_lines, each column denotes (x1, y1, x2, y2)
    '''
    if debug: assert islinesarray(lines_array), 'input array of lines are not correct'
    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)

    # plot lines
    num_lines = lines_array.shape[1]
    lines_all = []
    for line_index in range(num_lines):
        line_tmp = lines_array[:, line_index]
        lines_all.append([tuple([line_tmp[0], line_tmp[1]]), tuple([line_tmp[2], line_tmp[3]])])

    line_col = plycollections.LineCollection(lines_all, linewidths=line_width, colors=color_set[color_index])
    ax.add_collection(line_col)
        # ax.plot([line_tmp[0], line_tmp[2]], [line_tmp[1], line_tmp[3]], color=color_set[color_index], linewidth=line_width)

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig)

def visualize_pts_line(pts_array, line_index_list, method=2, seed=0, alpha=0.5,
    vis_threshold=0.3, pts_size=20, line_size=10, line_color_index=0, 
    fig=None, ax=None, save_path=None, vis=False, warning=True, debug=True, closefig=True):
    '''
    given a list of index, and a point array, to plot a set of points with line on it

    parameters:
        pts_array:          2(3) x num_pts
        line_index_list:    a list of index
        method:             1: all points are connected, if some points are missing in the middle, just ignore that point and connect the two nearby points
                            2: if some points are missing in the middle of a line, the line is decomposed to sub-lines
        vis_threshold:      confidence to draw the points

    '''
    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array) or is2dptsarray_confidence(pts_array), 'input points are not correct'
        assert islist(line_index_list), 'the list of index is not correct'
        assert method in [1, 2], 'the plot method is not correct'

    num_pts = pts_array.shape[1]
    # expand the pts_array to 3 rows if the confidence row is not provided
    if pts_array.shape[0] == 2: pts_array = np.vstack((pts_array, np.ones((1, num_pts))))
    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)
    np.random.seed(seed)
    color_option = 'hsv'

    if color_option == 'rgb': color_set_random = np.random.rand(3, num_pts)
    elif color_option == 'hsv':
        h_random = np.random.rand(num_pts, )
        color_set_random = np.zeros((3, num_pts), dtype='float32')
        for pts_index in range(num_pts): color_set_random[:, pts_index] = colorsys.hsv_to_rgb(h_random[pts_index], 1, 1) 

    line_color = color_set[line_color_index]
    pts_line = pts_array[:, line_index_list]

    if method == 1:    
        valid_pts_list = np.where(pts_line[2, :] > vis_threshold)[0].tolist()
        pts_line_tmp = pts_line[:, valid_pts_list]
        ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)      # plot all lines

        # plot all points
        for pts_index in valid_pts_list:
            pts_index_original = line_index_list[pts_index]
            # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha)
            ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index], alpha=alpha)
    else:
        not_valid_pts_list = np.where(pts_line[2, :] < vis_threshold)[0].tolist()
        if len(not_valid_pts_list) == 0:            # all valid
            ax.plot(pts_line[0, :], pts_line[1, :], lw=line_size, color=line_color, alpha=alpha)

            # plot points
            for pts_index in line_index_list:
                # ax.plot(pts_array[0, pts_index], pts_array[1, pts_index], 'o', color=color_set_big[pts_index % len(color_set_big)], alpha=alpha)
                ax.plot(pts_array[0, pts_index], pts_array[1, pts_index], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index], alpha=alpha)
        else:
            prev_index = 0
            for not_valid_index in not_valid_pts_list:
                plot_list = range(prev_index, not_valid_index)
                pts_line_tmp = pts_line[:, plot_list]
                ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)
                
                # plot points
                for pts_index in plot_list:
                    pts_index_original = line_index_list[pts_index]
                    ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index_original], alpha=alpha) 
                    # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha) 

                prev_index = not_valid_index + 1

            pts_line_tmp = pts_line[:, prev_index:]
            ax.plot(pts_line_tmp[0, :], pts_line_tmp[1, :], lw=line_size, color=line_color, alpha=alpha)      # plot last line

            # plot last points
            for pts_index in range(prev_index, pts_line.shape[1]):
                pts_index_original = line_index_list[pts_index]
                # ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], 'o', color=color_set_big[pts_index_original % len(color_set_big)], alpha=alpha) 
                ax.plot(pts_array[0, pts_index_original], pts_array[1, pts_index_original], marker='o', ms=pts_size, lw=line_size, color=color_set_random[:, pts_index_original], alpha=alpha) 

    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig)

def visualize_pts_covariance(pts_array, conf=None, std=None, fig=None, ax=None, debug=True, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        pts_array       : 2 x N numpy array of the data points.
        std            : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    if debug:
        assert is2dptsarray(pts_array), 'input points are not correct: (2, num_pts) vs %s' % print_np_shape(pts_array)
        if conf is not None: assert ifconfscalar(conf), 'the confidence is not in a good range'
        if std is not None: assert ispositiveinteger(std), 'the number of standard deviation should be a positive integer'

    pts_array = np.transpose(pts_array)
    center = pts_array.mean(axis=0)
    covariance = np.cov(pts_array, rowvar=False)
    return visualize_covariance_ellipse(covariance=covariance, center=center, conf=conf, std=std, fig=fig, ax=ax, debug=debug, **kwargs), np.sqrt(covariance[0, 0]**2 + covariance[1, 1]**2)

def visualize_covariance_ellipse(covariance, center, conf=None, std=None, fig=None, ax=None, debug=True, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
        covariance      : The 2x2 covariance matrix to base the ellipse on
        center          : The location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
        conf            : a floating number between [0, 1]
        std             : The radius of the ellipse in numbers of standard deviations. Defaults to 2 standard deviations.
        ax              : The axis that the ellipse will be plotted on. Defaults to the current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
        A covariance ellipse
    """
    if debug:
        if conf is not None: assert isscalar(conf) and conf >= 0 and conf <= 1, 'the confidence is not in a good range'
        if std is not None: assert ispositiveinteger(std), 'the number of standard deviation should be a positive integer'
    fig, ax = get_fig_ax_helper(fig=fig, ax=ax)

    def eigsorted(covariance):
        vals, vecs = np.linalg.eigh(covariance)
        # order = vals.argsort()[::-1]
        # return vals[order], vecs[:,order]
        return vals, vecs

    if conf is not None: conf = np.asarray(conf)
    elif std is not None: conf = 2 * norm.cdf(std) - 1
    else: raise ValueError('One of `conf` and `std` should be specified.')
    r2 = chi2.ppf(conf, 2)
    vals, vecs = eigsorted(covariance)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
    # Width and height are "full" widths, not radius
    # width, height = 2 * std * np.sqrt(vals)
    width, height = 2 * np.sqrt(np.sqrt(vals) * r2)
    # width, height = 2 * np.sqrt(vals[:, None] * r2)
    ellipse = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)
    ellipse.set_facecolor('none')

    ax.add_artist(ellipse)
    return ellipse

def visualize_pts(pts, title=None, fig=None, ax=None, display_range=False, xlim=[-100, 100], ylim=[-100, 100], display_list=None, covariance=False, mse=False, mse_value=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize point scatter plot

    parameter:
        pts:            2 x num_pts numpy array or a dictionary containing 2 x num_pts numpy array
    '''

    if debug:
        if isdict(pts):
            for pts_tmp in pts.values(): assert is2dptsarray(pts_tmp) , 'input points within dictionary are not correct: (2, num_pts) vs %s' % print_np_shape(pts_tmp)
            if display_list is not None:
                assert islist(display_list) and len(display_list) == len(pts), 'the input display list is not correct'
                assert CHECK_EQ_LIST_UNORDERED(display_list, pts.keys(), debug=debug), 'the input display list does not match the points key list'
            else: display_list = pts.keys()
        else: assert is2dptsarray(pts), 'input points are not correct: (2, num_pts) vs %s' % print_np_shape(pts)
        if title is not None: assert isstring(title), 'title is not correct'
        else: title = 'Point Error Vector Distribution Map'
        assert islogical(display_range), 'the flag determine if to display in a specific range should be logical value'
        if display_range:
            assert islist(xlim) and islist(ylim) and len(xlim) == 2 and len(ylim) == 2, 'the input range for x and y is not correct'
            assert xlim[1] > xlim[0] and ylim[1] > ylim[0], 'the input range for x and y is not correct'

    # figure setting
    width, height = 1024, 1024
    fig, _ = get_fig_ax_helper(fig=fig, ax=ax, width=width, height=height)
    if ax is None:
        plt.title(title, fontsize=20)
        if isdict(pts):
            num_pts_all = pts.values()[0].shape[1]
            if all(pts_tmp.shape[1] == num_pts_all for pts_tmp in pts.values()):
                plt.xlabel('x coordinate (%d points)' % pts.values()[0].shape[1], fontsize=16)
                plt.ylabel('y coordinate (%d points)' % pts.values()[0].shape[1], fontsize=16)
            else:
                print('number of points is different across different methods')
                plt.xlabel('x coordinate', fontsize=16)
                plt.ylabel('y coordinate', fontsize=16)
        else:
            plt.xlabel('x coordinate (%d points)' % pts.shape[1], fontsize=16)
            plt.ylabel('y coordinate (%d points)' % pts.shape[1], fontsize=16)
        plt.axis('equal')
        ax = plt.gca()
        ax.grid()
    
    # internal parameters
    pts_size = 5
    std = None
    conf = 0.98
    color_index = 0
    marker_index = 0
    hatch_index = 0
    alpha = 0.2
    legend_fontsize = 10
    scale_distance = 48.8
    linewidth = 2

    # plot points
    handle_dict = dict()    # for legend
    if isdict(pts):
        num_methods = len(pts)
        assert len(color_set) * len(marker_set) >= num_methods and len(color_set) * len(hatch_set) >= num_methods, 'color in color set is not enough to use, please use different markers'
        mse_return = dict()
        for method_name, pts_tmp in pts.items():
            color_tmp = color_set[color_index]
            marker_tmp = marker_set[marker_index]
            hatch_tmp = hatch_set[hatch_index]

            # plot covariance ellipse
            if covariance: _, covariance_number = visualize_pts_covariance(pts_tmp[0:2, :], std=std, conf=conf, ax=ax, debug=debug, color=color_tmp, hatch=hatch_tmp, linewidth=linewidth)
            handle_tmp = ax.scatter(pts_tmp[0, :], pts_tmp[1, :], color=color_tmp, marker=marker_tmp, s=pts_size, alpha=alpha)    
            if mse:
                if mse_value is None:
                    num_pts = pts_tmp.shape[1]
                    mse_tmp, _ = pts_euclidean(pts_tmp[0:2, :], np.zeros((2, num_pts), dtype='float32'), debug=debug)
                else:
                    mse_tmp = mse_value[method_name]
                display_string = '%s, MSE: %.1f (%.1f um), Covariance: %.1f' % (method_name, mse_tmp, mse_tmp * scale_distance, covariance_number)
                mse_return[method_name] = mse_tmp
            else: display_string = method_name
            handle_dict[display_string] = handle_tmp
            color_index += 1
            if color_index / len(color_set) == 1:            
                marker_index += 1
                hatch_index += 1
                color_index = color_index % len(color_set)

        # reorder the handle before plot
        handle_key_list = handle_dict.keys()
        handle_value_list = handle_dict.values()
        order_index_list = [display_list.index(method_name_tmp.split(', ')[0]) for method_name_tmp in handle_dict.keys()]
        ordered_handle_key_list = list_reorder(handle_key_list, order_index_list, debug=debug)
        ordered_handle_value_list = list_reorder(handle_value_list, order_index_list, debug=debug)
        plt.legend(list2tuple(ordered_handle_value_list), list2tuple(ordered_handle_key_list), scatterpoints=1, markerscale=4, loc='lower left', fontsize=legend_fontsize)
        
    else:
        color_tmp = color_set[color_index]
        marker_tmp = marker_set[marker_index]
        hatch_tmp = hatch_set[hatch_index]
        handle_tmp = ax.scatter(pts[0, :], pts[1, :], color=color_tmp, marker=marker_tmp, s=pts_size, alpha=alpha)    

        # plot covariance ellipse
        if covariance: _, covariance_number = visualize_pts_covariance(pts[0:2, :], std=std, conf=conf, ax=ax, debug=debug, color=color_tmp, hatch=hatch_tmp, linewidth=linewidth)

        if mse:
            if mse_value is None:
                num_pts = pts.shape[1]
                mse_tmp, _ = pts_euclidean(pts[0:2, :], np.zeros((2, num_pts), dtype='float32'), debug=debug)
                display_string = 'MSE: %.1f (%.1f um), Covariance: %.1f' % (mse_tmp, mse_tmp * scale_distance, covariance_number)
                mse_return = mse_tmp
            else:
                display_string = 'MSE: %.1f (%.1f um), Covariance: %.1f' % (mse_value, mse_value * scale_distance, covariance_number)
                mse_return = mse_value
            handle_dict[display_string] = handle_tmp
            plt.legend(list2tuple(handle_dict.values()), list2tuple(handle_dict.keys()), scatterpoints=1, markerscale=4, loc='lower left', fontsize=legend_fontsize)
            
    # display only specific range
    if display_range:
        axis_bin = 10 * 2
        interval_x = (xlim[1] - xlim[0]) / axis_bin
        interval_y = (ylim[1] - ylim[0]) / axis_bin
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xticks(np.arange(xlim[0], xlim[1] + interval_x, interval_x))
        plt.yticks(np.arange(ylim[0], ylim[1] + interval_y, interval_y))
    plt.grid()

    save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, warning=warning, debug=debug, closefig=closefig, transparent=False)
    return mse_return

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image

def visualize_image_with_bbox_mask(image, boxes, masks, class_ids, class_names, class_to_plot=None, scores=None, alpha=0.7, fig=None, ax=None, color_list=None, title='Mask & Bounding Box Visualization'):
    """
    visualize the image with bbox and mask (and text and score)

    parameters:
        boxes: [num_instance, (x1, y1, x2, y2, class_id)] in image coordinates.
        masks: [height, width, num_instances], numpy images, range in [0, 1]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        class_to_plot:     list of class index in the class_names to plot
        title:
    """
    max_numinstances = 20
    if class_to_plot is None: class_to_plot = range(len(class_names))
    num_instances = boxes.shape[0]          # Number of instances
    if not num_instances: print("\n*** No instances to display *** \n")
    else: assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = random_colors(max_numinstances)       # Generate random colors
    if color_list is None: color_list = range(num_instances)

    height, width = image.shape[:2]
    # print(height)
    # print(width)
    # zxc

    fig, _ = get_fig_ax_helper(fig=fig, ax=ax, width=width, height=height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # if not ax: fig = plt.figure(figsize=(16, 16))
    # ax = fig.add_axes([0, 0, 1, 0.5])
    
    # ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint8).copy()
    # print(masked_image.shape)
    # save_image(masked_image, '/home/xinshuo/test.jpg')
    # tmp_dir = '/home/xinshuo/Workspace/junk/vis_individual'

    for instance_index in range(num_instances):
        color = colors[color_list[instance_index] % max_numinstances]
        # print(color)
        # zxc

        # skip to visualize the class we do not care
        class_id = class_ids[instance_index]
        if not (class_id in class_to_plot): continue

        # zxc

        # visualize the bbox
        if not np.any(boxes[instance_index]): continue           # Skip this instance. Has no bbox. Likely lost in image cropping.
        x1, y1, x2, y2 = boxes[instance_index]
        # print(x1)
        # print(y1)
        # print(x2)
        # print(y2)
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=alpha, edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # add the text and score
        score = scores[instance_index] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.2f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption, color='w', size=8, backgroundcolor='none')

        # add the mask
        mask = masks[:, :, instance_index]
        # print(np.max(mask))
        # print(np.min(mask))
        # save_image(mask, '/home/xinshuo/test.jpg')
        # zxc
        masked_image = apply_mask(masked_image, mask, color)
        # save_image(masked_image, '/home/xinshuo/test%d.jpg' % instance_index)
        # zxc

        # save the individual mask one by one
        # save_tmp_dir = os.path.join(tmp_dir, 'instance_%04d.jpg' % instance_index); mkdir_if_missing(save_tmp_dir)
        # save_image(masked_image.astype('uint8'), save_path=save_tmp_dir)

        # add the contour of the mask
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1            # Subtract the padding and flip (y, x) to (x, y)
            p = patches.Polygon(verts, facecolor="none", edgecolor=color, alpha=alpha)
            ax.add_patch(p)


    # zxc
    # print(masked_image.shape)
    # zxc
    ax.imshow(masked_image.astype(np.uint8))
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    return fig, ax
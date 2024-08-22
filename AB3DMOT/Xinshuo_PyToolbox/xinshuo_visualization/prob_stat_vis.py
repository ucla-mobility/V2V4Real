# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import matplotlib.pyplot as plt, numpy as np
# import seaborn as sns
# from pandas import DataFrame
# from sklearn.neighbors import NearestNeighbors
from terminaltables import AsciiTable
from collections import Counter

from .private import save_vis_close_helper, get_fig_ax_helper
from xinshuo_miscellaneous import isdict, islogical, is_path_exists, isscalar, islist, is_path_exists_or_creatable, CHECK_EQ_LIST_UNORDERED, isnparray, isinteger, isstring, scalarlist2strlist, islistoflist, iscolorimage_dimension, isgrayimage_dimension, istuple
from xinshuo_math import calculate_truncated_mse

color_set = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'lime', 'cyan', 'aqua']
linestyle_set = ['-', '--', '-.', ':', None, ' ', 'solid', 'dashed']
dpi = 80

def visualize_ced(normed_mean_error_dict, error_threshold, normalized=True, truncated_list=None, display2terminal=True, display_list=None, title='2D PCK curve', debug=True, vis=False, pck_savepath=None, table_savepath=None, closefig=True):
    '''
    visualize the cumulative error distribution curve (alse called NME curve or pck curve)
    all parameters are represented by percentage

    parameter:
        normed_mean_error_dict:     a dictionary whose keys are the method name and values are (N, ) numpy array to represent error in evaluation
        error_threshold:            threshold to display in x axis

    return:
        AUC:                        area under the curve
        MSE:                        mean square error
    '''
    if debug:
        assert isdict(normed_mean_error_dict), 'the input normalized mean error dictionary is not correct'
        assert islogical(normalized), 'the normalization flag should be logical'
        if normalized: assert error_threshold > 0 and error_threshold < 100, 'threshold percentage is not well set'
        if save:
            assert is_path_exists_or_creatable(pck_savepath), 'please provide a valid path to save the pck results' 
            assert is_path_exists_or_creatable(table_savepath), 'please provide a valid path to save the table results' 
        assert isstring(title), 'title is not correct'
        if truncated_list is not None: assert islistofscalar(truncated_list), 'the input truncated list is not correct'
        if display_list is not None:
            assert islist(display_list) and len(display_list) == len(normed_mean_error_dict), 'the input display list is not correct'
            assert CHECK_EQ_LIST_UNORDERED(display_list, normed_mean_error_dict.keys(), debug=debug), 'the input display list does not match the error dictionary key list'
        else: display_list = normed_mean_error_dict.keys()

    # set display parameters
    width, height = 1000, 800
    legend_fontsize = 10
    scale_distance = 48.8
    line_index, color_index = 0, 0

    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)

    # set figure handle
    num_bins = 1000
    if normalized:
        maximum_x = 1
        scale = num_bins / 100
    else:
        maximum_x = error_threshold + 1
        scale = num_bins / maximum_x        
    x_axis = np.linspace(0, maximum_x, num_bins)            # error axis, percentage of normalization factor    
    y_axis = np.zeros(num_bins)
    interval_y = 10
    interval_x = 1
    plt.xlim(0, error_threshold)
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.xticks(np.arange(0, error_threshold + interval_x, interval_x))
    plt.grid()
    plt.title(title, fontsize=20)
    if normalized: plt.xlabel('Normalized error euclidean distance (%)', fontsize=16)
    else: plt.xlabel('Absolute error euclidean distance', fontsize=16)

    # calculate metrics for each method
    num_methods = len(normed_mean_error_dict)
    num_images = len(normed_mean_error_dict.values()[0])
    metrics_dict = dict()
    metrics_table = list()
    table_title = ['Method Name / Metrics', 'AUC', 'MSE']
    append2title = False
    assert num_images > 0, 'number of error array should be larger than 0'
    for ordered_index in range(num_methods):
        method_name = display_list[ordered_index]
        normed_mean_error = normed_mean_error_dict[method_name]

        if debug:
            assert isnparray(normed_mean_error) and normed_mean_error.ndim == 1, 'shape of error distance is not good'
            assert len(normed_mean_error) == num_images, 'number of testing images should be equal for all methods'
            assert len(linestyle_set) * len(color_set) >= len(normed_mean_error_dict)

        color_tmp = color_set[color_index]
        line_tmp = linestyle_set[line_index]
        
        for i in range(num_bins):
            y_axis[i] = float((normed_mean_error < x_axis[i]).sum()) / num_images                       # percentage of error

        # calculate area under the curve and mean square error
        entry = dict()
        entry['AUC'] = np.sum(y_axis[:error_threshold * scale]) / (error_threshold * scale)         # bigger, better
        entry['MSE'] = np.mean(normed_mean_error)                                                                      # smaller, better
        metrics_table_tmp = [str(method_name), '%.2f' % (entry['AUC']), '%.1f' % (entry['MSE'])]
        if truncated_list is not None:
            tmse_dict = calculate_truncated_mse(normed_mean_error.tolist(), truncated_list, debug=debug)
            for threshold in truncated_list:
                entry['AUC/%s'%threshold] = np.sum(y_axis[:error_threshold * scale]) / (error_threshold * scale)         # bigger, better
                entry['MSE/%s'%threshold] = tmse_dict[threshold]['T-MSE']
                entry['percentage/%s'%threshold] = tmse_dict[threshold]['percentage']
                
                if not append2title:
                    table_title.append('AUC/%s'%threshold)
                    table_title.append('MSE/%s'%threshold)
                    table_title.append('pct/%s'%threshold)
                metrics_table_tmp.append('%.2f' % (entry['AUC/%s'%threshold]))
                metrics_table_tmp.append('%.1f' % (entry['MSE/%s'%threshold]))
                metrics_table_tmp.append('%.1f' % (100 * entry['percentage/%s'%threshold]) + '%')

        # print metrics_table_tmp
        metrics_table.append(metrics_table_tmp)
        append2title = True
        metrics_dict[method_name] = entry

        # draw 
        label = '%s, AUC: %.2f, MSE: %.1f (%.0f um)' % (method_name, entry['AUC'], entry['MSE'], entry['MSE'] * scale_distance)
        if normalized: plt.plot(x_axis*100, y_axis*100, color=color_tmp, linestyle=line_tmp, label=label, lw=3)
        else: plt.plot(x_axis, y_axis*100, color=color_tmp, linestyle=line_tmp, label=label, lw=3)
        plt.legend(loc=4, fontsize=legend_fontsize)
        
        color_index += 1
        if color_index / len(color_set) == 1:            
            line_index += 1
            color_index = color_index % len(color_set)

    # plt.grid()
    plt.ylabel('{} Test Images (%)'.format(num_images), fontsize=16)
    save_vis_close_helper(fig=fig, ax=None, vis=vis, transparent=False, save_path=pck_savepath, debug=debug, closefig=closefig)

    # reorder the table
    order_index_list = [display_list.index(method_name_tmp) for method_name_tmp in normed_mean_error_dict.keys()]
    order_index_list = [0] + [order_index_tmp + 1 for order_index_tmp in order_index_list]

    # print table to terminal
    metrics_table = [table_title] + metrics_table
    # metrics_table = list_reorder([table_title] + metrics_table, order_index_list, debug=debug)
    table = AsciiTable(metrics_table)
    if display2terminal:
        print('\nprint detailed metrics')
        print(table.table)
        
    # save table to file
    if table_savepath is not None:
        table_file = open(table_savepath, 'w')
        table_file.write(table.table)
        table_file.close()
        if display2terminal: print('\nsave detailed metrics to %s' % table_savepath)
    
    return metrics_dict, metrics_table


def visualize_nearest_neighbor(featuremap_dict, num_neighbor=5, top_number=5, vis=True, save_csv=False, csv_save_path=None, save_vis=False, save_img=False, save_thumb_name='nearest_neighbor.png', img_src_folder=None, ext_filter='.jpg', nn_save_folder=None, debug=True):
    '''
    visualize nearest neighbor for featuremap from images

    parameter:
        featuremap_dict: a dictionary contains image path as key, and featuremap as value, the featuremap needs to be numpy array with any shape. No flatten needed
        num_neighbor: number of neighbor to visualize, the first nearest is itself
        top_number: number of top to visualize, since there might be tons of featuremap (length of dictionary), we choose the top ten with lowest distance with their nearest neighbor
        csv_save_path: path to save .csv file which contains indices and distance array for all elements
        nn_save_folder: save the nearest neighbor images for top featuremap

    return:
        all_sorted_nearest_id: a 2d matrix, each row is a feature followed by its nearest neighbor in whole feature dataset, the column is sorted by the distance of all nearest neighbor each row
        selected_nearest_id: only top number of sorted nearest id 
    '''
    print('processing feature map to nearest neightbor.......')
    if debug:
        assert isdict(featuremap_dict), 'featuremap should be dictionary'
        assert all(isnparray(featuremap_tmp) for featuremap_tmp in featuremap_dict.values()), 'value of dictionary should be numpy array'
        assert isinteger(num_neighbor) and num_neighbor > 1, 'number of neighborhodd is an integer larger than 1'
        if save_csv and csv_save_path is not None:
            assert is_path_exists_or_creatable(csv_save_path), 'path to save .csv file is not correct'
        
        if save_vis or save_img:
            if nn_save_folder is not None:  # save image directly
                assert isstring(ext_filter), 'extension filter is not correct'
                assert is_path_exists(img_src_folder), 'source folder for image is not correct'
                assert all(isstring(path_tmp) for path_tmp in featuremap_dict.keys())     # key should be the path for the image
                assert is_path_exists_or_creatable(nn_save_folder), 'folder to save top visualized images is not correct'
                assert isstring(save_thumb_name), 'name of thumbnail is not correct'

    if ext_filter.find('.') == -1:
        ext_filter = '.%s' % ext_filter

    # flatten the feature map
    nn_feature_dict = dict()
    for key, featuremap_tmp in featuremap_dict.items():
        nn_feature_dict[key] = featuremap_tmp.flatten()
    num_features = len(nn_feature_dict)

    # nearest neighbor
    featuremap = np.array(nn_feature_dict.values())
    nearbrs = NearestNeighbors(n_neighbors=num_neighbor, algorithm='ball_tree').fit(featuremap)
    distances, indices = nearbrs.kneighbors(featuremap)
    
    if debug:
        assert featuremap.shape[0] == num_features, 'shape of feature map is not correct'
        assert indices.shape == (num_features, num_neighbor), 'shape of indices is not correct'
        assert distances.shape == (num_features, num_neighbor), 'shape of indices is not correct'

    # convert the nearest indices for all featuremap to the key accordingly
    id_list = nn_feature_dict.keys()
    max_length = len(max(id_list, key=len))     # find the maximum length of string in the key
    nearest_id = np.chararray(indices.shape, itemsize=max_length+1)
    for x in range(nearest_id.shape[0]):
        for y in range(nearest_id.shape[1]):
            nearest_id[x, y] = id_list[indices[x, y]]

    if debug:
        assert list(nearest_id[:, 0]) == id_list, 'nearest neighbor has problem'
    
    # sort the feature based on distance
    print('sorting the feature based on distance')
    featuremap_distance = np.sum(distances, axis=1)
    if debug:
        assert featuremap_distance.shape == (num_features, ), 'distance is not correct'
    sorted_indices = np.argsort(featuremap_distance)
    all_sorted_nearest_id = nearest_id[sorted_indices, :]

    # save to the csv file
    if save_csv and csv_save_path is not None:
        print('Saving nearest neighbor result as .csv to path: %s' % csv_save_path)
        with open(csv_save_path, 'w+') as file:
            np.savetxt(file, distances, delimiter=',', fmt='%f')
            np.savetxt(file, all_sorted_nearest_id, delimiter=',', fmt='%s')
            file.close()

    # choose the best to visualize
    selected_sorted_indices = sorted_indices[0:top_number]
    if debug:
        for i in range(num_features-1):
            assert featuremap_distance[sorted_indices[i]] < featuremap_distance[sorted_indices[i+1]], 'feature map is not well sorted based on distance' 
    selected_nearest_id = nearest_id[selected_sorted_indices, :]

    if save_vis:
        fig, axarray = plt.subplots(top_number, num_neighbor)
        for index in range(top_number):
            for nearest_index in range(num_neighbor):
                img_path = os.path.join(img_src_folder, '%s%s'%(selected_nearest_id[index, nearest_index], ext_filter))
                if debug:
                    print('loading image from %s'%img_path)
                img = imread(img_path)
                if isgrayimage_dimension(img):
                    axarray[index, nearest_index].imshow(img, cmap='gray')
                elif iscolorimage_dimension(img):
                    axarray[index, nearest_index].imshow(img)
                else:
                    assert False, 'unknown error'
                axarray[index, nearest_index].axis('off')
        save_thumb = os.path.join(nn_save_folder, save_thumb_name)
        fig.savefig(save_thumb)
        if vis:
            plt.show()
        plt.close(fig)

    # save top visualization to the folder
    if save_img and nn_save_folder is not None:
        for top_index in range(top_number):
            file_list = selected_nearest_id[top_index]
            save_subfolder = os.path.join(nn_save_folder, file_list[0])
            mkdir_if_missing(save_subfolder)
            for file_tmp in file_list:
                file_src = os.path.join(img_src_folder, '%s%s'%(file_tmp, ext_filter))
                save_path = os.path.join(save_subfolder, '%s%s'%(file_tmp, ext_filter))
                if debug:
                    print('saving %s to %s' % (file_src, save_path))
                shutil.copyfile(file_src, save_path)

    return all_sorted_nearest_id, selected_nearest_id

def visualize_distribution(data, bin_size=None, xlim=None, ylim=None, vis=False, save_path=None, debug=True, closefig=True):
    '''
    visualize the histogram of a data, which can be a dictionary or list or numpy array or tuple or a list of list
    '''
    if debug:
        assert istuple(data) or isdict(data) or islist(data) or isnparray(data), 'input data is not correct'

    # convert data type
    if istuple(data):
        data = list(data)
    elif isdict(data):
        data = data.values()
    elif isnparray(data):
        data = data.tolist()

    num_bins = 1000.0
    fig, ax = get_fig_ax_helper(fig=None, ax=None)

    # calculate bin size
    if bin_size is None:
        if islistoflist(data):
            max_value = np.max(np.max(data))
            min_value = np.min(np.min(data))
        else:
            max_value = np.max(data)
            min_value = np.min(data)
        bin_size = (max_value - min_value) / num_bins
    else:
        try:
            bin_size = float(bin_size)
        except TypeError:
            print('size of bin should be an float value')

    # plot
    if islistoflist(data):
        max_value = np.max(np.max(data))
        min_value = np.min(np.min(data))
        bins = np.arange(min_value - bin_size, max_value + bin_size, bin_size)      # fixed bin size
        plt.xlim([min_value - bin_size, max_value + bin_size])
        for data_list_tmp in data:
            if debug:
                assert islist(data_list_tmp), 'the nested list is not correct!'
            # plt.hist(data_list_tmp, bins=bins, alpha=0.3)
            sns.distplot(data_list_tmp, bins=bins, kde=False)
            # sns.distplot(data_list_tmp, bins=bins, kde=False)
    else:
        bins = np.arange(min(data) - 10 * bin_size, max(data) + 10 * bin_size, bin_size)      # fixed bin size
        plt.xlim([min(data) - bin_size, max(data) + bin_size])
        plt.hist(data, bins=bins, alpha=0.5)
    
    plt.title('distribution of data')
    plt.xlabel('data (bin size = %f)' % bin_size)
    plt.ylabel('count')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, closefig=closefig)

def visualize_bar(data, bin_size=2.0, title='Bar Graph of Key-Value Pair', xlabel='index', ylabel='count', vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize the bar graph of a data, which can be a dictionary or list of dictionary

    different from function of visualize_bar_graph, this function does not depend on panda and dataframe, it's simpler but with less functionality
    also the key of this function takes continuous scalar variable
    '''
    if debug:
        assert isstring(title) and isstring(xlabel) and isstring(ylabel), 'title/xlabel/ylabel is not correct'
        assert isdict(data) or islist(data), 'input data is not correct'
        assert isscalar(bin_size), 'the bin size is not a floating number'

    if isdict(data):
        index_list = data.keys()
        if debug:
            assert islistofscalar(index_list), 'the input dictionary does not contain a scalar key'
        frequencies = data.values()
    else:
        index_list = range(len(data))
        frequencies = data

    index_str_list = scalarlist2strlist(index_list, debug=debug)
    index_list = np.array(index_list)
    fig, ax = get_fig_ax_helper(fig=None, ax=None)
    # ax.set_xticks(index_list)
    # ax.set_xticklabels(index_str_list)
    plt.bar(index_list, frequencies, bin_size, color='r', alpha=0.5)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_vis_close_helper(fig=fig, ax=ax, vis=vis, save_path=save_path, debug=debug, transparent=False, closefig=closefig)

def visualize_bar_graph(data, title='Bar Graph of Key-Value Pair', xlabel='pixel error', ylabel='keypoint index', label=False, label_list=None, vis=True, save_path=None, debug=True, closefig=True):
    '''
    visualize the bar graph of a data, which can be a dictionary or list of dictionary
    inside each dictionary, the keys (string) should be the same which is the y label, the values should be scalar
    '''
    if debug:
        assert isstring(title) and isstring(xlabel) and isstring(ylabel), 'title/xlabel/ylabel is not correct'
        assert isdict(data) or islistofdict(data), 'input data is not correct'
        if isdict(data):
            assert all(isstring(key_tmp) for key_tmp in data.keys()), 'the keys are not all strings'
            assert all(isscalar(value_tmp) for value_tmp in data.values()), 'the keys are not all strings'
        else:
            assert len(data) <= len(color_set), 'number of data set is larger than number of color to use'
            keys = sorted(data[0].keys())
            for dict_tmp in data:
                if not (sorted(dict_tmp.keys()) == keys):
                    print(dict_tmp.keys())
                    print(keys)
                    assert False, 'the keys are not equal across different input set'
                assert all(isstring(key_tmp) for key_tmp in dict_tmp.keys()), 'the keys are not all strings'
                assert all(isscalar(value_tmp) for value_tmp in dict_tmp.values()), 'the values are not all scalars'   

    # convert dictionary to DataFrame
    data_new = dict()
    if isdict(data):
        key_list = data.keys()
        sorted_index = sorted(range(len(key_list)), key=lambda k: key_list[k])
        data_new['names'] = (np.asarray(key_list)[sorted_index]).tolist()
        data_new['values'] = (np.asarray(data.values())[sorted_index]).tolist()
    else:
        key_list = data[0].keys()
        sorted_index = sorted(range(len(key_list)), key=lambda k: key_list[k])
        data_new['names'] = (np.asarray(key_list)[sorted_index]).tolist()
        num_sets = len(data)        
        for set_index in range(num_sets):
            data_new['value_%03d'%set_index] = (np.asarray(data[set_index].values())[sorted_index]).tolist()
    dataframe = DataFrame(data_new)

    # plot
    width = 2000
    height = 2000
    alpha = 0.5
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    sns.set(style='whitegrid')
    # fig, ax = get_fig_ax_helper(fig=None, ax=None)
    if isdict(data):
        g = sns.barplot(x='values', y='names', data=dataframe, label='data', color='b')
        plt.legend(ncol=1, loc='lower right', frameon=True, fontsize=5)
    else:
        num_sets = len(data)
        for set_index in range(num_sets):
            if set_index == 0:
                sns.set_color_codes('pastel')
            else:
                sns.set_color_codes('muted')

            if label:
                sns.barplot(x='value_%03d'%set_index, y='names', data=dataframe, label=label_list[set_index], color=color_set[set_index], alpha=alpha)
            else:
                sns.barplot(x='value_%03d'%set_index, y='names', data=dataframe, color=solor_set[set_index], alpha=alpha)
        plt.legend(ncol=len(data), loc='lower right', frameon=True, fontsize=5)

    sns.despine(left=True, bottom=True)
    plt.title(title, fontsize=20)
    plt.xlim([0, 50])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    num_yticks = len(data_new['names'])
    adaptive_fontsize = -0.0555556 * num_yticks + 15.111
    plt.yticks(fontsize=adaptive_fontsize)

    return save_vis_close_helper(fig=fig, vis=vis, save_path=save_path, debug=debug, closefig=closefig)
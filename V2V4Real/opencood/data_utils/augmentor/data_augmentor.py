# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

from opencood.data_utils.augmentor import augment_utils


class DataAugmentor(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True, intermediate=False):
        self.data_augmentor_queue = []
        self.train = train
        self.augment_config = augment_config

        for cur_cfg in augment_config:
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points, flip = data_dict['object_bbx_center'], \
                                          data_dict['object_bbx_mask'], \
                                          data_dict['lidar_np'], \
                                          data_dict['flip']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for i, cur_axis in enumerate(config['ALONG_AXIS_LIST']):
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points, flip[i] if flip is not None else flip
            )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points, noise_rotation = data_dict[
                                                        'object_bbx_center'], \
                                                    data_dict[
                                                        'object_bbx_mask'], \
                                                    data_dict['lidar_np'], \
                                                    data_dict['noise_rotation']
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        gt_boxes_valid, points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range, noise_rotation
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points, noise_scale = data_dict[
                                                        'object_bbx_center'], \
                                                    data_dict[
                                                        'object_bbx_mask'], \
                                                    data_dict['lidar_np'], \
                                                    data_dict['noise_scale']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(
            gt_boxes_valid, points, config['WORLD_SCALE_RANGE'],
            noise_scale
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if self.train:
            for cur_augmentor in self.data_augmentor_queue:
                data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict

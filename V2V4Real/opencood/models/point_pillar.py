import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'],
                                  7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        #print('voxel_features.shape: ', voxel_features.shape) # [5427, 32, 4]
        #print('voxel_coords.shape: ', voxel_coords.shape) # [5427, 4]
        #print('voxel_num_points.shape: ', voxel_num_points.shape) # [5427]

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        #print("batch_dict['pillar_features'].shape: ", batch_dict['pillar_features'].shape)
        # [5427, 64]

        batch_dict = self.scatter(batch_dict)
        #print("batch_dict['spatial_features'].shape: ", batch_dict['spatial_features'].shape)
        # [1, 64, 200, 352]
        spatial_features = batch_dict['spatial_features']
        

        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm,
                       'spatial_features_2d': spatial_features_2d,
                       'spatial_features': spatial_features }


        #print('spatial_features_2d.shape: ', spatial_features_2d.shape)
        # [1, 256, 50, 88]
        #print('rm.shape: ', rm.shape)
        # [1, 14, 50, 88]

        return output_dict

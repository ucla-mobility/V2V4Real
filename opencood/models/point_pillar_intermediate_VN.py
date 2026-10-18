import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class PointPillarIntermediateVN(nn.Module):
    """PointPillar AttFuse with a heteroscedastic regression variance head."""

    def __init__(self, args):
        super(PointPillarIntermediateVN, self).__init__()

        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        # AttBEVBackbone outputs 128 * 3 channels after multi-scale upsampling.
        head_channels = 128 * 3
        self.cls_head = nn.Conv2d(
            head_channels,
            args['anchor_number'],
            kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            head_channels,
            7 * args['anchor_number'],
            kernel_size=1
        )

        # Predict per-box-parameter log variance: s = log(sigma^2).
        self.reg_var_head = nn.Conv2d(
            head_channels,
            7 * args['anchor_number'],
            kernel_size=1
        )

        # Same initialization used by the probabilistic F-Cooper model.
        nn.init.constant_(self.reg_var_head.weight, 0.0)
        nn.init.constant_(self.reg_var_head.bias, -2.0)

        # Optional: train the variance head on detached features, so the
        # uncertainty loss cannot change the shared backbone (detection
        # training then equals the baseline). Default off = old behaviour.
        self.detach_var_input = args.get('detach_var_input', False)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        var_input = spatial_features_2d.detach() if self.detach_var_input else spatial_features_2d
        rm_logvar = self.reg_var_head(var_input)

        output_dict = {
            'psm': psm,
            'rm': rm,
            'rm_logvar': rm_logvar
        }

        return output_dict
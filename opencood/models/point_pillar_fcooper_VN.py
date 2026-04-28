import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.f_cooper_fuse import SpatialFusion


class PointPillarFCooperVN(nn.Module):
    """
    PointPillar F-Cooper with an additional variance network.

    Original outputs:
        psm: classification score map
        rm:  bounding box regression mean

    New output:
        rvar: bounding box log-variance

    Interpretation:
        rm   = predicted mean bbox regression values
        rvar = predicted log variance for each bbox regression value

    For each anchor, rm predicts 7 values:
        x, y, z, h, w, l, yaw

    Therefore rvar also predicts 7 values per anchor.
    """

    def __init__(self, args):
        super(PointPillarFCooperVN, self).__init__()

        self.max_cav = args['max_cav']
        self.anchor_number = args['anchor_number']

        # Pillar feature encoder
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])

        # Convert sparse pillar features to BEV pseudo-image
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        # BEV backbone
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Optional downsample module
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # Optional feature compression
        self.compression = False
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        # F-Cooper spatial feature fusion
        self.fusion_net = SpatialFusion()

        # Detection heads.
        # After fusion, fused_feature has 128 * 2 channels in your original model.
        in_channels = 128 * 2

        # Classification head: same as original
        self.cls_head = nn.Conv2d(in_channels,
                                  self.anchor_number,
                                  kernel_size=1)

        # Regression mean head: same as original
        self.reg_head = nn.Conv2d(in_channels,
                                  7 * self.anchor_number,
                                  kernel_size=1)

        # NEW: regression variance head.
        # It predicts log-variance, not variance directly.
        self.reg_var_head = nn.Conv2d(in_channels,
                                      7 * self.anchor_number,
                                      kernel_size=1)

        # Initialize variance head safely.
        # Bias = 0 means exp(log_var)=1 at the beginning.
        nn.init.normal_(self.reg_var_head.weight, mean=0.0, std=0.0001)
        nn.init.constant_(self.reg_var_head.bias, 0.0)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Freeze backbone and original detection heads during fine-tuning.

        Note:
            We do NOT freeze reg_var_head. This allows training only the new
            variance head on top of an already trained detector if desired.
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False

        for p in self.reg_head.parameters():
            p.requires_grad = False

        # Keep variance head trainable.
        for p in self.reg_var_head.parameters():
            p.requires_grad = True

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

        # Pillar feature extraction
        batch_dict = self.pillar_vfe(batch_dict)

        # Scatter pillars to BEV feature map
        batch_dict = self.scatter(batch_dict)

        # BEV backbone
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        # Optional downsampling
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # Optional compression
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # Cooperative feature fusion
        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        # Original outputs
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        # NEW variance-network output
        rvar = self.reg_var_head(fused_feature)

        output_dict = {
            'psm': psm,
            'rm': rm,
            'rvar': rvar
        }

        return output_dict
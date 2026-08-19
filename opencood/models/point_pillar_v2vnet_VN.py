import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.v2v_fuse import V2VNetFusion


class PointPillarV2VNetVN(nn.Module):
    """
    Same architecture as PointPillarV2VNet, with one added parallel
    reg_var_head predicting a log-variance for each of the 7 regression
    codes (x, y, z, h, w, l, yaw), exactly mirroring
    point_pillar_fcooper_VN.py's head design (same zero-weight / -2-bias
    init). Used with either point_pillar_VN_loss_nll.py or one of the
    Energy Score loss files -- see point_pillar_v2vnet_VN_nll.yaml /
    point_pillar_v2vnet_VN_es.yaml.
    """
    def __init__(self, args):
        super(PointPillarV2VNetVN, self).__init__()

        self.max_cav = args['max_cav']
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
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2VNetFusion(args['v2vfusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.reg_var_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                      kernel_size=1)

        # Same init as point_pillar_fcooper_VN.py: start variance small and
        # stable (exp(-2) ~= 0.135) instead of arbitrary, so the
        # precision-weighted loss term doesn't dominate/destabilize
        # training before the head has learned anything.
        nn.init.constant_(self.reg_var_head.weight, 0.0)
        nn.init.constant_(self.reg_var_head.bias, -2.0)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
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
        for p in self.reg_var_head.parameters():
            p.requires_grad = False

    def unpad_prior_encoding(self, x, record_len):
        # remove padded zeros to form tensor with shape (N, 3)
        # x: (B, L, 3); record_len: (B)
        B = x.shape[0]
        out = []
        for i in range(B):
            # (valid_len, 3)
            out.append(x[i, :record_len[i], :])
        out = torch.cat(out, dim=0)
        # (N, 3)
        return out

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        prior_encoding = data_dict['prior_encoding']
        prior_encoding = self.unpad_prior_encoding(prior_encoding, record_len)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        fused_feature = self.fusion_net(spatial_features_2d,
                                        record_len,
                                        pairwise_t_matrix,
                                        prior_encoding)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        rm_logvar = self.reg_var_head(fused_feature)

        output_dict = {'psm': psm, 'rm': rm, 'rm_logvar': rm_logvar}

        return output_dict

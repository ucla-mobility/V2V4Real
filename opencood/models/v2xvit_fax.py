import torch
import torch.nn as nn
from einops import repeat

from opencood.models.point_pillar_transformer import PointPillarTransformer
from opencood.models.sub_modules.fuse_utils import regroup


class V2XViTFax(PointPillarTransformer):
    def __init__(self, args):
        super(V2XViTFax, self).__init__(args)

    def forward(self, data_dict):

        # ===== Normal inference =====
        if not self.training:
            return super().forward(data_dict)

        features = []

        # ===== DA: loop over source + target =====
        for data_per in data_dict:

            voxel_features = data_per['processed_lidar']['voxel_features']
            voxel_coords = data_per['processed_lidar']['voxel_coords']
            voxel_num_points = data_per['processed_lidar']['voxel_num_points']
            record_len = data_per['record_len']
            spatial_correction_matrix = data_per['spatial_correction_matrix']

            prior_encoding = data_per['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

            batch_dict = {
                'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points,
                'record_len': record_len
            }

            # ===== Backbone =====
            batch_dict = self.pillar_vfe(batch_dict)
            print("did you go here")
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']

            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)

            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)

            # ===== Regroup =====
            regroup_feature, mask = regroup(
                spatial_features_2d,
                record_len,
                self.max_cav
            )

            # ===== Add prior encoding =====
            prior_encoding = prior_encoding.repeat(
                1, 1, 1,
                regroup_feature.shape[3],
                regroup_feature.shape[4]
            )

            regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

            # ===== Transformer =====
            regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)

            fused_feature = self.fusion_net(
                regroup_feature,
                mask,
                spatial_correction_matrix
            )

            fused_feature = fused_feature.permute(0, 3, 1, 2)

            features.append(fused_feature)

        # ===== Output =====
        output_dict = {
            'psm': self.cls_head(features[0]),
            'rm': self.reg_head(features[0]),
            'source_feature': features[0],
            'target_feature': features[1],
            'target_psm': self.cls_head(features[1]),
            'target_rm': self.reg_head(features[1])
        }

        return output_dict
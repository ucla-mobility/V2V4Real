"""
Domain adaption modules
Author: Jinlong Li CSU PhD
"""
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv

import torch.nn.functional as F


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)


class DA_feature_Head(nn.Module):
    """
    Adds a simple Feature-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DA_feature_Head, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, feature):
        t = F.relu(self.conv1_da(feature))
        img_features = self.conv2_da(t)
        return img_features


class DA_instance_Head(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DA_instance_Head, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)
        self.in_channels = in_channels

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(nn.Module):
    """
    Domain Adaptation Model
    """

    def __init__(self, args):
        super(DomainAdaptationModule, self).__init__()

        self.feature_head = DA_feature_Head(args['DA_feature_head'])
        self.instance_head = DA_instance_Head(args['DA_instance_head'])

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fea_weight = args['DA_feature_weight']
        self.ins_weight = args['DA_instance_weight']

        self.grl_feature = GradientScalarLayer(
            -1.0 * args['grl_feature_weight'])
        self.grl_instance = GradientScalarLayer(
            -1.0 * args['grl_instance_weight'])

    def __call__(self, output_dict):
        # source_fea: [B, C, H, W] [B, 384, 48, 176]
        # target_fea: [B, C, H, W] [B, 384, 48, 176]
        # source_ins: [B, C, H, W][B, anchor_num, 48, 176]
        # target_ins: [B, C, H, W][B, anchor_num, 48, 176]

        source_fea = output_dict['source_feature']
        target_fea = output_dict['target_feature']

        source_psm = output_dict['psm']
        target_psm = output_dict['target_psm']

        # Loss of DA feature component
        source_grl_feature = self.grl_feature(source_fea)
        target_grl_feature = self.grl_feature(target_fea)

        da_source_feature = self.feature_head(source_grl_feature)
        da_target_feature = self.feature_head(target_grl_feature)
        # source label is 1
        da_source_fea_label = torch.ones_like(da_source_feature,
                                              dtype=torch.float32)
        # target label is 0
        da_target_fea_label = torch.zeros_like(da_target_feature,
                                               dtype=torch.float32)

        da_source_fea_level = da_source_feature.reshape(
            da_source_feature.shape[0], -1)  # [B, C*H*W]
        da_target_fea_level = da_target_feature.reshape(
            da_target_feature.shape[0], -1)  # [B, C*H*W]

        da_source_fea_label = da_source_fea_label.reshape(
            da_source_fea_label.shape[0], -1)  ##[B, C*H*W]
        da_target_fea_label = da_target_fea_label.reshape(
            da_target_fea_label.shape[0], -1)  # [B, C*H*W]

        da_fea = torch.cat([da_source_fea_level, da_target_fea_level],
                           dim=0)  # [B*2, C*H*W]
        da_fea_label = torch.cat([da_source_fea_label, da_target_fea_label],
                                 dim=0)  # [B*2, C*H*W]
        # da feature loss
        da_fea_loss = F.binary_cross_entropy_with_logits(da_fea, da_fea_label)

        # Loss of DA instance component

        source_psm_grl = self.grl_instance(source_psm)
        target_psm_grl = self.grl_instance(target_psm)
        # refer to PointPillarLoss------->[B, 48, 176, 2]
        cls_preds_source = source_psm_grl.permute(0, 2, 3,
                                                  1).contiguous()
        # refer to  PointPillarLoss------->[B, 48, 176, 2]
        cls_preds_target = target_psm_grl.permute(0, 2, 3,
                                                  1).contiguous()
        cls_preds_source = self.avgpool(cls_preds_source)
        cls_preds_target = self.avgpool(cls_preds_target)  ##[B, 48, 88, 1]
        # [B*H, C *W]====[B*H, 88*1]
        cls_preds_source = cls_preds_source.view(
            source_psm.shape[0] * source_psm.shape[2],
            -1)
        # [B*H, C *W]====[B*H, 88*1]
        cls_preds_target = cls_preds_target.view(
            target_psm.shape[0] * source_psm.shape[2],
            -1)

        da_ins_source = self.instance_head(
            cls_preds_source)  # [B*H, 2* 88]----> [B*H, 1]
        da_ins_target = self.instance_head(
            cls_preds_target)  # [B*H, 2* 88]----> [B*H, 1]
        # source label is 1
        da_source_ins_label = torch.ones_like(da_ins_source,
                                              dtype=torch.float32)
        # target label is 0
        da_target_ins_label = torch.zeros_like(da_ins_target,
                                               dtype=torch.float32)

        da_ins = torch.cat([da_ins_source, da_ins_target],
                           dim=0)
        da_ins_label = torch.cat([da_source_ins_label, da_target_ins_label],
                                 dim=0)

        # da instance loss
        da_ins_loss = F.binary_cross_entropy_with_logits(da_ins, da_ins_label)

        losses = {}
        if self.fea_weight > 0:
            losses['fea_loss'] = da_fea_loss * self.fea_weight
        if self.ins_weight > 0:
            losses['ins_loss'] = da_ins_loss * self.ins_weight

        return losses

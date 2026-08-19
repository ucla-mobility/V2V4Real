import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.
    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset
    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        if cav_id == 'ego':
            output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def inference_intermediate_fusion_with_variance(batch_data, model, dataset):
    """
    Same as inference_intermediate_fusion, but also returns the per-box
    regression variance aligned with pred_box_tensor. Requires the
    dataset's post_process_with_variance method (see
    IntermediateFusionDataset).

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.IntermediateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    pred_score : torch.Tensor
    pred_var_tensor : torch.Tensor or None
        Shape (N, 7), one variance vector per surviving predicted box.
        None if the model has no uncertainty output (e.g. baseline model).
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor = \
        dataset.post_process_with_variance(batch_data, output_dict)

    return pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """

    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    # Save GT and PCD always
    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

    # Handle prediction safely
    if pred_tensor is not None:
        pred_np = torch_tensor_to_numpy(pred_tensor)
        np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    else:
        print(f"[Warning] No prediction at frame {timestamp}")


def save_prediction_gt_with_variance(pred_tensor, gt_tensor, pcd,
                                     timestamp, save_path,
                                     pred_var_tensor=None,
                                     pred_score=None):
    """
    Same as save_prediction_gt(), plus optionally saves the per-box
    variance and per-box confidence score, so a real frame's uncertainty
    can be plotted later without re-running the model.

    Files written (N = number of surviving predicted boxes):
        %04d_pcd.npy    (P, 4)    lidar points, xyzi           -- unchanged
        %04d_gt.npy     (G, 8, 3) GT box corners                -- unchanged
        %04d_pred.npy   (N, 8, 3) predicted box corners         -- unchanged
        %04d_var.npy    (N, 7)    predicted variance per box    -- NEW, only
                                    if pred_var_tensor is not None
        %04d_score.npy  (N,)      predicted confidence per box  -- NEW, only
                                    if pred_score is not None
    """
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    # Save GT and PCD always
    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

    # Handle prediction safely
    if pred_tensor is not None:
        pred_np = torch_tensor_to_numpy(pred_tensor)
        np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    else:
        print(f"[Warning] No prediction at frame {timestamp}")
        return

    if pred_var_tensor is not None:
        var_np = torch_tensor_to_numpy(pred_var_tensor)
        np.save(os.path.join(save_path, '%04d_var.npy' % timestamp), var_np)

    if pred_score is not None:
        score_np = torch_tensor_to_numpy(pred_score)
        np.save(os.path.join(save_path, '%04d_score.npy' % timestamp), score_np)
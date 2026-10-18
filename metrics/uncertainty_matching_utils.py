"""
uncertainty_matching_utils.py

One shared detection-to-GT matching protocol for uncertainty-quality evaluation
across MC-Dropout, VN-NLL, VN-Energy and Laplace.

For final post-NMS detections:
1. Sort predictions by decreasing confidence.
2. Greedily match to the highest-BEV-IoU GT not already used.
3. Keep only IoU >= threshold (default 0.5).
4. Pair 3-D centre localization error (metres) with that prediction's
   regression uncertainty.
5. Each GT is matched at most once.

This replaces the old nearest-GT + 2 metre threshold rule.
"""

import numpy as np
import torch
from shapely.geometry import MultiPoint


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def box_centers_3d(boxes):
    boxes = _to_numpy(boxes)
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError(f"Expected boxes [N,8,3], got {boxes.shape}")
    return boxes.mean(axis=1).astype(np.float64, copy=False)


def _bev_polygon(box):
    xy = np.asarray(box, dtype=np.float64)[:, :2]
    poly = MultiPoint(xy).convex_hull
    if poly.is_empty or poly.area <= 0.0:
        return None
    return poly


def bev_iou_matrix(pred_boxes, gt_boxes):
    pred_boxes = _to_numpy(pred_boxes)
    gt_boxes = _to_numpy(gt_boxes)
    n_pred = 0 if pred_boxes is None else len(pred_boxes)
    n_gt = 0 if gt_boxes is None else len(gt_boxes)
    out = np.zeros((n_pred, n_gt), dtype=np.float64)
    if n_pred == 0 or n_gt == 0:
        return out

    pred_polys = [_bev_polygon(b) for b in pred_boxes]
    gt_polys = [_bev_polygon(b) for b in gt_boxes]
    for i, p in enumerate(pred_polys):
        if p is None:
            continue
        for j, g in enumerate(gt_polys):
            if g is None:
                continue
            inter = p.intersection(g).area
            union = p.area + g.area - inter
            if union > 0.0:
                out[i, j] = inter / union
    return out


def centre_variance_m2(var, anchor_args, dim_axis=-1):
    """
    Box-centre variance in m^2 from encoded regression variance.

    The encoded order is [x, y, z, h, w, l, yaw]; x/y are divided by the anchor
    diagonal sqrt(l^2 + w^2) and z by the anchor height
    (VoxelPostprocessor.delta_to_boxes3d). Returns
    d^2 * (var_x + var_y) + h^2 * var_z, i.e. the expected squared 3-D centre
    error, which matches the centre error used for the error/uncertainty pairs.
    Works on torch tensors and numpy arrays; `dim_axis` is the 7-dim axis.
    """
    d2 = float(anchor_args['l']) ** 2 + float(anchor_args['w']) ** 2
    h2 = float(anchor_args['h']) ** 2
    if torch.is_tensor(var):
        comp = lambda k: var.select(dim_axis, k)
    else:
        comp = lambda k: np.take(var, k, axis=dim_axis)
    return d2 * (comp(0) + comp(1)) + h2 * comp(2)


def reduce_per_box_variance(pred_var_tensor, n_pred, anchor_args=None):
    """
    Produce one scalar uncertainty per final box.
    [N] or [N,1] stays scalar (already reduced by the caller);
    [N,7] is reduced to the box-centre variance in m^2 (needs `anchor_args`).
    """
    var = _to_numpy(pred_var_tensor)
    if var is None:
        return np.empty((0,), dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    if var.ndim == 0 or var.shape[0] != n_pred:
        raise ValueError(
            f"Variance/prediction mismatch: variance shape {var.shape}, "
            f"predictions {n_pred}."
        )
    if var.ndim == 1 or (var.ndim == 2 and var.shape[1] == 1):
        scalar = var.reshape(n_pred)
    elif var.ndim == 2 and var.shape[1] == 7:
        if anchor_args is None:
            raise ValueError("anchor_args is required to reduce [N,7] variance.")
        scalar = centre_variance_m2(var, anchor_args, dim_axis=1)
    else:
        raise ValueError(f"Unexpected per-box variance shape {var.shape}.")
    if not np.all(np.isfinite(scalar)):
        raise ValueError("Per-box uncertainty contains NaN/Inf.")
    return scalar


def match_error_uncertainty_pairs(
    pred_box_tensor,
    pred_score,
    pred_var_tensor,
    gt_box_tensor,
    iou_threshold=0.5,
    anchor_args=None,
):
    """Return one-to-one AP-style TP error/uncertainty pairs."""
    pred_boxes = _to_numpy(pred_box_tensor)
    gt_boxes = _to_numpy(gt_box_tensor)
    scores = _to_numpy(pred_score)

    n_pred = 0 if pred_boxes is None else len(pred_boxes)
    n_gt = 0 if gt_boxes is None else len(gt_boxes)
    empty = {
        "errors_m": np.empty(0, dtype=np.float64),
        "uncertainties": np.empty(0, dtype=np.float64),
        "pred_indices": np.empty(0, dtype=np.int64),
        "gt_indices": np.empty(0, dtype=np.int64),
        "ious": np.empty(0, dtype=np.float64),
    }
    if n_pred == 0 or n_gt == 0 or pred_var_tensor is None:
        return empty
    if scores is None:
        raise ValueError("pred_score is required for AP-style matching.")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != n_pred:
        raise ValueError(f"Got {len(scores)} scores for {n_pred} predictions.")

    uncertainty = reduce_per_box_variance(pred_var_tensor, n_pred, anchor_args)
    ious = bev_iou_matrix(pred_boxes, gt_boxes)
    pred_centers = box_centers_3d(pred_boxes)
    gt_centers = box_centers_3d(gt_boxes)

    order = np.argsort(-scores, kind="stable")
    used_gt = set()
    errors, uncs, pred_ids, gt_ids, matched_ious = [], [], [], [], []

    for pred_idx in order:
        available = [j for j in range(n_gt) if j not in used_gt]
        if not available:
            break
        candidate_ious = ious[pred_idx, available]
        gt_idx = available[int(np.argmax(candidate_ious))]
        best_iou = float(ious[pred_idx, gt_idx])
        if best_iou < float(iou_threshold):
            continue

        used_gt.add(gt_idx)
        errors.append(float(np.linalg.norm(
            pred_centers[pred_idx] - gt_centers[gt_idx]
        )))
        uncs.append(float(uncertainty[pred_idx]))
        pred_ids.append(int(pred_idx))
        gt_ids.append(int(gt_idx))
        matched_ious.append(best_iou)

    return {
        "errors_m": np.asarray(errors, dtype=np.float64),
        "uncertainties": np.asarray(uncs, dtype=np.float64),
        "pred_indices": np.asarray(pred_ids, dtype=np.int64),
        "gt_indices": np.asarray(gt_ids, dtype=np.int64),
        "ious": np.asarray(matched_ious, dtype=np.float64),
    }
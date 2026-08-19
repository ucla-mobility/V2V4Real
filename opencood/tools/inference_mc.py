import argparse
import os
import time

import torch
import torch.nn as nn
import open3d as o3d
from torch.utils.data import DataLoader
import cv2
import csv
import numpy as np

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils


def save_uncertainty_maps(var_psm, var_rm, save_dir, frame_id):

    os.makedirs(save_dir, exist_ok=True)

    psm_map = var_psm.mean(dim=1).squeeze().detach().cpu().numpy()
    psm_max = psm_map.max()
    if psm_max > 0:
        psm_map = psm_map / psm_max
    psm_map = np.uint8(psm_map * 255)
    psm_color = cv2.applyColorMap(psm_map, cv2.COLORMAP_JET)
    psm_color = cv2.resize(psm_color, (800, 400))

    rm_map = var_rm.mean(dim=1).squeeze().detach().cpu().numpy()
    rm_max = rm_map.max()
    if rm_max > 0:
        rm_map = rm_map / rm_max
    rm_map = np.uint8(rm_map * 255)
    rm_color = cv2.applyColorMap(rm_map, cv2.COLORMAP_JET)
    rm_color = cv2.resize(rm_color, (800, 400))

    cv2.imwrite(
        os.path.join(save_dir, f"psm_uncertainty_{frame_id:05d}.png"),
        psm_color
    )
    cv2.imwrite(
        os.path.join(save_dir, f"rm_uncertainty_{frame_id:05d}.png"),
        rm_color
    )


def enable_mc_dropout(model):
    """
    Keep model in eval mode, but enable dropout layers only.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def check_dropout_exists(model):
    """
    MC-Dropout only produces meaningful (non-zero) variance if the model
    actually contains at least one Dropout layer. If none exist, every
    stochastic forward pass is bit-identical, and every uncertainty number
    computed below is silently exactly zero, regardless of mc_passes.
    Run this ONCE before trusting any results from this script.
    """
    found = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            found.append(name)

    if len(found) == 0:
        print("=" * 70)
        print("[WARNING] No nn.Dropout layer found anywhere in this model.")
        print("MC-Dropout requires at least one Dropout layer to produce any")
        print("stochasticity. Every one of your mc_passes forward passes will")
        print("be BIT-IDENTICAL, and every uncertainty number reported below")
        print("will be exactly zero. Add a Dropout layer (e.g. before")
        print("cls_head/reg_head) and retrain before trusting these results.")
        print("=" * 70)
    else:
        print(f"[MC-Dropout check] Found {len(found)} Dropout layer(s): {found}")

    return len(found) > 0


def probability_to_logit(probability, eps=1e-6):
    """
    sigmoid(logit(p)) == p -- lets the T-pass-averaged probability be fed
    back through the normal OpenCOOD postprocessor, which expects raw psm
    logits (same trick used in the Laplace pipeline's
    DiagonalLaplace.probability_to_logit).
    """
    p = probability.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def nearest_anchor_value(box_center_xy, dense_map, cav_lidar_range):
    """
    dense_map: (anchor_number, H, W) tensor.
    Returns the scalar value at the anchor grid cell nearest to
    box_center_xy (real-world x, y), averaged over the anchor_number
    (rotation) channel dimension. Same technique used in
    laplace_uncertainty_analysis.py, kept consistent here so MC-Dropout
    and Laplace uncertainty numbers are computed the same way.
    """
    xmin, ymin, _, xmax, ymax, _ = cav_lidar_range
    _, H, W = dense_map.shape
    x, y = box_center_xy
    col = int((x - xmin) / (xmax - xmin) * W)
    row = int((y - ymin) / (ymax - ymin) * H)
    col = min(max(col, 0), W - 1)
    row = min(max(row, 0), H - 1)
    return dense_map[:, row, col].mean().item()


def get_box_center_xy(box_corners):
    pts = np.asarray(box_corners)[:, :2]
    return pts.mean(axis=0)


def compute_ece(confidences, corrects, n_bins=10):
    """
    Quick inline Expected Calibration Error, for an immediate console
    readout. Same logic as compute_classification_calibration.py, which
    is the script to use for the full per-bin breakdown / combined
    reliability diagram across runs.
    """
    confidences = np.asarray(confidences)
    corrects = np.asarray(corrects)
    n = len(confidences)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = corrects[mask].mean()
        conf_b = confidences[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return ece


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--mc_passes', type=int, default=20,
                        help='number of stochastic MC-Dropout forward '
                             'passes per frame')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    cav_lidar_range = hypes['postprocess']['anchor_args']['cav_lidar_range']

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)
    print("DATASET TYPE:", type(opencood_dataset))
    print("Dataset length:", len(opencood_dataset))
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)

    mc_passes = opt.mc_passes
    if opt.fusion_method == 'intermediate':
        enable_mc_dropout(model)
        # CRITICAL CHECK: MC-Dropout is silently a no-op if the model has
        # no Dropout layer anywhere. Verify before spending any GPU time.
        check_dropout_exists(model)
    else:
        model.eval()

    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True
        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    total_var_psm = 0.0
    total_var_rm = 0.0
    count = 0

    # NOTE: pred variance is now in ENCODED regression-target space (same
    # convention as the Laplace pipeline's var_reg_encoded), not decoded
    # physical meters -- keeps MC-Dropout and Laplace numbers on the same
    # footing for direct comparison.
    prediction_errors = []
    prediction_variances = []

    classification_confidences = []
    classification_correct = []

    # Standardized per-frame files used by plot_all_uncertainty_methods.py
    visualization_npz_dir = os.path.join(opt.model_dir, 'visualization_npz_mc')
    os.makedirs(visualization_npz_dir, exist_ok=True)

    for i, batch_data in enumerate(data_loader):
        frame_start = time.time()
        print(f'\n===== Frame {i}/{len(data_loader)} =====', flush=True)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)

            elif opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                # ==========================================================
                # Detection-level MC Dropout -- FAST version.
                #
                # Previously: each of the T stochastic passes was decoded +
                # NMS'd independently, then detections were associated
                # across passes with a Python-level greedy nearest-track
                # matcher: T separate NMS calls plus an O(T x N^2)
                # association loop per frame -- the dominant cost after the
                # T forward passes themselves.
                #
                # Now: average psm/rm across the T passes FIRST (in
                # probability space for classification), run NMS ONCE on
                # the averaged output, and recover per-box uncertainty via
                # a nearest-anchor spatial lookup on the averaged variance
                # maps. This is the standard MCBI approach (Yun et al.,
                # CoRL 2022, Algorithm 1), and matches exactly how the
                # Laplace pipeline computes uncertainty for this same
                # model -- so MC-Dropout and Laplace results become
                # directly, apples-to-apples comparable. 1 NMS call
                # instead of T, no association loop.
                #
                # Trade-off: this can no longer express "the object was
                # completely missed in some passes" as a separate signal
                # the way the old track-based min_track_observations
                # filter could -- if you need that behavior back, the
                # per-pass-NMS approach is the one to restore.
                # ==========================================================
                psm_samples = []
                rm_samples = []

                for mc_idx in range(mc_passes):
                    output_t = model(batch_data['ego'])
                    psm_samples.append(output_t['psm'].detach())
                    rm_samples.append(output_t['rm'].detach())

                psm_stack = torch.stack(psm_samples, dim=0)   # T,B,anchor_num,H,W
                rm_stack = torch.stack(rm_samples, dim=0)     # T,B,7*anchor_num,H,W

                cls_prob_samples = torch.sigmoid(psm_stack)
                mean_cls_prob = cls_prob_samples.mean(dim=0)
                var_cls_prob = cls_prob_samples.var(dim=0, unbiased=False)

                mean_rm = rm_stack.mean(dim=0)
                var_rm = rm_stack.var(dim=0, unbiased=False)

                total_var_psm += var_cls_prob.mean().item()
                total_var_rm += var_rm.mean().item()
                count += 1

                # Single NMS call on the T-pass-averaged prediction.
                bayes_psm = probability_to_logit(mean_cls_prob)
                output_dict_mean = {'ego': {'psm': bayes_psm, 'rm': mean_rm}}

                pred_box_tensor, pred_score, gt_box_tensor = \
                    opencood_dataset.post_process(batch_data, output_dict_mean)

                # Per-box regression uncertainty via nearest-anchor lookup
                # into the averaged variance map (same technique used in
                # laplace_uncertainty_analysis.py's Step 9 analysis).
                if pred_box_tensor is not None and len(pred_box_tensor) > 0:
                    var_cls_map = var_cls_prob.squeeze(0)          # anchor_num,H,W
                    n_anchor = var_cls_map.shape[0]
                    var_rm_map = var_rm.squeeze(0).view(
                        n_anchor, 7, *var_rm.shape[2:]).mean(dim=1)  # anchor_num,H,W

                    per_box_var = []
                    for box_idx in range(len(pred_box_tensor)):
                        center_xy = get_box_center_xy(
                            pred_box_tensor[box_idx].detach().cpu().numpy())
                        per_box_var.append(
                            nearest_anchor_value(center_xy, var_rm_map,
                                                 cav_lidar_range))
                    pred_var_tensor = torch.tensor(
                        per_box_var, device=device).unsqueeze(-1)  # K x 1
                else:
                    pred_var_tensor = None

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            if opt.fusion_method != 'intermediate':
                pred_var_tensor = None

            # ==========================================================
            # Regression: per-box (error, predicted variance) pairs for
            # the Error-vs-Variance correlation / AUSE-UCE report below.
            # ==========================================================
            if (pred_var_tensor is not None and pred_box_tensor is not None
                    and gt_box_tensor is not None and len(gt_box_tensor) > 0
                    and len(pred_box_tensor) > 0):
                pred_centers_reg = pred_box_tensor.mean(
                    dim=1).detach().cpu().numpy()
                gt_centers_reg = gt_box_tensor.mean(
                    dim=1).detach().cpu().numpy()
                dist_matrix_reg = cdist(pred_centers_reg, gt_centers_reg)
                nearest_gt_idx_reg = dist_matrix_reg.argmin(axis=1)
                nearest_gt_dist_reg = dist_matrix_reg.min(axis=1)

                match_threshold = 2.0  # meters; same as classification match
                var_np = pred_var_tensor.detach().cpu().numpy()  # N x 1

                for box_idx in range(len(pred_box_tensor)):
                    if nearest_gt_dist_reg[box_idx] >= match_threshold:
                        continue
                    prediction_errors.append(
                        float(nearest_gt_dist_reg[box_idx]))
                    prediction_variances.append(
                        float(var_np[box_idx].mean()))

            # ==========================================================
            # Classification calibration (per final detected box, ECE)
            # ==========================================================
            if pred_box_tensor is not None and pred_score is not None:
                scores_np = pred_score.detach().cpu().numpy()
                if len(scores_np) > 0:
                    if gt_box_tensor is not None and len(gt_box_tensor) > 0:
                        pred_centers_cls = pred_box_tensor.mean(
                            dim=1).detach().cpu().numpy()
                        gt_centers_cls = gt_box_tensor.mean(
                            dim=1).detach().cpu().numpy()
                        dist_matrix_cls = cdist(pred_centers_cls, gt_centers_cls)
                        nearest_gt_dist_cls = dist_matrix_cls.min(axis=1)
                        match_threshold = 2.0
                        is_correct = (nearest_gt_dist_cls < match_threshold).astype(np.float32)
                    else:
                        is_correct = np.zeros(len(scores_np), dtype=np.float32)

                    classification_confidences.extend(scores_np.tolist())
                    classification_correct.extend(is_correct.tolist())
            # ==========================================================

            # ==========================================================
            # Standardized uncertainty visualization package (MC Dropout)
            # This only SAVES quantities already computed above.
            # It does not change MC-Dropout inference or evaluation.
            # ==========================================================
            if opt.fusion_method == 'intermediate':
                pcd_np = batch_data['ego']['origin_lidar'][0]
                if isinstance(pcd_np, torch.Tensor):
                    pcd_np = pcd_np.detach().cpu().numpy()
                else:
                    pcd_np = np.asarray(pcd_np)

                pred_np = (pred_box_tensor.detach().cpu().numpy()
                           if pred_box_tensor is not None
                           else np.empty((0, 8, 3), dtype=np.float32))
                gt_np = (gt_box_tensor.detach().cpu().numpy()
                         if gt_box_tensor is not None
                         else np.empty((0, 8, 3), dtype=np.float32))
                score_np = (pred_score.detach().cpu().numpy()
                            if pred_score is not None
                            else np.empty((0,), dtype=np.float32))

                np.savez_compressed(
                    os.path.join(visualization_npz_dir, f"{i:05d}.npz"),
                    method=np.asarray("MC Dropout"),
                    pcd=pcd_np,
                    gt_boxes=gt_np,
                    pred_boxes=pred_np,
                    pred_scores=score_np,
                    classification_uncertainty=var_cls_prob.detach().cpu().numpy(),
                    regression_uncertainty=var_rm.detach().cpu().numpy(),
                    cls_variance=var_cls_prob.detach().cpu().numpy(),
                    reg_variance=var_rm.detach().cpu().numpy(),
                    mean_cls_prob=mean_cls_prob.detach().cpu().numpy(),
                    mc_passes=np.asarray(mc_passes, dtype=np.int32),
                    cav_lidar_range=np.asarray(cav_lidar_range, dtype=np.float32),
                )
            # ==========================================================

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.7)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_short, 0.5,
                                       left_range=0, right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_short, 0.7,
                                       left_range=0, right_range=30)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_middle, 0.5,
                                       left_range=30, right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_middle, 0.7,
                                       left_range=30, right_range=50)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_long, 0.5,
                                       left_range=50, right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_long, 0.7,
                                       left_range=50, right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy_mc')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt_with_variance(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path,
                    pred_var_tensor=pred_var_tensor,
                    pred_score=pred_score)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis, vis_aabbs_pred,
                                                 pred_o3d_box, update_mode='add')
                    vis_utils.linset_assign_list(vis, vis_aabbs_gt,
                                                 gt_o3d_box, update_mode='add')

                vis_utils.linset_assign_list(vis, vis_aabbs_pred, pred_o3d_box)
                vis_utils.linset_assign_list(vis, vis_aabbs_gt, gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    if count > 0:
        avg_var_psm = total_var_psm / count
        avg_var_rm = total_var_rm / count
        print("Average MC uncertainty / variance psm (raw feature map):", avg_var_psm)
        print("Average MC uncertainty / variance rm (raw feature map):", avg_var_rm)

    if len(prediction_errors) > 10:
        corr, p_value = pearsonr(prediction_errors, prediction_variances)
        s_corr, s_p_value = spearmanr(prediction_errors, prediction_variances)
        print("====================================")
        print(f"Error vs Variance Pearson correlation:  {corr:.4f} "
              f"(p={p_value:.4g})")
        print(f"Error vs Variance Spearman correlation: {s_corr:.4f} "
              f"(p={s_p_value:.4g})")
        print(f"Matched boxes used: {len(prediction_errors)}")
        print("====================================")

        err_var_csv = os.path.join(opt.model_dir, "error_vs_variance_mc.csv")
        with open(err_var_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["error_m", "mean_variance"])
            for e, v in zip(prediction_errors, prediction_variances):
                writer.writerow([e, v])
        print(f"Per-box error/variance pairs saved to {err_var_csv}")
    else:
        print("Not enough matched boxes to compute Error-vs-Variance "
              "correlation (need >10 matched boxes across the test set).")

    if len(classification_confidences) > 10:
        ece = compute_ece(classification_confidences, classification_correct)
        print("====================================")
        print(f"Classification ECE: {ece:.5f} "
              f"(lower is better, 0 = perfectly calibrated)")
        print(f"Boxes used: {len(classification_confidences)}")
        print("====================================")

        cls_csv = os.path.join(opt.model_dir, "classification_calibration_mc.csv")
        with open(cls_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["confidence", "correct"])
            for c, k in zip(classification_confidences, classification_correct):
                writer.writerow([c, k])
        print(f"Per-box confidence/correctness pairs saved to {cls_csv}")
    else:
        print("Not enough predicted boxes to compute classification ECE "
              "(need >10 boxes across the test set).")

    eval_utils.eval_final_results(result_stat, opt.model_dir)
    eval_utils.eval_final_results(result_stat_short, opt.model_dir, "short")
    eval_utils.eval_final_results(result_stat_middle, opt.model_dir, "middle")
    eval_utils.eval_final_results(result_stat_long, opt.model_dir, "long")
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
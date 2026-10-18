import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import open3d as o3d
from torch.utils.data import DataLoader
import cv2
import csv
import numpy as np


import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from metrics.calibration_utils import PaperCalibrationAccumulator


from metrics.uncertainty_matching_utils import match_error_uncertainty_pairs

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
    If the model has a dedicated layer named 'mc_dropout' (our MC models),
    only that layer is switched on; other dropout layers inside the network
    (e.g. in CoBEVT's attention fusion) stay off, so the MC method is the same
    for every architecture. Models without it: all dropout layers (old
    behaviour; F-Cooper MC has only mc_dropout, so it is unchanged).
    """
    model.eval()
    dropouts = [(name, m) for name, m in model.named_modules()
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))]
    dedicated = [(name, m) for name, m in dropouts
                 if name.split('.')[-1] == 'mc_dropout']
    for name, m in (dedicated or dropouts):
        m.train()
    print(f"[MC-Dropout] active at test time: {[n for n, _ in (dedicated or dropouts)]}")


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


@torch.no_grad()
def estimate_noise_var(model, hypes, device, n_frames, train_dataset='opv2v'):
    """
    Homoscedastic observation-noise variance per encoded regression dim
    [x, y, z, h, w, l, sin(dyaw)] = mean squared residual of the
    deterministic (dropout-off) model on positive anchors of n_frames
    TRAINING frames. This is the tau^-1 term of MC-dropout regression
    (Gal & Ghahramani, 2016): predictive var = tau^-1 + sample variance.
    """
    train_hypes = copy.deepcopy(hypes)
    train_hypes['dataset'] = train_dataset
    ds = build_dataset(train_hypes, visualize=False, train=True)
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False,
                        collate_fn=ds.collate_batch_train, drop_last=False)
    model.eval()
    sq_sum = torch.zeros(7, device=device, dtype=torch.float64)
    count = 0
    for i, batch in enumerate(loader):
        if i >= n_frames:
            break
        batch = train_utils.to_device(batch, device)
        out = model(batch['ego'])
        lab = batch['ego']['label_dict']
        rm = out['rm'].permute(0, 2, 3, 1).reshape(-1, 7)
        tgt = lab['targets'].reshape(-1, 7)
        pos = lab['pos_equal_one'].reshape(-1) > 0
        if pos.any():
            r = rm[pos] - tgt[pos]
            r[:, 6] = torch.sin(r[:, 6])
            r = r[torch.isfinite(r).all(dim=1)]
            sq_sum += r.double().pow(2).sum(dim=0)
            count += r.shape[0]
    noise_var = (sq_sum / max(count, 1)).float()
    print(f'[MC] Observation noise variance from {min(i + 1, n_frames)} '
          f'training frames [x,y,z,h,w,l,sin(yaw)]: '
          f'{[round(v, 6) for v in noise_var.tolist()]}')
    return noise_var


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
    parser.add_argument('--regression_variance', default='total',
                        choices=['total', 'epistemic'],
                        help="'total': MC sample variance + observation noise "
                             "(aleatoric) estimated on training data; "
                             "'epistemic': MC sample variance only.")
    parser.add_argument('--noise_frames', type=int, default=500,
                        help='training frames used to estimate the '
                             'observation noise (total variance only)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for the dropout masks')
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
    torch.manual_seed(opt.seed)

    noise_var_map = None
    if opt.fusion_method == 'intermediate' and \
            opt.regression_variance == 'total':
        noise_var = estimate_noise_var(model, hypes, device, opt.noise_frames)

    if opt.fusion_method == 'intermediate':
        enable_mc_dropout(model)
        # CRITICAL CHECK: MC-Dropout is silently a no-op if the model has
        # no Dropout layer anywhere. Verify before spending any GPU time.
        check_dropout_exists(model)
    else:
        model.eval()

    # Same IoU thresholds (0.3 / 0.5) as the baseline, VN and Laplace scripts.
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.3: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.3: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.3: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.3: {'tp': [], 'fp': [], 'gt': 0}}

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

    # Feng et al. paper-style dense classification + marginal regression ECE.
    paper_calibration = PaperCalibrationAccumulator(n_cls_bins=10, n_reg_levels=10)

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
                # the variance of the anchor that produced each final box
                # (carried through post_process_with_variance). This is the standard MCBI approach (Yun et al.,
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
                var_rm_epistemic = rm_stack.var(dim=0, unbiased=False)
                var_rm = var_rm_epistemic
                if opt.regression_variance == 'total':
                    if noise_var_map is None:
                        # rm channels are anchor-major: channel = anchor*7 + dim
                        n_anchor_rm = var_rm.shape[1] // 7
                        noise_var_map = noise_var.repeat(n_anchor_rm).view(
                            1, -1, 1, 1)
                    var_rm = var_rm_epistemic + noise_var_map

                total_var_psm += var_cls_prob.mean().item()
                total_var_rm += var_rm.mean().item()
                count += 1

                # Paper-style calibration uses dense object/background labels and
                # positive-anchor encoded regression targets. No 2 m matching.
                paper_calibration.update(
                    cls_prob=mean_cls_prob,
                    reg_mean=mean_rm,
                    reg_var=var_rm,
                    batch_data=batch_data,
                )

                # Single NMS call on the T-pass-averaged prediction. The
                # regression variance is carried through the same threshold /
                # NMS / range filter, so each final box gets the variance of
                # the anchor that produced it (same protocol as VN and
                # Laplace). Boxes and scores are identical to post_process.
                bayes_psm = probability_to_logit(mean_cls_prob)
                output_dict_mean = {'ego': {
                    'psm': bayes_psm, 'rm': mean_rm,
                    'rm_var': var_rm.expand_as(mean_rm)}}

                pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor = \
                    opencood_dataset.post_process_with_variance(
                        batch_data, output_dict_mean)

            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            if opt.fusion_method != 'intermediate':
                pred_var_tensor = None

            # ==========================================================
            # Regression uncertainty-quality pairs.
            # Shared one-to-one AP-style BEV-IoU matching (IoU >= 0.5).
            # ==========================================================
            match_result = match_error_uncertainty_pairs(
                pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor,
                iou_threshold=0.5,
                anchor_args=hypes['postprocess']['anchor_args'])
            prediction_errors.extend(match_result["errors_m"].tolist())
            prediction_variances.extend(match_result["uncertainties"].tolist())

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
                    epistemic_reg_variance=var_rm_epistemic.detach().cpu().numpy(),
                    mean_cls_prob=mean_cls_prob.detach().cpu().numpy(),
                    mc_passes=np.asarray(mc_passes, dtype=np.int32),
                    cav_lidar_range=np.asarray(cav_lidar_range, dtype=np.float32),
                )
            # ==========================================================

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat, 0.3)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_short, 0.5,
                                       left_range=0, right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_short, 0.3,
                                       left_range=0, right_range=30)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_middle, 0.5,
                                       left_range=30, right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_middle, 0.3,
                                       left_range=30, right_range=50)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_long, 0.5,
                                       left_range=50, right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                       gt_box_tensor, result_stat_long, 0.3,
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

    # Save matched TP error/uncertainty pairs only.
    # Pearson, Spearman and AUSE are computed centrally in metrics/.
    err_var_csv = os.path.join(opt.model_dir, "error_vs_variance_mc.csv")
    with open(err_var_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_m", "mean_variance"])
        writer.writerows(zip(prediction_errors, prediction_variances))
    print(f"Saved {len(prediction_errors)} matched error/variance pairs to {err_var_csv}")

    # Canonical paper-style classification + regression ECE.
    paper_calibration.save(opt.model_dir, "mc_dropout")

    eval_utils.eval_final_results(result_stat, opt.model_dir)
    eval_utils.eval_final_results(result_stat_short, opt.model_dir, "short")
    eval_utils.eval_final_results(result_stat_middle, opt.model_dir, "middle")
    eval_utils.eval_final_results(result_stat_long, opt.model_dir, "long")
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
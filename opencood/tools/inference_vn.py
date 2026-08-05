import argparse
import os
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import csv
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist


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
                             'it cannot be set true with show_vis together')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether the dataset is simulation dataset')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize the results in single image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
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
    model.eval()

    # Evaluation dictionaries
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}

    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}

    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}

    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    # ---------------- VN uncertainty storage ----------------
    vn_mean_logvar_list = []
    vn_mean_var_list = []

    # Optional: per-dimension uncertainty
    vn_dim_var_list = []
    vn_statistics = []
    all_frame_variances = []
    all_frame_logvars = []
    # --------------------------------------------------------

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

    prediction_errors = []
    prediction_variances = []

    for i, batch_data in enumerate(data_loader):
        print(i)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            batch_data = train_utils.to_device(batch_data, device)

            # ==========================================================
            # VN uncertainty calculation
            # ==========================================================
            # This is for variance network model only.
            # It will be ignored automatically for normal baseline model.
            #
            # rvar = log variance
            # variance = exp(rvar)
            #
            # NOTE:
            # This calls model once for uncertainty.
            # Then inference_utils calls model again for normal prediction.
            # This is okay for debugging, but slower.
            # ==========================================================
            vn_output_dict = model(batch_data['ego'])

            if 'rm_logvar' in vn_output_dict:
                rlogvar = vn_output_dict['rm_logvar']

                if 'rm_var' in vn_output_dict:
                    var = vn_output_dict['rm_var']
                else:
                    var = torch.exp(rlogvar)

                mean_logvar = rlogvar.mean().item()
                mean_var = var.mean().item()

                max_var = var.max().item()
                min_var = var.min().item()
                std_var = var.std().item()
                median_var = var.median().item()

                vn_mean_logvar_list.append(mean_logvar)
                vn_mean_var_list.append(mean_var)

                # var shape: [B, 7*anchor_num, H, W] -- same flat channel
                # layout as 'rm'. Reshape the same way the loss function
                # does: move channels to the end, then split into groups
                # of 7 (one group per anchor).
                B, C, H, W = var.shape

                # [B, C, H, W] -> [B, H, W, C]
                var_per_dim = var.permute(0, 2, 3, 1).contiguous()

                # [B, H, W, C] -> [B, H*W*anchor_num, 7]
                var_per_dim = var_per_dim.reshape(B, -1, 7)

                # Mean uncertainty for each regression dimension
                mean_var_per_dim = var_per_dim.mean(dim=(0, 1))

                vn_dim_var_list.append(mean_var_per_dim.detach().cpu())
                vn_statistics.append({
                    "frame": i,
                    "mean_logvar": mean_logvar,
                    "mean_var": mean_var,
                    "median_var": median_var,
                    "max_var": max_var,
                    "min_var": min_var,
                    "std_var": std_var,
                    "x": mean_var_per_dim[0].item(),
                    "y": mean_var_per_dim[1].item(),
                    "z": mean_var_per_dim[2].item(),
                    "h": mean_var_per_dim[3].item(),
                    "w": mean_var_per_dim[4].item(),
                    "l": mean_var_per_dim[5].item(),
                    "yaw": mean_var_per_dim[6].item()
                })

                all_frame_variances.append(var.detach().cpu().numpy())

                all_frame_logvars.append(rlogvar.detach().cpu().numpy())

                print("VN Mean LogVar:", mean_logvar)
                print("VN Mean Variance:", mean_var)
                print("VN Mean Variance per dim [x,y,z,h,w,l,yaw]:",
                      mean_var_per_dim.detach().cpu().numpy())
            # ==========================================================

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
                pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion_with_variance(
                        batch_data,
                        model,
                        opencood_dataset)

            else:
                raise NotImplementedError('Only early, late and intermediate fusion is supported.')

            if opt.fusion_method != 'intermediate':
                pred_var_tensor = None

            # ==========================================================
            # Error vs Variance correlation
            # ==========================================================
            # For every predicted box, find its nearest ground-truth box
            # (by center distance). If that distance is within
            # match_threshold, treat it as a matched prediction: pair its
            # error (distance to matched GT) with its own predicted
            # variance (averaged over the 7 box dimensions).
            # ==========================================================
            if pred_var_tensor is not None and pred_box_tensor is not None \
                    and gt_box_tensor is not None:
                pred_centers = pred_box_tensor.mean(dim=1).detach().cpu().numpy()
                gt_centers = gt_box_tensor.mean(dim=1).detach().cpu().numpy()

                if len(pred_centers) > 0 and len(gt_centers) > 0:
                    dist_matrix = cdist(pred_centers, gt_centers)
                    nearest_gt_dist = dist_matrix.min(axis=1)

                    match_threshold = 2.0  # meters; adjust if needed
                    matched = nearest_gt_dist < match_threshold

                    if matched.any():
                        per_box_mean_var = pred_var_tensor.mean(
                            dim=1).detach().cpu().numpy()
                        prediction_errors.extend(
                            nearest_gt_dist[matched].tolist())
                        prediction_variances.extend(
                            per_box_mean_var[matched].tolist())
            # ==========================================================

            # Overall AP calculation
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            # Short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # Middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # Long range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)

                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego']['origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''

                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego']['origin_lidar'][0],
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

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    # ---------------- Final VN uncertainty report ----------------
    if len(vn_mean_var_list) > 0:
        avg_logvar = sum(vn_mean_logvar_list) / len(vn_mean_logvar_list)
        avg_var = sum(vn_mean_var_list) / len(vn_mean_var_list)

        print("====================================")
        print("Average VN Mean LogVar:", avg_logvar)
        print("Average VN Mean Variance:", avg_var)

        if len(vn_dim_var_list) > 0:
            vn_dim_var_tensor = torch.stack(vn_dim_var_list, dim=0)
            avg_dim_var = vn_dim_var_tensor.mean(dim=0)

            print("Average VN Variance per dim [x,y,z,h,w,l,yaw]:",
                  avg_dim_var.numpy())

        print("====================================")

        csv_path = os.path.join(opt.model_dir, "variance_statistics.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Frame", "MeanLogVar", "MeanVar", "MedianVar", "MaxVar",
                "MinVar", "StdVar", "X", "Y", "Z", "H", "W", "L", "Yaw"
            ])
            for s in vn_statistics:
                writer.writerow([
                    s["frame"], s["mean_logvar"], s["mean_var"],
                    s["median_var"], s["max_var"], s["min_var"],
                    s["std_var"], s["x"], s["y"], s["z"], s["h"],
                    s["w"], s["l"], s["yaw"]
                ])
        print(f"Variance statistics saved to {csv_path}")
    else:
        print("No VN uncertainty found. This is normal if you used baseline model.")
    # -------------------------------------------------------------

    # ---------------- Error vs Variance correlation report ----------------
    if len(prediction_errors) > 10:
        corr, p_value = pearsonr(prediction_errors, prediction_variances)
        print("====================================")
        print(f"Error vs Variance Pearson correlation: {corr:.4f} "
              f"(p={p_value:.4g})")
        print(f"Matched boxes used: {len(prediction_errors)}")
        print("====================================")

        err_var_csv = os.path.join(opt.model_dir, "error_vs_variance.csv")
        with open(err_var_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["error_m", "mean_variance"])
            for e, v in zip(prediction_errors, prediction_variances):
                writer.writerow([e, v])
        print(f"Per-box error/variance pairs saved to {err_var_csv}")
    else:
        print("Not enough matched boxes to compute Error-vs-Variance "
              "correlation (need VN model output and >10 matched boxes).")
    # -------------------------------------------------------------------

    if len(all_frame_variances) > 0:
        np.save(
            os.path.join(opt.model_dir, "all_variances.npy"),
            np.concatenate(all_frame_variances, axis=0)
        )

    if len(all_frame_logvars) > 0:
        np.save(
            os.path.join(opt.model_dir, "all_logvars.npy"),
            np.concatenate(all_frame_logvars, axis=0)
        )

    print("Saved raw variance arrays.")

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)

    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  "short")

    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  "middle")

    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  "long")

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
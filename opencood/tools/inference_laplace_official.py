import argparse
import os
import csv

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

from opencood.uncertainty.official_laplace_heads import (
    predict_with_head_laplace,
    make_opencood_output,
)


def parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_dir", required=True, type=str)
    p.add_argument("--laplace_file", required=True, type=str)

    p.add_argument(
        "--fusion_method",
        default="intermediate",
        choices=["intermediate"],
    )

    p.add_argument(
        "--param_scope",
        default="heads",
        choices=["heads", "full"],
    )

    # alpha is consumed by predict_with_head_laplace below.
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--prior_precision", type=float, default=1.0)
    p.add_argument("--T", type=int, default=10)

    p.add_argument("--isSim", action="store_true")
    p.add_argument("--save_uncertainty", action="store_true")

    return p.parse_args()


def compute_ece(confidences, corrects, n_bins=10):
    """Match the ECE convention used by inference_mc.py / inference_vn.py."""
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)

    n = len(confidences)
    if n == 0:
        return float("nan")

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

    return float(ece)


def classification_nll(confidences, corrects, eps=1e-8):
    p = np.clip(np.asarray(confidences, dtype=float), eps, 1.0 - eps)
    y = np.asarray(corrects, dtype=float)

    if len(p) == 0:
        return float("nan")

    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def nearest_gt_error(pred, gt):
    if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
        return None

    pred_centers = pred.mean(dim=1)[:, :3]
    gt_centers = gt.mean(dim=1)[:, :3]
    return torch.cdist(pred_centers, gt_centers).min(dim=1).values


def get_box_center_xy(box_corners):
    pts = np.asarray(box_corners)[:, :2]
    return pts.mean(axis=0)


def nearest_anchor_value(box_center_xy, dense_map, cav_lidar_range):
    xmin, ymin, _, xmax, ymax, _ = cav_lidar_range
    _, height, width = dense_map.shape
    x, y = box_center_xy

    col = int((x - xmin) / (xmax - xmin) * width)
    row = int((y - ymin) / (ymax - ymin) * height)

    col = min(max(col, 0), width - 1)
    row = min(max(row, 0), height - 1)

    return dense_map[:, row, col].mean().item()


def laplace_detection_regression_variance(
    result,
    pred_box_tensor,
    cav_lidar_range,
):
    """
    Convert dense encoded-space Laplace regression variance [B, 7*A, H, W]
    into one scalar per final detection, matching the MC-Dropout convention.
    """
    if pred_box_tensor is None or len(pred_box_tensor) == 0:
        return None

    var_rm = result["var_reg_encoded"].detach().squeeze(0)

    channels, height, width = var_rm.shape
    if channels % 7 != 0:
        raise ValueError(
            f"Laplace regression variance has {channels} channels; "
            "expected a multiple of 7."
        )

    n_anchor = channels // 7
    var_rm_map = var_rm.reshape(n_anchor, 7, height, width).mean(dim=1)

    pred_np = pred_box_tensor.detach().cpu().numpy()
    per_box_var = []

    for box in pred_np:
        center_xy = get_box_center_xy(box)
        per_box_var.append(
            nearest_anchor_value(center_xy, var_rm_map, cav_lidar_range)
        )

    return np.asarray(per_box_var, dtype=np.float64)


def compute_uce(errors, variances, n_bins=15):
    """
    Regression UCE using RMSE versus predicted standard deviation in each bin.
    Keep this same function across MC/VN/Laplace if reporting cross-method UCE.
    """
    errors = np.asarray(errors, dtype=float)
    variances = np.maximum(np.asarray(variances, dtype=float), 0.0)

    if len(errors) == 0:
        return float("nan")

    std = np.sqrt(variances + 1e-12)

    if np.allclose(std.min(), std.max()):
        return float(abs(np.sqrt(np.mean(errors ** 2)) - std.mean()))

    edges = np.linspace(std.min(), std.max(), n_bins + 1)
    uce = 0.0

    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if j == n_bins - 1:
            mask = (std >= lo) & (std <= hi)
        else:
            mask = (std >= lo) & (std < hi)

        if mask.any():
            rmse = np.sqrt(np.mean(errors[mask] ** 2))
            predicted_std = std[mask].mean()
            uce += mask.mean() * abs(rmse - predicted_std)

    return float(uce)


def compute_ause(errors, variances):
    e = np.asarray(errors, dtype=float)
    u = np.asarray(variances, dtype=float)
    n = len(e)

    if n < 2:
        return float("nan")

    # Fraction of predictions removed
    xs = np.linspace(0.0, 0.95, 20)

    # Predicted uncertainty ranking
    pred_order = np.argsort(-u)

    # Oracle ranking using actual error
    oracle_order = np.argsort(-e)

    pred_curve = []
    oracle_curve = []

    for x in xs:
        k = min(int(round(x * n)), n - 1)

        pred_curve.append(
            e[pred_order[k:]].mean()
        )

        oracle_curve.append(
            e[oracle_order[k:]].mean()
        )

    pred_curve = np.asarray(pred_curve)
    oracle_curve = np.asarray(oracle_curve)

    # np.trapz is used instead of np.trapezoid
    # for compatibility with your NumPy version.
    return float(
        np.trapz(
            pred_curve - oracle_curve,
            x=xs
        )
    )


def extract_lidar_points_for_vis(ego_data):
    for key in ("origin_lidar", "lidar_np", "projected_lidar"):
        if key not in ego_data:
            continue

        value = ego_data[key]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            value = value[0]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)
        else:
            value = np.asarray(value)

        # origin_lidar is commonly [B, N, C] after collation.
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]

        if value.ndim >= 2:
            return value

    return None


def main():
    opt = parser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hypes = yaml_utils.load_yaml(None, opt)
    cav_lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]

    print("Building validation/test dataset...")

    dataset = build_dataset(
        hypes,
        visualize=True,
        train=False,
        isSim=opt.isSim,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    model.to(device)

    epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"Loaded MAP checkpoint epoch {epoch}")
    model.eval()

    checkpoint = torch.load(opt.laplace_file, map_location=device)
    cls_la = checkpoint["cls_la"]
    reg_la = checkpoint["reg_la"]

    cls_la.to(device)
    reg_la.to(device)

    # Warn rather than silently claiming a scope based only on CLI text.
    saved_scope = checkpoint.get("param_scope")
    if saved_scope is not None and saved_scope != opt.param_scope:
        print(
            f"[WARNING] --param_scope={opt.param_scope}, but checkpoint "
            f"reports param_scope={saved_scope}. The checkpoint determines "
            "which parameters actually have a Laplace posterior."
        )

    print(
        f"Laplace settings: scope={saved_scope or opt.param_scope}, "
        f"alpha={opt.alpha}, prior_precision={opt.prior_precision}, T={opt.T}"
    )

    def new_stat():
        return {
            0.5: {"tp": [], "fp": [], "gt": 0},
            0.7: {"tp": [], "fp": [], "gt": 0},
        }

    result_stat = new_stat()
    result_stat_short = new_stat()
    result_stat_middle = new_stat()
    result_stat_long = new_stat()

    result_dir = os.path.join(
        opt.model_dir,
        "laplace_results",
        f"{opt.param_scope}_alpha_{opt.alpha}_T{opt.T}",
    )
    os.makedirs(result_dir, exist_ok=True)

    uncertainty_dir = os.path.join(result_dir, "uncertainty_npz")
    vis_dir = os.path.join(result_dir, "visualization_npz_laplace")

    if opt.save_uncertainty:
        os.makedirs(uncertainty_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    classification_confidences = []
    classification_correct = []
    prediction_errors = []
    prediction_variances = []

    total_var_cls = 0.0
    total_var_reg = 0.0
    uncertainty_frame_count = 0

    for i, batch_data in enumerate(loader):
        print(
            f"\rLaplace inference: {i + 1}/{len(loader)}",
            end="",
            flush=True,
        )

        batch_data = train_utils.to_device(batch_data, device)
        ego_data = batch_data["ego"]

        # IMPORTANT:
        # alpha must be used by the sampler itself. Merely recording alpha in
        # filenames/metadata does not scale the posterior.
        result = predict_with_head_laplace(
            model,
            cls_la,
            reg_la,
            ego_data,
            T=opt.T,
            alpha=opt.alpha,
        )

        output_dict = make_opencood_output(result)
        output_dict_all = {"ego": output_dict}

        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
            batch_data,
            output_dict_all,
        )

        # Raw predictive variance summary.
        total_var_cls += result["var_cls_prob"].mean().item()
        total_var_reg += result["var_reg_encoded"].mean().item()
        uncertainty_frame_count += 1

        # Classification calibration:
        # Preserve the same 2 m center-distance proxy used in MC/VN.
        if (
            pred_box_tensor is not None
            and pred_score is not None
            and len(pred_box_tensor) > 0
        ):
            scores = pred_score.detach().cpu().reshape(-1).numpy()

            if gt_box_tensor is not None and len(gt_box_tensor) > 0:
                errors = nearest_gt_error(pred_box_tensor, gt_box_tensor)
                correct = (errors < 2.0).to(torch.float32)

                classification_confidences.extend(scores.tolist())
                classification_correct.extend(
                    correct.detach().cpu().reshape(-1).tolist()
                )

                per_box_var = laplace_detection_regression_variance(
                    result,
                    pred_box_tensor,
                    cav_lidar_range,
                )

                errors_np = errors.detach().cpu().numpy().reshape(-1)
                matched = errors_np < 2.0

                if per_box_var is not None and matched.any():
                    prediction_errors.extend(errors_np[matched].tolist())
                    prediction_variances.extend(per_box_var[matched].tolist())
            else:
                # Match MC/VN: predictions in a frame with no GT are incorrect.
                classification_confidences.extend(scores.tolist())
                classification_correct.extend(
                    np.zeros(len(scores), dtype=np.float32).tolist()
                )

        # Overall AP.
        eval_utils.caluclate_tp_fp(
            pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
        )
        eval_utils.caluclate_tp_fp(
            pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
        )

        # 0--30 m.
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_short,
            0.5,
            left_range=0,
            right_range=30,
        )
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_short,
            0.7,
            left_range=0,
            right_range=30,
        )

        # 30--50 m.
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_middle,
            0.5,
            left_range=30,
            right_range=50,
        )
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_middle,
            0.7,
            left_range=30,
            right_range=50,
        )

        # 50--100 m.
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_long,
            0.5,
            left_range=50,
            right_range=100,
        )
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_long,
            0.7,
            left_range=50,
            right_range=100,
        )

        if opt.save_uncertainty:
            np.savez_compressed(
                os.path.join(uncertainty_dir, f"{i:05d}.npz"),
                mean_cls_prob=result["mean_cls_prob"].detach().cpu().numpy(),
                cls_variance=result["var_cls_prob"].detach().cpu().numpy(),
                classification_uncertainty=result[
                    "var_cls_prob"
                ].detach().cpu().numpy(),
                predictive_entropy=result[
                    "predictive_entropy"
                ].detach().cpu().numpy(),
                mutual_information=result[
                    "mutual_information"
                ].detach().cpu().numpy(),
                reg_variance=result[
                    "var_reg_encoded"
                ].detach().cpu().numpy(),
                regression_uncertainty=result[
                    "var_reg_encoded"
                ].detach().cpu().numpy(),
                alpha=np.asarray(opt.alpha, dtype=np.float32),
                prior_precision=np.asarray(
                    opt.prior_precision, dtype=np.float32
                ),
                T=np.asarray(opt.T, dtype=np.int32),
            )

            lidar_points = extract_lidar_points_for_vis(ego_data)

            vis_payload = {
                "method": np.asarray("Laplace"),
                "gt_boxes": (
                    gt_box_tensor.detach().cpu().numpy()
                    if gt_box_tensor is not None
                    else np.empty((0, 8, 3), dtype=np.float32)
                ),
                "pred_boxes": (
                    pred_box_tensor.detach().cpu().numpy()
                    if pred_box_tensor is not None
                    else np.empty((0, 8, 3), dtype=np.float32)
                ),
                "pred_scores": (
                    pred_score.detach().cpu().numpy()
                    if pred_score is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                "mean_cls_prob": result[
                    "mean_cls_prob"
                ].detach().cpu().numpy(),
                "classification_uncertainty": result[
                    "var_cls_prob"
                ].detach().cpu().numpy(),
                "cls_variance": result[
                    "var_cls_prob"
                ].detach().cpu().numpy(),
                "predictive_entropy": result[
                    "predictive_entropy"
                ].detach().cpu().numpy(),
                "mutual_information": result[
                    "mutual_information"
                ].detach().cpu().numpy(),
                "regression_uncertainty": result[
                    "var_reg_encoded"
                ].detach().cpu().numpy(),
                "reg_variance": result[
                    "var_reg_encoded"
                ].detach().cpu().numpy(),
                "alpha": np.asarray(opt.alpha, dtype=np.float32),
                "prior_precision": np.asarray(
                    opt.prior_precision, dtype=np.float32
                ),
                "T": np.asarray(opt.T, dtype=np.int32),
                "cav_lidar_range": np.asarray(
                    cav_lidar_range, dtype=np.float32
                ),
            }

            if lidar_points is not None:
                vis_payload["pcd"] = lidar_points

            np.savez_compressed(
                os.path.join(vis_dir, f"{i:05d}.npz"),
                **vis_payload,
            )

    print()

    if uncertainty_frame_count > 0:
        print(
            "Average Laplace classification variance (raw feature map):",
            total_var_cls / uncertainty_frame_count,
        )
        print(
            "Average Laplace regression variance (encoded feature map):",
            total_var_reg / uncertainty_frame_count,
        )

    # Standardized CSVs.
    cls_csv = os.path.join(
        result_dir,
        "classification_calibration_laplace.csv",
    )
    with open(cls_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence", "correct"])
        writer.writerows(
            zip(classification_confidences, classification_correct)
        )

    err_csv = os.path.join(
        result_dir,
        "error_vs_variance_laplace.csv",
    )
    with open(err_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_m", "mean_variance"])
        writer.writerows(zip(prediction_errors, prediction_variances))

    ece = compute_ece(
        classification_confidences,
        classification_correct,
        n_bins=10,
    )
    nll = classification_nll(
        classification_confidences,
        classification_correct,
    )
    uce = compute_uce(prediction_errors, prediction_variances)
    ause = compute_ause(prediction_errors, prediction_variances)

    if len(prediction_errors) > 10:
        pearson, pearson_p = pearsonr(
            prediction_errors,
            prediction_variances,
        )
        spearman, spearman_p = spearmanr(
            prediction_errors,
            prediction_variances,
        )
    else:
        pearson = pearson_p = spearman = spearman_p = float("nan")

    print("\n========== Laplace metrics ==========")
    print(f"ECE (10 bins): {ece:.6f}")
    print(f"Classification NLL: {nll:.6f}")
    print(f"UCE: {uce:.6f}")
    print(f"AUSE: {ause:.6f}")
    print(f"Pearson: {pearson:.6f} (p={pearson_p:.4g})")
    print(f"Spearman: {spearman:.6f} (p={spearman_p:.4g})")
    print(f"Matched regression boxes: {len(prediction_errors)}")
    print(f"Classification boxes: {len(classification_confidences)}")
    print(f"Metric outputs: {result_dir}")
    print("=====================================")

    with open(
        os.path.join(result_dir, "laplace_metrics.txt"),
        "w",
    ) as f:
        f.write(f"param_scope={saved_scope or opt.param_scope}\n")
        f.write(f"alpha={opt.alpha}\n")
        f.write(f"prior_precision_cli={opt.prior_precision}\n")
        f.write(f"T={opt.T}\n")
        f.write("ece_bins=10\n")
        f.write("classification_correct_definition=center_distance_lt_2m\n")
        f.write("regression_match_definition=center_distance_lt_2m\n")
        f.write("regression_variance_space=encoded_target_space\n")
        f.write(f"ECE={ece}\n")
        f.write(f"classification_NLL={nll}\n")
        f.write(f"UCE={uce}\n")
        f.write(f"AUSE={ause}\n")
        f.write(f"Pearson={pearson}\n")
        f.write(f"Pearson_p={pearson_p}\n")
        f.write(f"Spearman={spearman}\n")
        f.write(f"Spearman_p={spearman_p}\n")
        f.write(f"matched_regression_boxes={len(prediction_errors)}\n")
        f.write(
            f"classification_boxes={len(classification_confidences)}\n"
        )

    # Final AP metrics.
    eval_utils.eval_final_results(result_stat, result_dir)
    eval_utils.eval_final_results(
        result_stat_short,
        result_dir,
        "short",
    )
    eval_utils.eval_final_results(
        result_stat_middle,
        result_dir,
        "middle",
    )
    eval_utils.eval_final_results(
        result_stat_long,
        result_dir,
        "long",
    )


if __name__ == "__main__":
    main()
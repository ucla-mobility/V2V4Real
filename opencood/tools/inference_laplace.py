"""
Bayesian inference for OpenCOOD using a diagonal Laplace approximation.

Workflow
--------
1. Load trained MAP OpenCOOD detector.
2. Load previously computed diagonal curvature.
3. Construct the Laplace posterior.
4. Draw posterior weight samples.
5. Run the detector for each sampled parameter vector.
6. Aggregate:
       - classification probabilities
       - classification epistemic uncertainty
       - regression predictions
       - regression uncertainty
7. Feed posterior-mean predictions into the normal OpenCOOD
   post-processing/evaluation pipeline.
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.uncertainty.diagonal_laplace import DiagonalLaplace
from metrics.calibration_utils import PaperCalibrationAccumulator


from metrics.uncertainty_matching_utils import match_error_uncertainty_pairs

def parse_args():

    parser = argparse.ArgumentParser(
        description="OpenCOOD inference with diagonal Laplace uncertainty."
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Directory containing the trained OpenCOOD checkpoint.",
    )

    parser.add_argument(
        "--laplace_dir",
        default=None,
        type=str,
        help=(
            "Directory containing Laplace artifacts and receiving Laplace "
            "inference outputs. Default: <model_dir>_laplace"
        ),
    )

    parser.add_argument(
        "--laplace_path",
        default=None,
        type=str,
        help=(
            "Saved curvature file. "
            "Default: <laplace_dir>/laplace_diag.pt"
        ),
    )

    parser.add_argument(
        "--param_scope",
        default="heads",
        choices=["heads", "full"],
        help=(
            "'heads': Laplace approximation over cls_head/reg_head. "
            "'full': approximation over the complete detector."
        ),
    )

    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="Laplace posterior covariance scaling.",
    )

    parser.add_argument(
        "--prior_precision",
        default=1.0,
        type=float,
        help="Gaussian prior precision.",
    )

    parser.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Number of posterior Monte Carlo samples.",
    )

    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="cuda, cuda:0, cpu, etc.",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    return parser.parse_args()


def resolve_device(device_argument):

    if device_argument is not None:
        return torch.device(device_argument)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def move_to_device(data, device):

    if torch.is_tensor(data):
        return data.to(
            device,
            non_blocking=True,
        )

    if isinstance(data, dict):
        return {
            key: move_to_device(value, device)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            move_to_device(value, device)
            for value in data
        ]

    if isinstance(data, tuple):
        return tuple(
            move_to_device(value, device)
            for value in data
        )

    return data


# ==============================================================
# Classification uncertainty helpers
# ==============================================================

def binary_entropy(probability, eps=1e-8):

    p = probability.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return -(
        p * torch.log(p)
        +
        (1.0 - p)
        * torch.log(1.0 - p)
    )


def compute_classification_uncertainty(
    probability_samples,
):

    # probability_samples:
    #
    # [T, B, anchors, H, W]
    #
    # T = posterior samples

    mean_probability = probability_samples.mean(
        dim=0
    )

    probability_variance = probability_samples.var(
        dim=0,
        unbiased=False,
    )

    predictive_entropy = binary_entropy(
        mean_probability
    )

    expected_entropy = binary_entropy(
        probability_samples
    ).mean(
        dim=0
    )

    # Epistemic classification uncertainty.
    mutual_information = (
        predictive_entropy
        - expected_entropy
    ).clamp_min(
        0.0
    )

    return {
        "mean_probability":
            mean_probability,

        "variance":
            probability_variance,

        "predictive_entropy":
            predictive_entropy,

        "expected_entropy":
            expected_entropy,

        "mutual_information":
            mutual_information,
    }


# ==============================================================
# Regression uncertainty
# ==============================================================

def compute_regression_uncertainty(
    regression_samples,
):

    mean_regression = regression_samples.mean(
        dim=0
    )

    regression_variance = regression_samples.var(
        dim=0,
        unbiased=False,
    )

    regression_std = torch.sqrt(
        regression_variance.clamp_min(0.0)
    )

    return {
        "mean":
            mean_regression,

        "variance":
            regression_variance,

        "std":
            regression_std,
    }


# ==============================================================
# Probability -> logit
# ==============================================================

def probability_to_logit(
    probability,
    eps=1e-6,
):

    p = probability.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return torch.log(
        p / (1.0 - p)
    )


# ==============================================================
# Laplace posterior inference
# ==============================================================

@torch.no_grad()
def laplace_forward(
    laplace_model,
    ego_data,
    num_samples,
):

    if num_samples < 1:
        raise ValueError(
            "num_samples must be >= 1."
        )

    classification_samples = []
    regression_samples = []

    try:

        for sample_idx in range(
            num_samples
        ):

            # --------------------------------------------------
            # Draw theta ~ q(theta | D)
            # --------------------------------------------------

            laplace_model.sample_weights()

            # --------------------------------------------------
            # Detector forward pass
            # --------------------------------------------------

            output_dict = laplace_model.model(
                ego_data
            )

            classification_logits = (
                output_dict["psm"]
            )

            regression_output = (
                output_dict["rm"]
            )

            # --------------------------------------------------
            # Convert logits to probability before averaging.
            #
            # IMPORTANT:
            #
            #     E[sigmoid(z)] != sigmoid(E[z])
            #
            # so Bayesian classification should average the
            # probabilities from individual posterior samples.
            # --------------------------------------------------

            classification_probability = (
                torch.sigmoid(
                    classification_logits
                )
            )

            classification_samples.append(
                classification_probability.detach()
            )

            regression_samples.append(
                regression_output.detach()
            )

    finally:

        # Critical:
        #
        # Never leave a posterior sample installed in the detector.
        laplace_model.restore_map_weights()

    classification_samples = torch.stack(
        classification_samples,
        dim=0,
    )

    regression_samples = torch.stack(
        regression_samples,
        dim=0,
    )

    classification = (
        compute_classification_uncertainty(
            classification_samples
        )
    )

    regression = (
        compute_regression_uncertainty(
            regression_samples
        )
    )

    return {
        "classification":
            classification,

        "regression":
            regression,

        "classification_samples":
            classification_samples,

        "regression_samples":
            regression_samples,
    }


# ==============================================================
# Build OpenCOOD-compatible posterior-mean output
# ==============================================================

def build_bayesian_output(
    laplace_result,
):

    mean_probability = (
        laplace_result[
            "classification"
        ][
            "mean_probability"
        ]
    )

    mean_regression = (
        laplace_result[
            "regression"
        ][
            "mean"
        ]
    )

    # OpenCOOD postprocessor expects psm to contain logits.
    #
    # We have the posterior predictive probability:
    #
    #     p_bar = E_q[sigmoid(z_theta)]
    #
    # Therefore construct an equivalent logit:
    #
    #     logit(p_bar)
    #
    # such that:
    #
    #     sigmoid(logit(p_bar)) = p_bar

    bayesian_psm = probability_to_logit(
        mean_probability
    )

    return {
        "psm":
            bayesian_psm,

        "rm":
            mean_regression,
    }


# ==============================================================
# Evaluation helpers (kept aligned with MC/VN protocol)
# ==============================================================


def box_centers(box_tensor):
    """Center of OpenCOOD corner boxes [N,8,3]."""
    return box_tensor.mean(dim=1).detach().cpu().numpy()


def nearest_anchor_value(box_center_xy, dense_map, cav_lidar_range):
    """Same nearest-grid lookup convention used by MC-Dropout."""
    xmin, ymin, _, xmax, ymax, _ = cav_lidar_range
    _, h, w = dense_map.shape
    x, y = box_center_xy
    col = int((x - xmin) / (xmax - xmin) * w)
    row = int((y - ymin) / (ymax - ymin) * h)
    col = min(max(col, 0), w - 1)
    row = min(max(row, 0), h - 1)
    return dense_map[:, row, col].mean().item()


def per_box_regression_variance(pred_box_tensor, reg_variance, cav_lidar_range):
    """Reduce dense encoded regression variance to one scalar per final box."""
    if pred_box_tensor is None or len(pred_box_tensor) == 0:
        return None
    # reg_variance: [B, 7*anchor_num, H, W]
    b, c, h, w = reg_variance.shape
    if b != 1 or c % 7 != 0:
        raise ValueError(f"Unexpected regression variance shape: {tuple(reg_variance.shape)}")
    n_anchor = c // 7
    dense = reg_variance.squeeze(0).view(n_anchor, 7, h, w).mean(dim=1)
    values = []
    for box in pred_box_tensor:
        center_xy = box.detach().cpu().numpy()[:, :2].mean(axis=0)
        values.append(nearest_anchor_value(center_xy, dense, cav_lidar_range))
    return torch.tensor(values, device=reg_variance.device).unsqueeze(-1)


# ==============================================================
# Main
# ==============================================================

def main():

    args = parse_args()

    # The original training directory is only used to load config/MAP weights.
    # Laplace artifacts live in a separate sibling directory.
    if args.laplace_dir is None:
        args.laplace_dir = args.model_dir.rstrip(os.sep) + "_laplace"

    args.laplace_dir = os.path.abspath(args.laplace_dir)
    os.makedirs(args.laplace_dir, exist_ok=True)

    # Create standard locations now so later evaluation/visualization code can
    # write there without polluting the original training directory.
    for subdir in ("results", "uncertainty", "npy"):
        os.makedirs(os.path.join(args.laplace_dir, subdir), exist_ok=True)

    print(f"[Laplace] Source model directory: {os.path.abspath(args.model_dir)}")
    print(f"[Laplace] Laplace directory:      {args.laplace_dir}")

    device = resolve_device(
        args.device
    )

    print(
        f"\n[Laplace] Device: {device}"
    )

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    hypes = load_yaml(
        None,
        args
    )

    # ----------------------------------------------------------
    # Test dataset
    # ----------------------------------------------------------

    print(
        "\n[Laplace] Building evaluation dataset..."
    )

    test_dataset = build_dataset(
        hypes,
        visualize=True,
        train=False,
    )

    if hasattr(
        test_dataset,
        "collate_batch_test",
    ):

        collate_fn = (
            test_dataset.collate_batch_test
        )

    else:

        collate_fn = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # ----------------------------------------------------------
    # MAP detector
    # ----------------------------------------------------------

    print(
        "\n[Laplace] Loading MAP detector..."
    )

    model = train_utils.create_model(
        hypes
    )

    model = model.to(
        device
    )

    _, model = train_utils.load_saved_model(
        args.model_dir,
        model,
    )

    model = model.to(
        device
    )

    model.eval()

    # ----------------------------------------------------------
    # Parameter scope
    # ----------------------------------------------------------

    if args.param_scope == "heads":

        param_scope = (
            "cls_head",
            "reg_head",
        )

    else:

        param_scope = "full"

    # ----------------------------------------------------------
    # Laplace object
    # ----------------------------------------------------------

    laplace_model = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device,
    )

    # ----------------------------------------------------------
    # Load curvature
    # ----------------------------------------------------------

    if args.laplace_path is None:

        laplace_path = os.path.join(
            args.laplace_dir,
            "laplace_diag.pt",
        )

    else:

        laplace_path = (
            args.laplace_path
        )

    print(
        "\n[Laplace] Loading curvature:"
    )

    print(
        laplace_path
    )

    laplace_model.load(
        laplace_path
    )

    # ----------------------------------------------------------
    # Construct posterior
    # ----------------------------------------------------------

    laplace_model.set_posterior(
        alpha=args.alpha,
        prior_precision=args.prior_precision,
    )

    print(
        "\n[Laplace] Posterior ready."
    )

    print(
        f"[Laplace] MC samples: "
        f"{args.num_samples}"
    )

    # ----------------------------------------------------------
    # Evaluation state
    # ----------------------------------------------------------

    cav_lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
    result_stat = {0.5: {"tp": [], "fp": [], "gt": 0},
                   0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_short = {0.5: {"tp": [], "fp": [], "gt": 0},
                         0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_middle = {0.5: {"tp": [], "fp": [], "gt": 0},
                          0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_long = {0.5: {"tp": [], "fp": [], "gt": 0},
                        0.3: {"tp": [], "fp": [], "gt": 0}}

    prediction_errors = []
    prediction_variances = []
    # Feng et al. paper-style dense classification + marginal regression ECE.
    paper_calibration = PaperCalibrationAccumulator(n_cls_bins=10, n_reg_levels=10)
    visualization_npz_dir = os.path.join(args.laplace_dir, "visualization_npz_laplace")
    os.makedirs(visualization_npz_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    for batch_idx, batch_data in enumerate(
        test_loader
    ):

        batch_data = move_to_device(
            batch_data,
            device,
        )

        ego_data = batch_data[
            "ego"
        ]

        # ------------------------------------------------------
        # Bayesian forward pass
        # ------------------------------------------------------

        result = laplace_forward(
            laplace_model,
            ego_data,
            args.num_samples,
        )

        # ------------------------------------------------------
        # Posterior predictive mean in normal OpenCOOD format
        # ------------------------------------------------------

        bayesian_output = (
            build_bayesian_output(
                result
            )
        )

        # ------------------------------------------------------
        # Uncertainty tensors
        # ------------------------------------------------------

        cls_variance = result[
            "classification"
        ][
            "variance"
        ]

        predictive_entropy = result[
            "classification"
        ][
            "predictive_entropy"
        ]

        mutual_information = result[
            "classification"
        ][
            "mutual_information"
        ]

        reg_variance = result[
            "regression"
        ][
            "variance"
        ]

        # ------------------------------------------------------
        # Paper-style calibration: dense classification labels and
        # positive-anchor encoded regression targets. No 2 m matching.
        # ------------------------------------------------------
        paper_calibration.update(
            cls_prob=result["classification"]["mean_probability"],
            reg_mean=result["regression"]["mean"],
            reg_var=reg_variance,
            batch_data=batch_data,
        )

        # ------------------------------------------------------
        # Standard OpenCOOD post-processing of posterior mean
        # ------------------------------------------------------
        output_dict_mean = {"ego": bayesian_output}
        pred_box_tensor, pred_score, gt_box_tensor = test_dataset.post_process(
            batch_data, output_dict_mean
        )

        # ------------------------------------------------------
        # Per-final-box regression uncertainty (same MC protocol)
        # ------------------------------------------------------
        pred_var_tensor = per_box_regression_variance(
            pred_box_tensor, reg_variance, cav_lidar_range
        )

        # ------------------------------------------------------
        # Regression uncertainty-quality pairs.
        # Shared one-to-one AP-style BEV-IoU matching (IoU >= 0.5).
        # ------------------------------------------------------
        match_result = match_error_uncertainty_pairs(
            pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor,
            iou_threshold=0.5)
        prediction_errors.extend(match_result["errors_m"].tolist())
        prediction_variances.extend(match_result["uncertainties"].tolist())

        # ------------------------------------------------------
        # Standardized per-frame NPZ for plot_heatmap.py
        # ------------------------------------------------------
        pcd = batch_data["ego"].get("origin_lidar", [np.empty((0, 4), dtype=np.float32)])[0]
        if torch.is_tensor(pcd):
            pcd = pcd.detach().cpu().numpy()
        pred_np = (pred_box_tensor.detach().cpu().numpy()
                   if pred_box_tensor is not None else np.empty((0, 8, 3), dtype=np.float32))
        gt_np = (gt_box_tensor.detach().cpu().numpy()
                 if gt_box_tensor is not None else np.empty((0, 8, 3), dtype=np.float32))
        score_np = (pred_score.detach().cpu().numpy()
                    if pred_score is not None else np.empty((0,), dtype=np.float32))
        np.savez_compressed(
            os.path.join(visualization_npz_dir, f"{batch_idx:05d}.npz"),
            method=np.asarray("Diagonal Laplace"),
            pcd=np.asarray(pcd),
            gt_boxes=gt_np,
            pred_boxes=pred_np,
            pred_scores=score_np,
            classification_uncertainty=mutual_information.detach().cpu().numpy(),
            mutual_information=mutual_information.detach().cpu().numpy(),
            predictive_entropy=predictive_entropy.detach().cpu().numpy(),
            cls_variance=cls_variance.detach().cpu().numpy(),
            regression_uncertainty=reg_variance.detach().cpu().numpy(),
            reg_variance=reg_variance.detach().cpu().numpy(),
            cav_lidar_range=np.asarray(cav_lidar_range, dtype=np.float32),
            num_samples=np.asarray(args.num_samples, dtype=np.int32),
        )

        # ------------------------------------------------------
        # AP evaluation. Use the baseline/VN dictionary convention here:
        # IoU 0.3 and 0.5, including the same distance ranges.
        # ------------------------------------------------------
        for thr in (0.5, 0.3):
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_short, thr,
                left_range=0, right_range=30
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_middle, thr,
                left_range=30, right_range=50
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_long, thr,
                left_range=50, right_range=100
            )

        if batch_idx % 50 == 0:
            print(
                f"[Laplace] Processed {batch_idx + 1}/{len(test_loader)}"
            )

    # ----------------------------------------------------------
    # Save the same metric CSV protocol as MC/VN
    # ----------------------------------------------------------
    err_var_csv = os.path.join(args.laplace_dir, "error_vs_variance_laplace.csv")
    with open(err_var_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_m", "mean_variance"])
        writer.writerows(zip(prediction_errors, prediction_variances))
    print(f"[Laplace] Saved {len(prediction_errors)} matched error/variance pairs: {err_var_csv}")

    # Canonical paper-style classification + regression ECE.
    paper_calibration.save(args.laplace_dir, "diagonal_laplace")

    # Standard OpenCOOD AP summaries, kept inside laplace_dir.
    eval_utils.eval_final_results(result_stat, args.laplace_dir)
    eval_utils.eval_final_results(result_stat_short, args.laplace_dir, "short")
    eval_utils.eval_final_results(result_stat_middle, args.laplace_dir, "middle")
    eval_utils.eval_final_results(result_stat_long, args.laplace_dir, "long")

    print(
        "\n[Laplace] Inference complete."
    )


if __name__ == "__main__":
    main()
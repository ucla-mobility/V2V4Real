import argparse
import os

import numpy as np
import torch
import csv
from scipy.stats import pearsonr, spearmanr

from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.tools import train_utils

from opencood.data_utils.datasets import (
    build_dataset
)

from opencood.utils import eval_utils

from opencood.uncertainty.diagonal_laplace import (
    DiagonalLaplace
)


def parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_dir",
        required=True,
        type=str
    )

    p.add_argument(
        "--laplace_file",
        required=True,
        type=str
    )

    p.add_argument(
        "--fusion_method",
        default="intermediate",
        choices=[
            "intermediate"
        ]
    )

    p.add_argument(
        "--param_scope",
        default="heads",
        choices=[
            "heads",
            "full"
        ]
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.001
    )

    p.add_argument(
        "--prior_precision",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--T",
        type=int,
        default=10
    )

    p.add_argument(
        "--isSim",
        action="store_true"
    )

    p.add_argument(
        "--save_uncertainty",
        action="store_true"
    )

    return p.parse_args()



def compute_ece(conf, correct, n_bins=15):
    conf=np.asarray(conf,float); correct=np.asarray(correct,float)
    if len(conf)==0: return float("nan")
    edges=np.linspace(0,1,n_bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): ece += m.mean()*abs(correct[m].mean()-conf[m].mean())
    return float(ece)

def classification_nll(conf, correct, eps=1e-8):
    p=np.clip(np.asarray(conf,float),eps,1-eps); y=np.asarray(correct,float)
    if len(p)==0: return float("nan")
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))

def nearest_gt_error(pred, gt):
    if pred is None or gt is None or len(pred)==0 or len(gt)==0: return None
    pc=pred.mean(dim=1)[:,:3]; gc=gt.mean(dim=1)[:,:3]
    return torch.cdist(pc,gc).min(dim=1).values

def get_box_center_xy(box_corners):
    pts = np.asarray(box_corners)[:, :2]
    return pts.mean(axis=0)


def nearest_anchor_value(box_center_xy, dense_map, cav_lidar_range):
    """
    dense_map: [anchor_num, H, W].
    Look up the dense uncertainty at the BEV grid cell nearest the final
    detected box center, then average over anchor rotations.

    This is evaluation-only and does NOT change Laplace inference/AP.
    """
    xmin, ymin, _, xmax, ymax, _ = cav_lidar_range
    _, H, W = dense_map.shape
    x, y = box_center_xy

    col = int((x - xmin) / (xmax - xmin) * W)
    row = int((y - ymin) / (ymax - ymin) * H)

    col = min(max(col, 0), W - 1)
    row = min(max(row, 0), H - 1)

    return dense_map[:, row, col].mean().item()


def laplace_detection_regression_variance(result, pred_box_tensor,
                                           cav_lidar_range):
    """
    Convert native dense Laplace regression variance into one uncertainty
    scalar per final detected box.

    result["var_reg_encoded"] has shape [B, 7*A, H, W].
    We reshape it to [A, 7, H, W], average the 7 encoded regression
    dimensions, and sample the map at each final box center.

    This matches the nearest-anchor convention used by the MC-Dropout
    evaluation, rather than the previous incorrect top-k MI proxy.
    """
    if pred_box_tensor is None or len(pred_box_tensor) == 0:
        return None

    var_rm = result["var_reg_encoded"].detach()
    var_rm = var_rm.squeeze(0)  # [7*A, H, W]

    channels, H, W = var_rm.shape
    if channels % 7 != 0:
        raise ValueError(
            f"Laplace regression variance has {channels} channels; "
            "expected a multiple of 7."
        )

    n_anchor = channels // 7
    var_rm_map = var_rm.view(n_anchor, 7, H, W).mean(dim=1)

    per_box_var = []
    pred_np = pred_box_tensor.detach().cpu().numpy()

    for box in pred_np:
        center_xy = get_box_center_xy(box)
        per_box_var.append(
            nearest_anchor_value(center_xy, var_rm_map, cav_lidar_range)
        )

    return np.asarray(per_box_var, dtype=np.float64)

def compute_uce(errors, variances, n_bins=15):
    e=np.asarray(errors,float); v=np.maximum(np.asarray(variances,float),0)
    if len(e)==0: return float("nan")
    sd=np.sqrt(v+1e-12)
    edges=np.linspace(sd.min(),sd.max()+1e-12,n_bins+1); ans=0.
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m=(sd>=lo)&((sd<=hi) if j==n_bins-1 else (sd<hi))
        if m.any(): ans += m.mean()*abs(np.sqrt(np.mean(e[m]**2))-sd[m].mean())
    return float(ans)

def compute_ause(errors, variances):
    e=np.asarray(errors,float); u=np.asarray(variances,float); n=len(e)
    if n<2: return float("nan")
    xs=np.linspace(0,.95,20); po=np.argsort(-u); oo=np.argsort(-e)
    pc=[]; oc=[]
    for x in xs:
        k=min(int(round(x*n)),n-1)
        pc.append(e[po[k:]].mean()); oc.append(e[oo[k:]].mean())
    return float(np.trapezoid(np.asarray(pc)-np.asarray(oc),xs))



def extract_lidar_points_for_vis(ego_data):
    """
    Best-effort extraction of raw LiDAR points from common OpenCOOD batch keys.
    Returns numpy array or None. This is visualization-only.
    """
    candidates = [
        "origin_lidar",
        "lidar_np",
        "projected_lidar",
    ]
    for key in candidates:
        if key in ego_data:
            x = ego_data[key]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            elif isinstance(x, (list, tuple)) and len(x) > 0:
                x = x[0]
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                else:
                    x = np.asarray(x)
            else:
                x = np.asarray(x)
            if x.ndim >= 2:
                return x
    return None

def main():

    opt = parser()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------
    # Load config
    # ------------------------------------------------

    hypes = yaml_utils.load_yaml(
        None,
        opt
    )

    # Needed only to map final box centers back to the dense uncertainty grid.
    cav_lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    print("Building validation/test dataset...")

    dataset = build_dataset(
        hypes,
        visualize=True,
        train=False,
        isSim=opt.isSim
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=(
            dataset.collate_batch_test
        ),
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    # ------------------------------------------------
    # MAP model
    # ------------------------------------------------

    model = train_utils.create_model(
        hypes
    )

    model.to(device)

    epoch, model = (
        train_utils.load_saved_model(
            opt.model_dir,
            model
        )
    )

    print(
        f"Loaded MAP checkpoint epoch {epoch}"
    )

    model.eval()

    # ------------------------------------------------
    # Laplace
    # ------------------------------------------------

    if opt.param_scope == "heads":

        param_scope = (
            "cls_head",
            "reg_head"
        )

    else:

        param_scope = "full"

    laplace = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device
    )

    laplace.load(
        opt.laplace_file
    )

    laplace.set_posterior(
        alpha=opt.alpha,
        prior_precision=opt.prior_precision
    )

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------

    result_stat = {
        0.5: {
            "tp": [],
            "fp": [],
            "gt": 0
        },
        0.7: {
            "tp": [],
            "fp": [],
            "gt": 0
        }
    }

    result_stat_short = {
        0.5: {"tp": [], "fp": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "gt": 0}
    }

    result_stat_middle = {
        0.5: {"tp": [], "fp": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "gt": 0}
    }

    result_stat_long = {
        0.5: {"tp": [], "fp": [], "gt": 0},
        0.7: {"tp": [], "fp": [], "gt": 0}
    }

    uncertainty_dir = os.path.join(
        opt.model_dir,
        "laplace_uncertainty",
        f"alpha_{opt.alpha}"
    )

    if opt.save_uncertainty:

        os.makedirs(
            uncertainty_dir,
            exist_ok=True
        )

    result_dir = os.path.join(
        opt.model_dir, "laplace_results",
        f"{opt.param_scope}_alpha_{opt.alpha}_T{opt.T}"
    )
    os.makedirs(result_dir, exist_ok=True)

    vis_dir = os.path.join(result_dir, "visualization_npz")
    if opt.save_uncertainty:
        os.makedirs(vis_dir, exist_ok=True)

    classification_confidences = []
    classification_correct = []
    prediction_errors = []
    prediction_variances = []

    # ------------------------------------------------
    # Main inference loop
    # ------------------------------------------------

    for i, batch_data in enumerate(loader):

        print(
            f"\rLaplace inference: "
            f"{i + 1}/{len(loader)}",
            end=""
        )

        batch_data = train_utils.to_device(
            batch_data,
            device
        )

        ego_data = batch_data["ego"]

        # --------------------------------------------
        # Bayesian MC prediction
        # --------------------------------------------

        result = laplace.predict_raw(
            ego_data,
            T=opt.T
        )

        # --------------------------------------------
        # Convert Bayesian mean back into normal
        # OpenCOOD output format
        # --------------------------------------------

        output_dict = (
            laplace.make_opencood_output(
                result
            )
        )

        # ------------------------------------------------
        # IMPORTANT:
        #
        # Use the SAME dataset postprocessor as normal
        # intermediate-fusion inference.
        #
        # For intermediate fusion, model output normally
        # corresponds to ego.
        # ------------------------------------------------

        output_dict_all = {
            "ego": output_dict
        }

        pred_box_tensor, pred_score, gt_box_tensor = (
            dataset.post_process(
                batch_data,
                output_dict_all
            )
        )

        # ------------------------------------------------
        # Metrics only -- Laplace prediction/aggregation above is unchanged.
        # ------------------------------------------------
        if pred_box_tensor is not None and pred_score is not None \
                and len(pred_box_tensor) > 0:

            # Classification calibration: keep the existing definition.
            errors = nearest_gt_error(pred_box_tensor, gt_box_tensor)
            if errors is not None:
                correct = (errors < 2.0).long()

                classification_confidences.extend(
                    pred_score.detach().cpu().reshape(-1).tolist()
                )
                classification_correct.extend(
                    correct.detach().cpu().reshape(-1).tolist()
                )

                # Regression uncertainty:
                # pair EACH matched final detection with the Laplace
                # regression variance at that detection's BEV location.
                #
                # IMPORTANT: unlike the old code, this does NOT sort/top-k
                # mutual information and pretend it corresponds to boxes.
                per_box_var = laplace_detection_regression_variance(
                    result,
                    pred_box_tensor,
                    cav_lidar_range
                )

                errors_np = (
                    errors.detach().cpu().numpy().reshape(-1)
                )
                matched = errors_np < 2.0

                if per_box_var is not None and matched.any():
                    prediction_errors.extend(
                        errors_np[matched].tolist()
                    )
                    prediction_variances.extend(
                        per_box_var[matched].tolist()
                    )

        # ------------------------------------------------
        # Normal AP evaluation
        # ------------------------------------------------

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            0.5
        )

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            0.7
        )

        # short: 0-30 m

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_short,
            0.5,
            left_range=0,
            right_range=30
        )

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_short,
            0.7,
            left_range=0,
            right_range=30
        )

        # middle: 30-50 m

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_middle,
            0.5,
            left_range=30,
            right_range=50
        )

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_middle,
            0.7,
            left_range=30,
            right_range=50
        )

        # long: 50-100 m

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_long,
            0.5,
            left_range=50,
            right_range=100
        )

        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat_long,
            0.7,
            left_range=50,
            right_range=100
        )

        # ------------------------------------------------
        # Save uncertainty maps
        # ------------------------------------------------

        if opt.save_uncertainty:

            np.savez_compressed(
                os.path.join(
                    uncertainty_dir,
                    f"{i:05d}.npz"
                ),

                mean_cls_prob=(
                    result[
                        "mean_cls_prob"
                    ]
                    .cpu()
                    .numpy()
                ),

                cls_variance=(
                    result[
                        "var_cls_prob"
                    ]
                    .cpu()
                    .numpy()
                ),

                predictive_entropy=(
                    result[
                        "predictive_entropy"
                    ]
                    .cpu()
                    .numpy()
                ),

                mutual_information=(
                    result[
                        "mutual_information"
                    ]
                    .cpu()
                    .numpy()
                ),

                reg_variance=(
                    result[
                        "var_reg_encoded"
                    ]
                    .cpu()
                    .numpy()
                )
            )


            # ------------------------------------------------
            # Visualization package.
            # Does NOT alter Laplace inference or metrics.
            # ------------------------------------------------
            lidar_points = extract_lidar_points_for_vis(ego_data)

            vis_payload = {
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
                "gt_boxes": (
                    gt_box_tensor.detach().cpu().numpy()
                    if gt_box_tensor is not None
                    else np.empty((0, 8, 3), dtype=np.float32)
                ),
                "mean_cls_prob": result["mean_cls_prob"].detach().cpu().numpy(),
                "cls_variance": result["var_cls_prob"].detach().cpu().numpy(),
                "predictive_entropy": result["predictive_entropy"].detach().cpu().numpy(),
                "mutual_information": result["mutual_information"].detach().cpu().numpy(),
                "reg_variance": result["var_reg_encoded"].detach().cpu().numpy(),
                "alpha": np.asarray(opt.alpha, dtype=np.float32),
                "prior_precision": np.asarray(opt.prior_precision, dtype=np.float32),
                "T": np.asarray(opt.T, dtype=np.int32),
            }

            if lidar_points is not None:
                vis_payload["pcd"] = lidar_points

            np.savez_compressed(
                os.path.join(vis_dir, f"{i:05d}.npz"),
                **vis_payload
            )

    print()

    # ------------------------------------------------
    # Added metrics (evaluation only)
    # ------------------------------------------------
    cls_csv = os.path.join(result_dir, "classification_calibration_laplace.csv")
    with open(cls_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["confidence","correct"])
        w.writerows(zip(classification_confidences, classification_correct))

    err_csv = os.path.join(result_dir, "error_vs_variance_laplace.csv")
    with open(err_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["error_m","mean_variance"])
        w.writerows(zip(prediction_errors, prediction_variances))

    ece=compute_ece(classification_confidences, classification_correct)
    nll=classification_nll(classification_confidences, classification_correct)
    uce=compute_uce(prediction_errors, prediction_variances)
    ause=compute_ause(prediction_errors, prediction_variances)

    if len(prediction_errors)>10:
        pr,pp=pearsonr(prediction_errors,prediction_variances)
        sr,sp=spearmanr(prediction_errors,prediction_variances)
    else:
        pr=pp=sr=sp=float("nan")

    print("\n========== Laplace metrics ==========")
    print(f"ECE: {ece:.6f}")
    print(f"Classification NLL: {nll:.6f}")
    print(f"UCE: {uce:.6f}")
    print(f"AUSE: {ause:.6f}")
    print(f"Pearson: {pr:.6f} (p={pp:.4g})")
    print(f"Spearman: {sr:.6f} (p={sp:.4g})")
    print(f"Metric outputs: {result_dir}")
    print("=====================================")

    with open(os.path.join(result_dir,"laplace_metrics.txt"),"w") as f:
        f.write(f"alpha={opt.alpha}\nprior_precision={opt.prior_precision}\nT={opt.T}\n")
        f.write(f"ECE={ece}\nclassification_NLL={nll}\nUCE={uce}\nAUSE={ause}\n")
        f.write(f"Pearson={pr}\nPearson_p={pp}\nSpearman={sr}\nSpearman_p={sp}\n")

    # ------------------------------------------------
    # Final AP
    # ------------------------------------------------

    eval_utils.eval_final_results(
        result_stat,
        result_dir
    )

    eval_utils.eval_final_results(
        result_stat_short,
        result_dir,
        "short"
    )

    eval_utils.eval_final_results(
        result_stat_middle,
        result_dir,
        "middle"
    )

    eval_utils.eval_final_results(
        result_stat_long,
        result_dir,
        "long"
    )


if __name__ == "__main__":

    main()
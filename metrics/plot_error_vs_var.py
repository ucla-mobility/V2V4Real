"""
Error vs predicted uncertainty for several methods (error-based calibration).

Reads the error_vs_variance_*.csv files written by inference_vn.py /
inference_mc.py / inference_laplace.py (columns error_m, mean_variance), where
  error_m        = 3-D distance between the matched predicted and true box
                   centre (m), one-to-one matches with BEV IoU >= 0.5;
  mean_variance  = predicted box-centre variance in m^2,
                   d^2 (var_x + var_y) + h^2 var_z
                   (metrics.uncertainty_matching_utils.centre_variance_m2).

If the predicted variance is correct, E[error^2] = variance. So, following
Levi et al., "Evaluating and Calibrating Uncertainty Prediction in Regression
Tasks" (2019), the matches are split into equal-count bins by predicted
variance and, per bin,
  RMV  = sqrt(mean predicted variance)   (m)  — what the model says
  RMSE = sqrt(mean squared error)        (m)  — what really happens
are compared. Perfect calibration: RMSE = RMV (the diagonal).
Below the diagonal = overconfident, above = underconfident.
  ENCE = mean_bins |RMV - RMSE| / RMV  (lower is better).

Usage (repeat --csv/--label, same order):
    python metrics/plot_error_vs_var.py \
        --csv a/error_vs_variance_laplace.csv --label "Laplace" \
        --csv b/error_vs_variance_vn.csv      --label "VN-ES" \
        --out_dir results/error_vs_variance

Outputs in --out_dir:
    error_vs_variance_reliability_combined_attfuse.png   RMV vs RMSE, all methods
    error_vs_variance_scatter_combined.png       one density panel per method
    error_vs_variance_summary.csv                n, Pearson, Spearman, ENCE,
                                                 overall RMV / RMSE
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr


def reliability_bins(errors, variances, num_bins):
    """Equal-count bins by predicted variance -> per-bin RMV, RMSE, count."""
    order = np.argsort(variances, kind="stable")
    rmv, rmse, counts = [], [], []
    for idx in np.array_split(order, num_bins):
        if len(idx) == 0:
            continue
        rmv.append(np.sqrt(variances[idx].mean()))
        rmse.append(np.sqrt((errors[idx] ** 2).mean()))
        counts.append(len(idx))
    return np.array(rmv), np.array(rmse), np.array(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Error vs predicted uncertainty (RMV vs RMSE) for several methods.")
    parser.add_argument("--csv", action="append", required=True,
                        help="error_vs_variance_*.csv (repeat with --label)")
    parser.add_argument("--label", action="append", required=True,
                        help="Label for each --csv, same order")
    parser.add_argument("--num_bins", type=int, default=10,
                        help="Equal-count bins per method")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory for the PNGs and the summary CSV")
    opt = parser.parse_args()

    if len(opt.csv) != len(opt.label):
        raise ValueError("Number of --csv and --label arguments must match")
    os.makedirs(opt.out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    runs = []
    for csv_path, label in zip(opt.csv, opt.label):
        if not os.path.exists(csv_path):
            print(f"[skip] File not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["error_m", "mean_variance"])
        errors = df["error_m"].to_numpy(dtype=np.float64)
        variances = df["mean_variance"].to_numpy(dtype=np.float64)
        rmv, rmse, counts = reliability_bins(errors, variances, opt.num_bins)
        runs.append({
            "label": label, "errors": errors, "variances": variances,
            "rmv": rmv, "rmse": rmse, "counts": counts,
            "pearson": pearsonr(errors, variances)[0],
            "spearman": spearmanr(errors, variances)[0],
            "ence": float(np.mean(np.abs(rmv - rmse) / rmv)),
            "rmv_all": float(np.sqrt(variances.mean())),
            "rmse_all": float(np.sqrt((errors ** 2).mean())),
        })

    # ---- 1. reliability diagram: RMV vs RMSE, all methods ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    lo = min(min(r["rmv"].min(), r["rmse"].min()) for r in runs) * 0.8
    hi = max(max(r["rmv"].max(), r["rmse"].max()) for r in runs) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="perfect calibration (RMSE = RMV)")
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="0.9", zorder=0)
    ax.text(0.03, 0.95, "overconfident\n(error larger than predicted)",
            transform=ax.transAxes, va="top", fontsize=8, color="0.35")
    for k, r in enumerate(runs):
        ax.plot(r["rmv"], r["rmse"], "o-", color=colors[k % len(colors)], ms=5,
                label=f"{r['label']} (ENCE={r['ence']:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted uncertainty per bin: RMV = sqrt(mean variance) (m)")
    ax.set_ylabel("Real error per bin: RMSE (m)")
    ax.set_title(f"Error vs predicted uncertainty ({opt.num_bins} equal-count bins)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(opt.out_dir, "error_vs_variance_reliability_combined_attfuse.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- 2. scatter: one density panel per method ----
    ncols = min(3, len(runs))
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows),
                            sharex=True, sharey=True, squeeze=False,
                            constrained_layout=True)
    std_all = np.concatenate([np.sqrt(r["variances"]) for r in runs])
    err_all = np.concatenate([r["errors"] for r in runs])
    x_lim = (np.percentile(std_all, 0.5) * 0.8, np.percentile(std_all, 99.5) * 1.2)
    y_lim = (max(np.percentile(err_all, 0.5), 1e-3) * 0.8, np.percentile(err_all, 99.9) * 1.2)
    for k, (ax, r) in enumerate(zip(axs.ravel(), runs)):
        std = np.sqrt(r["variances"])
        err = np.maximum(r["errors"], y_lim[0])
        hb = ax.hexbin(std, err, xscale="log", yscale="log", gridsize=40,
                       extent=(np.log10(x_lim[0]), np.log10(x_lim[1]),
                               np.log10(y_lim[0]), np.log10(y_lim[1])),
                       cmap="Greys", norm=LogNorm(), mincnt=1)
        ax.plot(x_lim, x_lim, "k--", lw=1.0, label="error = predicted std")
        ax.plot(r["rmv"], r["rmse"], "o-", color=colors[k % len(colors)], ms=4,
                lw=1.8, label="binned RMSE vs RMV")
        ax.set_title(f"{r['label']}\nn={len(err):,}  Pearson {r['pearson']:.2f}  "
                     f"Spearman {r['spearman']:.2f}  ENCE {r['ence']:.2f}", fontsize=8)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.2, which="both")
    for ax in axs.ravel()[len(runs):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("Predicted centre std (m)", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("Real centre error (m)", fontsize=8)
    axs[0, 0].legend(fontsize=7, loc="upper left")
    fig.colorbar(hb, ax=axs, fraction=0.02, pad=0.01, label="boxes per cell")
    fig.suptitle("Real error vs predicted uncertainty per box (log-log; grey = density)",
                 fontsize=11)
    path = os.path.join(opt.out_dir, "error_vs_variance_scatter_combined.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- 3. summary ----
    summary = pd.DataFrame([{
        "method": r["label"], "n": len(r["errors"]),
        "pearson_r": round(r["pearson"], 4), "spearman_r": round(r["spearman"], 4),
        "ence": round(r["ence"], 4),
        "rmv_all_m": round(r["rmv_all"], 4), "rmse_all_m": round(r["rmse_all"], 4),
        "rmse_over_rmv": round(r["rmse_all"] / r["rmv_all"], 3),
    } for r in runs]).sort_values("ence")
    path = os.path.join(opt.out_dir, "error_vs_variance_summary.csv")
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

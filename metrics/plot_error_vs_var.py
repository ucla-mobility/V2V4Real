"""
Plot Error vs Variance calibration results, overlaying multiple runs (e.g.
MC-Dropout, NLL-VN, Energy-Score-VN) on the SAME scatter plot and the SAME
reliability diagram, instead of one pair of PNGs per run.

Reads error_vs_variance_mc.csv / error_vs_variance_vn.csv (produced by
inference_mc.py / inference_vn.py -- separate filenames per script so they
don't overwrite each other in the same --model_dir).

Usage
-----
Single run (still works, same as before):
    python plot_error_vs_var.py \
        --csv opencood/logs/<run_folder>/error_vs_variance_vn.csv --label "NLL-VN"

Multiple runs on one pair of combined plots (repeat --csv/--label together,
same order, one pair per run):
    python plot_error_vs_var.py \
        --csv mc_run/error_vs_variance_mc.csv  --label "MC-Dropout" \
        --csv nll_run/error_vs_variance_vn.csv --label "NLL-VN" \
        --csv es_run/error_vs_variance_vn.csv  --label "Energy Score-VN" \
        --out_dir combined_plots

Output
------
- Prints Pearson correlation + per-bin table for each run to the console.
- Saves two combined PNGs to --out_dir (default: current directory):
    error_vs_variance_scatter_combined.png
    error_vs_variance_reliability_combined.png    (the one worth putting in your thesis)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def compute_reliability_bins(errors, variances, num_bins):
    """
    Equal-count bins (quantiles) so every bin has a similar sample size,
    which is more robust than equal-width bins on skewed variance data.
    """
    bin_edges = np.quantile(variances, np.linspace(0, 1, num_bins + 1))
    bin_edges[-1] += 1e-9  # make sure the max value falls into the last bin
    bin_idx = np.digitize(variances, bin_edges[1:-1])

    bin_mean_var, bin_mean_err, bin_std_err, bin_counts = [], [], [], []
    for b in range(num_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_mean_var.append(variances[mask].mean())
        bin_mean_err.append(errors[mask].mean())
        bin_std_err.append(errors[mask].std())
        bin_counts.append(mask.sum())

    return bin_mean_var, bin_mean_err, bin_std_err, bin_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                        help="Path to error_vs_variance_mc.csv or "
                             "error_vs_variance_vn.csv (repeat --csv/--label "
                             "together, same order, for multiple runs)")
    parser.add_argument("--label", action="append", required=True,
                        help="Label for each --csv, in the same order "
                             "(e.g. 'MC-Dropout', 'NLL-VN', 'Energy Score-VN')")
    parser.add_argument("--num_bins", type=int, default=10,
                        help="Number of variance bins for the reliability diagram")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory to save the two combined PNGs into")
    opt = parser.parse_args()

    if len(opt.csv) != len(opt.label):
        raise ValueError("Number of --csv and --label arguments must match")

    os.makedirs(opt.out_dir, exist_ok=True)

    # a distinct color per run, cycling if there are more than 10
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
    fig_reliab, ax_reliab = plt.subplots(figsize=(7, 5))

    for i, (csv_path, label) in enumerate(zip(opt.csv, opt.label)):
        if not os.path.exists(csv_path):
            print(f"[skip] File not found: {csv_path}")
            continue

        color = colors[i % len(colors)]
        df = pd.read_csv(csv_path)
        errors = df["error_m"].to_numpy()
        variances = df["mean_variance"].to_numpy()

        corr, p_value = pearsonr(errors, variances)
        s_corr, s_p_value = spearmanr(errors, variances)
        print("=" * 60)
        print(f"Run: {label}  (n = {len(df)})")
        print(f"  Pearson correlation:  {corr:.4f} (p={p_value:.4g})")
        print(f"  Spearman correlation: {s_corr:.4f} (p={s_p_value:.4g})")

        # --------------------------------------------------------------
        # 1. Raw scatter, one series per run, overlaid on shared axes
        # --------------------------------------------------------------
        ax_scatter.scatter(variances, errors, s=4, alpha=0.15, color=color,
                           label=f"{label} (r_p={corr:.3f}, r_s={s_corr:.3f}, n={len(df)})")

        # --------------------------------------------------------------
        # 2. Reliability diagram, one line per run, overlaid on shared axes
        # --------------------------------------------------------------
        bin_mean_var, bin_mean_err, bin_std_err, bin_counts = \
            compute_reliability_bins(errors, variances, opt.num_bins)

        ax_reliab.errorbar(bin_mean_var, bin_mean_err, yerr=bin_std_err,
                           fmt="o-", color=color, ecolor=color, alpha=0.8,
                           capsize=3, markersize=6, label=label)

        print("  Per-bin (mean_variance, mean_error_m, count):")
        for v, e, c in zip(bin_mean_var, bin_mean_err, bin_counts):
            print(f"    var={v:10.4f}  err={e:8.4f}  n={c}")

    print("=" * 60)

    ax_scatter.set_xlabel("Predicted variance (mean over 7 box dims)")
    ax_scatter.set_ylabel("Prediction error (m, distance to matched GT box)")
    ax_scatter.set_title("Error vs Variance -- all runs")
    ax_scatter.legend(markerscale=4)
    fig_scatter.tight_layout()
    scatter_path = os.path.join(opt.out_dir, "error_vs_variance_scatter_combined.png")
    fig_scatter.savefig(scatter_path, dpi=200)
    plt.close(fig_scatter)
    print(f"Saved: {scatter_path}")

    ax_reliab.set_xlabel("Mean predicted variance (binned)")
    ax_reliab.set_ylabel("Mean prediction error (m)")
    ax_reliab.set_title(f"Reliability diagram: error vs predicted uncertainty\n"
                        f"({opt.num_bins} bins per run)")
    ax_reliab.legend()
    fig_reliab.tight_layout()
    reliability_path = os.path.join(opt.out_dir, "error_vs_variance_reliability_combined.png")
    fig_reliab.savefig(reliability_path, dpi=200)
    plt.close(fig_reliab)
    print(f"Saved: {reliability_path}")


if __name__ == "__main__":
    main()
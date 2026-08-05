"""
Plot Error vs Variance calibration results from error_vs_variance.csv
(produced by inference_vn.py).

Usage:
    python plot_error_vs_variance.py --csv opencood/logs/<run_folder>/error_vs_variance.csv

Produces two PNGs in the same folder as the CSV:
    error_vs_variance_scatter_nll_sd.png   -- raw scatter, log-scale x axis
    error_vs_variance_reliability_nll_sd.png -- binned reliability diagram
                                         (the one worth putting in your thesis)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to error_vs_variance.csv")
    parser.add_argument("--num_bins", type=int, default=10,
                        help="Number of variance bins for the reliability diagram")
    opt = parser.parse_args()

    out_dir = os.path.dirname(opt.csv)
    df = pd.read_csv(opt.csv)

    errors = df["error_m"].values
    variances = df["mean_variance"].values

    corr, p_value = pearsonr(errors, variances)
    print(f"Pearson correlation: {corr:.4f} (p={p_value:.4g})")
    print(f"N = {len(df)} matched boxes")

    # ------------------------------------------------------------------
    # 1. Raw scatter plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(variances, errors, s=4, alpha=0.15, color="steelblue")
    plt.xlabel("Predicted variance (mean over 7 box dims)")
    plt.ylabel("Prediction error (m, distance to matched GT box)")
    plt.title(f"Error vs Variance (Pearson r = {corr:.3f}, n = {len(df)})")
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "error_vs_variance_scatter_nll_dd.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"Saved: {scatter_path}")

    # ------------------------------------------------------------------
    # 2. Reliability diagram: bin by variance, plot mean (+ std) error
    #    per bin. This is the plot to actually put in your thesis --
    #    a rising staircase means the model's confidence is calibrated.
    # ------------------------------------------------------------------
    # equal-count bins (quantiles) so every bin has a similar sample size,
    # which is more robust than equal-width bins on skewed variance data
    bin_edges = np.quantile(variances, np.linspace(0, 1, opt.num_bins + 1))
    bin_edges[-1] += 1e-9  # make sure the max value falls into the last bin
    bin_idx = np.digitize(variances, bin_edges[1:-1])

    bin_mean_var = []
    bin_mean_err = []
    bin_std_err = []
    bin_counts = []

    for b in range(opt.num_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_mean_var.append(variances[mask].mean())
        bin_mean_err.append(errors[mask].mean())
        bin_std_err.append(errors[mask].std())
        bin_counts.append(mask.sum())

    plt.figure(figsize=(6, 5))
    plt.errorbar(bin_mean_var, bin_mean_err, yerr=bin_std_err,
                fmt="o-", color="darkorange", ecolor="lightgray",
                capsize=3, markersize=6)
    plt.xlabel("Mean predicted variance (binned)")
    plt.ylabel("Mean prediction error (m)")
    plt.title(f"Reliability diagram: error vs predicted uncertainty\n"
             f"(Pearson r = {corr:.3f}, {opt.num_bins} bins, n = {len(df)})")
    plt.tight_layout()
    reliability_path = os.path.join(out_dir, "error_vs_variance_reliability_nll_dd.png")
    plt.savefig(reliability_path, dpi=200)
    plt.close()
    print(f"Saved: {reliability_path}")

    # ------------------------------------------------------------------
    # Print the bin table too, useful to quote directly in your thesis text
    # ------------------------------------------------------------------
    print("\nBin | Mean Variance | Mean Error (m) | Count")
    print("-" * 50)
    for v, e, c in zip(bin_mean_var, bin_mean_err, bin_counts):
        print(f"{v:12.4f} | {e:14.4f} | {c}")


if __name__ == "__main__":
    main()
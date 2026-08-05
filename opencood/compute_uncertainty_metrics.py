"""
compute_uncertainty_metrics.py

Post-processing script: computes AUSE (Area Under the Sparsification Error
curve) and UCE (Uncertainty Calibration Error) from the
`error_vs_variance.csv` file that inference_vn.py already saves.

No model, no GPU, no re-running inference needed -- just reads the CSV.

Usage
-----
Single model:
    python compute_uncertainty_metrics.py --csv path/to/error_vs_variance.csv --label "NLL-VN"

Compare multiple models on one plot (e.g. NLL vs Energy Score):
    python compute_uncertainty_metrics.py \
        --csv nll_run/error_vs_variance.csv --label "NLL-VN" \
        --csv es_run/error_vs_variance.csv  --label "Energy Score-VN"

Output
------
- Prints AUSE and UCE for each CSV to the console.
- Saves a sparsification-error plot (predicted-uncertainty curve vs. oracle
  curve) to sparsification_plot.png in the current directory.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_ause(errors: np.ndarray, variances: np.ndarray, n_steps: int = 100):
    """
    Sparsification-based AUSE.

    We progressively remove the most uncertain (or, for the oracle curve,
    the highest-error) fraction of predictions and track the mean error of
    what remains. A well-calibrated uncertainty estimate should sparsify
    almost as well as the oracle (which sorts by true error, the best
    possible ordering). AUSE is the area between the two curves -- lower is
    better, 0 is perfect.

    Returns
    -------
    ause : float
    fracs : np.ndarray            fraction removed, from 0 to just under 1
    pred_curve : np.ndarray       mean remaining error, sorted by predicted variance
    oracle_curve : np.ndarray     mean remaining error, sorted by true error (best case)
    """
    n = len(errors)
    order_pred = np.argsort(-variances)     # most uncertain first
    order_oracle = np.argsort(-errors)      # highest error first (best possible ordering)

    errors_pred_sorted = errors[order_pred]
    errors_oracle_sorted = errors[order_oracle]

    fracs = np.linspace(0.0, 0.99, n_steps)  # stop just short of removing everything
    pred_curve = np.zeros(n_steps)
    oracle_curve = np.zeros(n_steps)

    for i, f in enumerate(fracs):
        keep_from = int(np.floor(f * n))
        pred_curve[i] = errors_pred_sorted[keep_from:].mean()
        oracle_curve[i] = errors_oracle_sorted[keep_from:].mean()

    ause = np.trapz(pred_curve - oracle_curve, fracs)
    return ause, fracs, pred_curve, oracle_curve


def compute_uce(errors: np.ndarray, variances: np.ndarray, n_bins: int = 10):
    """
    Uncertainty Calibration Error.

    Bins predictions by predicted std, and for each bin compares the mean
    predicted std against the RMSE of the actual errors in that bin.
    UCE = sum over bins of (n_bin / N) * |RMSE_bin - mean_std_bin|
    Lower is better, 0 is perfectly calibrated.
    """
    std = np.sqrt(np.maximum(variances, 0))
    n = len(errors)

    bin_edges = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # make sure the max value falls in the last bin
    bin_idx = np.digitize(std, bin_edges[1:-1], right=True)

    uce = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        rmse_b = np.sqrt(np.mean(errors[mask] ** 2))
        mean_std_b = std[mask].mean()
        uce += (n_b / n) * abs(rmse_b - mean_std_b)
        bin_stats.append((n_b, mean_std_b, rmse_b))

    return uce, bin_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                        help="Path to error_vs_variance.csv (repeat --csv/--label for multiple models)")
    parser.add_argument("--label", action="append", required=True,
                        help="Label for each --csv, in the same order")
    parser.add_argument("--out", type=str, default="sparsification_plot.png",
                        help="Where to save the sparsification plot")
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match")

    plt.figure(figsize=(7, 5))
    oracle_plotted = False

    for csv_path, label in zip(args.csv, args.label):
        if not os.path.exists(csv_path):
            print(f"[skip] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        errors = df["error_m"].to_numpy()
        variances = df["mean_variance"].to_numpy()

        ause, fracs, pred_curve, oracle_curve = compute_ause(errors, variances)
        uce, bin_stats = compute_uce(errors, variances)

        print("=" * 60)
        print(f"Model: {label}  (n = {len(df)})")
        print(f"  AUSE = {ause:.5f}   (lower is better, 0 = matches oracle)")
        print(f"  UCE  = {uce:.5f}   (lower is better, 0 = perfectly calibrated)")
        print("  Per-bin (n, mean_predicted_std, rmse):")
        for n_b, mean_std_b, rmse_b in bin_stats:
            print(f"    n={n_b:5d}  mean_std={mean_std_b:.4f}  rmse={rmse_b:.4f}")

        plt.plot(fracs, pred_curve, marker="o", markersize=3, label=f"{label} (predicted)")
        if not oracle_plotted:
            # oracle differs slightly per dataset but is usually close enough
            # to show once; plot it for the first model only to keep the
            # legend readable, or comment this out to plot per-model oracles
            plt.plot(fracs, oracle_curve, linestyle="--", color="black", label="Oracle (best possible)")
            oracle_plotted = True

    plt.xlabel("Fraction of predictions removed (most uncertain first)")
    plt.ylabel("Mean remaining error (m)")
    plt.title("Sparsification plot: predicted uncertainty vs. oracle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("=" * 60)
    print(f"Saved sparsification plot to {args.out}")


if __name__ == "__main__":
    main()
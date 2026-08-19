"""
compute_classification_calibration.py

Post-processing script: computes ECE (Expected Calibration Error) for the
classification head (psm confidence vs. actual detection correctness) from
the `classification_calibration_mc.csv` / `classification_calibration_vn.csv`
files that inference_mc.py / inference_vn.py already save.

No model, no GPU, no re-running inference needed -- just reads the CSV.

"Correct" is defined the same way as the Error-vs-Variance matching
elsewhere in this pipeline: a predicted box counts as a true positive if
its center is within match_threshold (2.0m) of some GT box's center. This
is a coarse center-distance proxy, not the exact IoU-based TP/FP used for
AP in eval_utils -- good enough for a calibration sanity check, but say so
if you quote this number next to the AP metrics.

Usage
-----
Single run:
    python compute_classification_calibration.py \
        --csv mc_run/classification_calibration_mc.csv --label "MC-Dropout"

Compare multiple runs on one reliability diagram:
    python compute_classification_calibration.py \
        --csv mc_run/classification_calibration_mc.csv  --label "MC-Dropout" \
        --csv nll_run/classification_calibration_vn.csv --label "VN-NLL" \
        --csv es_run/classification_calibration_vn.csv  --label "VN-ES" \
        --out_dir combined_plots

Output
------
- Prints ECE for each CSV to the console.
- Saves one combined reliability diagram (accuracy vs. confidence, with the
  y=x perfect-calibration line) to --out_dir.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_ece(confidences, corrects, n_bins=10):
    """
    Standard (Guo et al. 2017) Expected Calibration Error: equal-width bins
    over confidence in [0, 1]. For each bin, compares mean predicted
    confidence against the actual fraction correct (accuracy) in that bin.

    ECE = sum over bins of (n_bin / N) * |accuracy_bin - mean_confidence_bin|
    Lower is better, 0 is perfectly calibrated.

    Returns
    -------
    ece : float
    bin_stats : list of (bin_lo, bin_hi, n_bin, mean_confidence, accuracy)
    """
    confidences = np.asarray(confidences)
    corrects = np.asarray(corrects)
    n = len(confidences)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # include the right edge only in the last bin, so confidence==1.0
        # lands somewhere
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
        bin_stats.append((lo, hi, n_b, conf_b, acc_b))

    return ece, bin_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                        help="Path to classification_calibration_mc.csv or "
                             "classification_calibration_vn.csv (repeat "
                             "--csv/--label together, same order, for "
                             "multiple runs)")
    parser.add_argument("--label", action="append", required=True,
                        help="Label for each --csv, in the same order")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of equal-width confidence bins")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory to save the reliability diagram into")
    parser.add_argument("--out", type=str, default="classification_reliability.png",
                        help="Output PNG filename (inside --out_dir)")
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match")

    os.makedirs(args.out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black",
           label="Perfect calibration")

    for i, (csv_path, label) in enumerate(zip(args.csv, args.label)):
        if not os.path.exists(csv_path):
            print(f"[skip] File not found: {csv_path}")
            continue

        color = colors[i % len(colors)]
        df = pd.read_csv(csv_path)
        confidences = df["confidence"].to_numpy()
        corrects = df["correct"].to_numpy()

        ece, bin_stats = compute_ece(confidences, corrects, n_bins=args.n_bins)

        print("=" * 60)
        print(f"Run: {label}  (n = {len(df)})")
        print(f"  ECE = {ece:.5f}   (lower is better, 0 = perfectly calibrated)")
        print("  Per-bin (range, n, mean_confidence, accuracy):")
        for lo, hi, n_b, conf_b, acc_b in bin_stats:
            print(f"    [{lo:.2f}, {hi:.2f})  n={n_b:5d}  "
                  f"conf={conf_b:.4f}  acc={acc_b:.4f}")

        bin_conf = [s[3] for s in bin_stats]
        bin_acc = [s[4] for s in bin_stats]
        bin_n = [s[2] for s in bin_stats]
        ax.plot(bin_conf, bin_acc, marker="o", markersize=6, color=color,
               label=f"{label} (ECE={ece:.4f}, n={len(df)})")

    print("=" * 60)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (binned)")
    ax.set_ylabel("Accuracy (fraction of true positives)")
    ax.set_title(f"Classification calibration ({args.n_bins} bins per run)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, args.out)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
plot_sparsification.py

Plot sparsification curves and report AUSE using the EXACT SAME compute_ause()
function for MC-Dropout, VN-NLL, VN-Energy and Laplace.

Important:
- Each method is evaluated against its OWN oracle because the matched prediction
  set can differ between methods.
- Do not draw one method's oracle and pretend it applies to all methods.
- AUSE values are therefore the primary cross-method scalar comparison.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from metrics.compute_uncertainty_metrics import evaluate_csv
except ModuleNotFoundError:
    # Allows: python metrics/plot_sparsification.py ... from project root.
    from compute_uncertainty_metrics import evaluate_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", action="append", required=True,
        help="CSV with columns error_m,mean_variance. Repeat once per method."
    )
    parser.add_argument(
        "--label", action="append", required=True,
        help="Label corresponding to each --csv."
    )
    parser.add_argument(
        "--out", default="sparsification_plot.png",
        help="Output figure path."
    )
    parser.add_argument(
        "--summary-csv", default="sparsification_ause_summary.csv",
        help="Output CSV containing method, sample count and AUSE."
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--show-oracles", action="store_true",
        help="Also draw each method's own oracle curve. Off by default for readability."
    )
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match.")

    results = [
        evaluate_csv(path, label, n_steps=args.steps)
        for path, label in zip(args.csv, args.label)
    ]

    plt.figure(figsize=(8, 5.5))

    for result in results:
        plt.plot(
            result["fractions"],
            result["predicted_curve"],
            linewidth=2,
            label=f"{result['label']} (AUSE={result['ause']:.4f})",
        )
        if args.show_oracles:
            plt.plot(
                result["fractions"],
                result["oracle_curve"],
                linestyle="--",
                linewidth=1,
                alpha=0.55,
                label=f"{result['label']} oracle",
            )

    plt.xlabel("Fraction of matched predictions removed")
    plt.ylabel("Mean remaining localization error (m)")
    plt.title("Sparsification by predicted regression uncertainty")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    summary = pd.DataFrame(
        [{"method": r["label"], "n_matched": r["n"],
          "pearson_r": r["pearson_r"], "spearman_r": r["spearman_r"],
          "ause": r["ause"]}
         for r in results]
    ).sort_values("ause", ascending=True)

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved sparsification plot: {out_path}")
    print(f"Saved AUSE summary:       {summary_path}")
    print(
        "\nNote: each method has its own oracle because its matched prediction set "
        "may differ. AUSE is computed with the same rule for every method."
    )


if __name__ == "__main__":
    main()
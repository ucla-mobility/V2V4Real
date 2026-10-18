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
import numpy as np
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
        help="Also draw each method's own oracle curve (dashed, same colour). "
             "Off by default for readability."
    )
    parser.add_argument(
        "--error-plot", action="store_true",
        help="Also save <out>_error.png: sparsification error (method minus its "
             "own oracle) for all methods; the oracle is the common zero line and "
             "the area under each curve is its AUSE (Ilg et al., ECCV 2018)."
    )
    parser.add_argument(
        "--panels", action="store_true",
        help="Also save <out>_panels.png: one panel per method with its oracle "
             "and the area between them (= AUSE) shaded."
    )
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match.")

    results = [
        evaluate_csv(path, label, n_steps=args.steps)
        for path, label in zip(args.csv, args.label)
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8, 5.5))

    for k, result in enumerate(results):
        color = colors[k % len(colors)]
        plt.plot(
            result["fractions"],
            result["predicted_curve"],
            linewidth=2,
            color=color,
            label=f"{result['label']} (AUSE={result['ause']:.4f})",
        )
        if args.show_oracles:
            plt.plot(
                result["fractions"],
                result["oracle_curve"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                color=color,
            )
    if args.show_oracles:
        plt.plot([], [], "k--", linewidth=1.2,
                 label="dashed = oracle of the same colour")

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

    if args.error_plot:
        plt.figure(figsize=(8, 5.5))
        for k, result in enumerate(results):
            gap = np.maximum(result["predicted_curve"] - result["oracle_curve"], 0.0)
            plt.plot(result["fractions"], gap, linewidth=2,
                     color=colors[k % len(colors)],
                     label=f"{result['label']} (AUSE={result['ause']:.4f})")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.5,
                    label="oracle (perfect uncertainty)")
        plt.xlabel("Fraction of matched predictions removed (most uncertain first)")
        plt.ylabel("Sparsification error: method − own oracle (m)")
        plt.title("Sparsification error (area under each curve = AUSE; lower is better)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        error_path = out_path.with_name(out_path.stem + "_error" + out_path.suffix)
        plt.savefig(error_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved sparsification error: {error_path}")

    if args.panels:
        ncols = min(3, len(results))
        nrows = int(np.ceil(len(results) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                                sharex=True, sharey=True, squeeze=False,
                                constrained_layout=True)
        for k, (ax, result) in enumerate(zip(axs.ravel(), results)):
            color = colors[k % len(colors)]
            x = result["fractions"]
            ax.plot(x, result["predicted_curve"], color=color, linewidth=2,
                    label="method")
            ax.plot(x, result["oracle_curve"], "k--", linewidth=1.2,
                    label="oracle")
            ax.fill_between(x, result["oracle_curve"], result["predicted_curve"],
                            where=result["predicted_curve"] >= result["oracle_curve"],
                            color=color, alpha=0.2, label="gap (area = AUSE)")
            ax.set_title(f"{result['label']}\nAUSE = {result['ause']:.3f}, "
                         f"n = {result['n']:,}", fontsize=9)
            ax.grid(alpha=0.25)
        for ax in axs.ravel()[len(results):]:
            ax.axis("off")
        for ax in axs[-1]:
            ax.set_xlabel("Fraction removed (most uncertain first)", fontsize=8)
        for ax in axs[:, 0]:
            ax.set_ylabel("Mean remaining error (m)", fontsize=8)
        axs[0, 0].legend(fontsize=7)
        fig.suptitle("Sparsification: each method vs its own oracle", fontsize=11)
        panels_path = out_path.with_name(out_path.stem + "_panels" + out_path.suffix)
        fig.savefig(panels_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved per-method panels:   {panels_path}")

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
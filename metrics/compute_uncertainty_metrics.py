"""
compute_uncertainty_metrics.py

Common uncertainty-ranking metrics for MC-Dropout, Variance Network (NLL /
Energy Score), and Laplace.

This script intentionally DOES NOT compute UCE. Calibration is handled by the
paper-style classification/regression ECE pipeline.

Expected CSV columns
--------------------
error_m, mean_variance

Each inference method should save the same two-column format:
- MC-Dropout: error_vs_variance_mc.csv
- VN-NLL / VN-Energy: error_vs_variance_vn.csv (use separate run directories)
- Laplace: error_vs_variance_laplace.csv

AUSE definition
---------------
For every method:
1. Sort matched predictions by decreasing predicted mean variance.
2. Progressively remove the most uncertain fraction.
3. Compute mean localization error (metres) among predictions that remain.
4. Construct an oracle by sorting the SAME errors by decreasing true error.
5. AUSE is the trapezoidal area between the method and its own oracle curve.

Lower AUSE is better; 0 means the uncertainty ranking matches the oracle.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REQUIRED_COLUMNS = ("error_m", "mean_variance")


def load_error_variance_csv(csv_path):
    """Load and validate one method's error/variance pairs."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {missing}. "
            f"Required columns are {list(REQUIRED_COLUMNS)}."
        )

    clean = df.loc[:, REQUIRED_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
    errors = clean["error_m"].to_numpy(dtype=np.float64)
    variances = clean["mean_variance"].to_numpy(dtype=np.float64)

    if len(errors) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 valid matched predictions.")

    return errors, variances


def compute_ause(errors, variances, n_steps=100):
    """
    Compute sparsification curves and AUSE identically for every method.

    Parameters
    ----------
    errors : array-like
        Localization error in metres for matched predictions.
    variances : array-like
        Scalar predicted uncertainty used only for ranking.
    n_steps : int
        Number of common removal fractions.

    Returns
    -------
    ause, fractions, predicted_curve, oracle_curve
    """
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    variances = np.asarray(variances, dtype=np.float64).reshape(-1)

    if errors.shape != variances.shape:
        raise ValueError("errors and variances must have the same length.")
    if len(errors) < 2:
        raise ValueError("At least two predictions are required.")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(variances)):
        raise ValueError("errors and variances must contain only finite values.")

    # Identical rule for MC, VN-NLL, VN-Energy and Laplace.
    predicted_order = np.argsort(-variances, kind="stable")
    oracle_order = np.argsort(-errors, kind="stable")

    predicted_errors = errors[predicted_order]
    oracle_errors = errors[oracle_order]

    # Common x-axis for all methods. Stop before 1.0 so at least one item remains.
    fractions = np.linspace(0.0, 0.99, n_steps, dtype=np.float64)
    predicted_curve = np.empty_like(fractions)
    oracle_curve = np.empty_like(fractions)

    n = len(errors)
    for i, fraction in enumerate(fractions):
        remove_count = min(int(np.floor(fraction * n)), n - 1)
        predicted_curve[i] = predicted_errors[remove_count:].mean()
        oracle_curve[i] = oracle_errors[remove_count:].mean()

    # Numerical noise can make a point microscopically below the oracle.
    gap = np.maximum(predicted_curve - oracle_curve, 0.0)
    ause = float(np.trapz(gap, fractions))

    return ause, fractions, predicted_curve, oracle_curve


def evaluate_csv(csv_path, label, n_steps=100):
    errors, variances = load_error_variance_csv(csv_path)
    ause, fractions, predicted_curve, oracle_curve = compute_ause(
        errors, variances, n_steps=n_steps
    )
    pearson_r, pearson_p = pearsonr(errors, variances)
    spearman_r, spearman_p = spearmanr(errors, variances)
    return {
        "label": label,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "csv": str(csv_path),
        "n": len(errors),
        "ause": ause,
        "fractions": fractions,
        "predicted_curve": predicted_curve,
        "oracle_curve": oracle_curve,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute AUSE from error-vs-variance CSV files. UCE is intentionally removed."
    )
    parser.add_argument(
        "--csv", action="append", required=True,
        help="CSV containing error_m and mean_variance. Repeat for multiple methods."
    )
    parser.add_argument(
        "--label", action="append", required=True,
        help="Method label corresponding to each --csv."
    )
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("Number of --csv and --label arguments must match.")

    print("AUSE — common sparsification protocol")
    print("UCE is not computed; use paper-style ECE for calibration.")
    print("=" * 72)

    for csv_path, label in zip(args.csv, args.label):
        result = evaluate_csv(csv_path, label, n_steps=args.steps)
        print(
            f"{result['label']:<24} "
            f"n={result['n']:<7d} "
            f"Pearson={result['pearson_r']:+.4f}  "
            f"Spearman={result['spearman_r']:+.4f}  "
            f"AUSE={result['ause']:.6f}"
        )


if __name__ == "__main__":
    main()
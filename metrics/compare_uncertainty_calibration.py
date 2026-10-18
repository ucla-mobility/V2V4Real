import argparse
import csv
import json

def main():
    p = argparse.ArgumentParser(
        description="Compare paper-style classification and regression ECE summaries."
    )
    p.add_argument("--summary", action="append", required=True,
                   help="paper_ece_summary_*.json; repeat for each method")
    p.add_argument("--out_csv", default="paper_ece_comparison.csv")
    args = p.parse_args()

    rows = []
    for path in args.summary:
        with open(path) as f:
            s = json.load(f)
        r = {
            "method": s["method"],
            "classification_ece": s["classification_ece"],
            "mean_regression_ece": s["mean_regression_ece"],
        }
        for name, value in s["regression_ece_per_dimension"].items():
            r[f"reg_ece_{name}"] = value
        rows.append(r)

    fields = ["method", "classification_ece", "mean_regression_ece",
              "reg_ece_x", "reg_ece_y", "reg_ece_z",
              "reg_ece_h", "reg_ece_w", "reg_ece_l", "reg_ece_yaw"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("Paper-style ECE comparison (lower is better)")
    print(f"{'Method':24s} {'Cls ECE':>12s} {'Mean Reg ECE':>16s}")
    for r in rows:
        print(f"{r['method'][:24]:24s} "
              f"{r['classification_ece']:12.6f} "
              f"{r['mean_regression_ece']:16.6f}")
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
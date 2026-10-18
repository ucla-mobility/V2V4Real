"""
Paper-style uncertainty calibration utilities.

Implements the calibration definitions used by:
  Feng et al., "Can We Trust You? On Calibration of a Probabilistic
  Object Detector for Autonomous Driving" (2019).

Classification:
  Equal-width confidence bins. In each bin compare mean predicted object
  probability with empirical object frequency, then compute weighted ECE.

Regression:
  For each encoded box-regression dimension independently, assume the
  predictive marginal is Gaussian N(mean, variance). For confidence levels
  p in (0,1), compare p with the empirical frequency
      P(Y <= F^{-1}(p)).
  This is equivalent to checking whether PIT values F(Y) are Uniform(0,1).

This module accumulates only sufficient statistics, so it does not need to
save millions of dense anchor/pixel predictions.
"""
import csv
import json
import os
import numpy as np
import torch


def _normal_cdf(z):
    # Phi(z) without SciPy: 0.5 * (1 + erf(z/sqrt(2))).
    return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))


def _get_label_dict(batch_data):
    ego = batch_data["ego"]
    if "label_dict" not in ego:
        raise KeyError(
            "batch_data['ego']['label_dict'] is required for paper-style "
            "calibration, but was not found."
        )
    return ego["label_dict"]


def _get_reg_targets(label_dict):
    for key in ("targets", "reg_targets", "regression_targets"):
        if key in label_dict:
            return label_dict[key]
    raise KeyError(
        "Could not find encoded regression targets in label_dict. "
        "Expected one of: targets, reg_targets, regression_targets."
    )


def _as_bhwa(x, anchors):
    """Convert common classification-label layouts to [B,H,W,A]."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4-D classification tensor, got {tuple(x.shape)}")
    # OpenCOOD labels are normally [B,H,W,A].
    if x.shape[-1] == anchors:
        return x
    # Defensive support for [B,A,H,W].
    if x.shape[1] == anchors:
        return x.permute(0, 2, 3, 1).contiguous()
    raise ValueError(
        f"Cannot align classification label shape {tuple(x.shape)} "
        f"with {anchors} anchors."
    )


def _reg_to_bhwa7(x, anchors):
    """Convert [B,7*A,H,W] or [B,H,W,7*A] to [B,H,W,A,7]."""
    if x.ndim == 5 and x.shape[-2:] == (anchors, 7):
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected 4-D regression tensor, got {tuple(x.shape)}")
    if x.shape[1] == anchors * 7:
        x = x.permute(0, 2, 3, 1).contiguous()
    elif x.shape[-1] != anchors * 7:
        raise ValueError(
            f"Cannot align regression shape {tuple(x.shape)} with {anchors} anchors."
        )
    return x.reshape(x.shape[0], x.shape[1], x.shape[2], anchors, 7)


class PaperCalibrationAccumulator:
    """
    Streaming classification + marginal-regression calibration evaluator.

    Regression dimension names follow the OpenCOOD 7-D encoded target order
    used by the supplied scripts: [x, y, z, h, w, l, yaw].
    """
    def __init__(self, n_cls_bins=10, n_reg_levels=10, eps=1e-8):
        self.n_cls_bins = int(n_cls_bins)
        self.n_reg_levels = int(n_reg_levels)
        self.eps = float(eps)

        self.cls_edges = np.linspace(0.0, 1.0, self.n_cls_bins + 1)
        self.cls_count = np.zeros(self.n_cls_bins, dtype=np.int64)
        self.cls_conf_sum = np.zeros(self.n_cls_bins, dtype=np.float64)
        self.cls_label_sum = np.zeros(self.n_cls_bins, dtype=np.float64)

        # Midpoint levels avoid the infinite Gaussian quantiles at exactly 0/1.
        self.reg_levels = (
            np.arange(self.n_reg_levels, dtype=np.float64) + 0.5
        ) / self.n_reg_levels
        self.reg_below = np.zeros((7, self.n_reg_levels), dtype=np.int64)
        self.reg_count = np.zeros(7, dtype=np.int64)

        self.dim_names = ["x", "y", "z", "h", "w", "l", "yaw"]

    @torch.no_grad()
    def update(self, cls_prob, reg_mean, reg_var, batch_data):
        """
        cls_prob: [B,A,H,W] posterior/direct predictive object probability
        reg_mean: [B,7*A,H,W] encoded predictive mean
        reg_var:  [B,7*A,H,W] encoded predictive marginal variance
        """
        label_dict = _get_label_dict(batch_data)
        pos = label_dict["pos_equal_one"]
        neg = label_dict.get("neg_equal_one", None)

        anchors = int(cls_prob.shape[1])
        prob = cls_prob.permute(0, 2, 3, 1).contiguous()
        pos = _as_bhwa(pos, anchors).to(prob.device)

        if neg is not None:
            neg = _as_bhwa(neg, anchors).to(prob.device)
            valid = (pos > 0) | (neg > 0)
        else:
            # If no explicit negative mask exists, use all non-positive anchors.
            valid = torch.ones_like(pos, dtype=torch.bool)

        labels = (pos > 0).to(prob.dtype)
        p = prob[valid].clamp(0.0, 1.0).detach().cpu().numpy()
        y = labels[valid].detach().cpu().numpy()

        if p.size:
            # np.digitize gives 1..bins; map p=1 exactly to final bin.
            b = np.digitize(p, self.cls_edges[1:-1], right=False)
            self.cls_count += np.bincount(b, minlength=self.n_cls_bins)
            self.cls_conf_sum += np.bincount(
                b, weights=p, minlength=self.n_cls_bins
            )
            self.cls_label_sum += np.bincount(
                b, weights=y, minlength=self.n_cls_bins
            )

        # Regression calibration is defined only where a positive object target
        # exists, and independently for every encoded regression dimension.
        target = _get_reg_targets(label_dict).to(reg_mean.device)
        mu = _reg_to_bhwa7(reg_mean, anchors)
        var = _reg_to_bhwa7(reg_var, anchors).clamp_min(self.eps)
        target = _reg_to_bhwa7(target, anchors)

        pos_mask = (pos > 0)
        if not pos_mask.any():
            return

        mu_pos = mu[pos_mask]          # [Npos, 7]
        var_pos = var[pos_mask]        # [Npos, 7]
        target_pos = target[pos_mask]  # [Npos, 7]

        resid = target_pos - mu_pos
        # Yaw: training regresses sin(pred - gt) (add_sin_difference), so the
        # predicted variance describes sin(dyaw), not raw dyaw. Use the same
        # residual here; it also ignores 180-degree heading flips, which the
        # sin encoding treats as zero error.
        resid[:, 6] = torch.sin(resid[:, 6])
        z = resid / torch.sqrt(var_pos)
        pit = _normal_cdf(z).clamp(0.0, 1.0).detach().cpu().numpy()

        for d in range(7):
            vals = pit[:, d]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            self.reg_count[d] += vals.size
            # Empirical P(Y <= F^-1(p)) == empirical P(F(Y) <= p).
            self.reg_below[d] += (vals[:, None] <= self.reg_levels[None, :]).sum(axis=0)

    def classification_rows(self):
        rows = []
        total = int(self.cls_count.sum())
        ece = 0.0
        for i in range(self.n_cls_bins):
            n = int(self.cls_count[i])
            if n:
                mean_conf = self.cls_conf_sum[i] / n
                empirical = self.cls_label_sum[i] / n
                gap = abs(empirical - mean_conf)
                ece += (n / total) * gap
            else:
                mean_conf = np.nan
                empirical = np.nan
                gap = np.nan
            rows.append({
                "bin": i,
                "lower": self.cls_edges[i],
                "upper": self.cls_edges[i + 1],
                "count": n,
                "mean_predicted_probability": mean_conf,
                "empirical_probability": empirical,
                "abs_gap": gap,
            })
        return rows, float(ece) if total else float("nan")

    def regression_rows(self):
        rows = []
        dim_eces = {}
        for d, name in enumerate(self.dim_names):
            n = int(self.reg_count[d])
            if n == 0:
                dim_eces[name] = float("nan")
                continue
            empirical = self.reg_below[d] / n
            gaps = np.abs(empirical - self.reg_levels)
            # The paper compares the calibration curve to y=x. With equally
            # spaced confidence levels, the discrete mean absolute gap is the
            # natural marginal ECE summary.
            dim_eces[name] = float(gaps.mean())
            for level, emp, gap in zip(self.reg_levels, empirical, gaps):
                rows.append({
                    "dimension": name,
                    "count": n,
                    "predicted_confidence_level": float(level),
                    "empirical_confidence_level": float(emp),
                    "abs_gap": float(gap),
                })
        finite = [v for v in dim_eces.values() if np.isfinite(v)]
        avg = float(np.mean(finite)) if finite else float("nan")
        return rows, dim_eces, avg

    def save(self, out_dir, method):
        os.makedirs(out_dir, exist_ok=True)
        safe = method.lower().replace(" ", "_").replace("-", "_")

        cls_rows, cls_ece = self.classification_rows()
        reg_rows, reg_eces, reg_avg = self.regression_rows()

        cls_csv = os.path.join(out_dir, f"paper_classification_calibration_{safe}.csv")
        with open(cls_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
            w.writeheader()
            w.writerows(cls_rows)

        reg_csv = os.path.join(out_dir, f"paper_regression_calibration_{safe}.csv")
        with open(reg_csv, "w", newline="") as f:
            fields = ["dimension", "count", "predicted_confidence_level",
                      "empirical_confidence_level", "abs_gap"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(reg_rows)

        summary = {
            "method": method,
            "classification_ece": cls_ece,
            "classification_samples": int(self.cls_count.sum()),
            "regression_ece_per_dimension": reg_eces,
            "mean_regression_ece": reg_avg,
            "regression_samples_per_dimension": {
                name: int(self.reg_count[i]) for i, name in enumerate(self.dim_names)
            },
            "regression_definition":
                "Marginal Gaussian calibration in encoded OpenCOOD target space; "
                "empirical P(Y <= F^-1(p)) compared with p.",
            "classification_definition":
                "Dense object/background probability calibration using "
                "pos_equal_one/neg_equal_one labels; no 2 m or IoU threshold.",
        }
        summary_json = os.path.join(out_dir, f"paper_ece_summary_{safe}.json")
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Paper ECE] {method}: classification ECE = {cls_ece:.6f}")
        print(f"[Paper ECE] {method}: regression ECE per dim = {reg_eces}")
        print(f"[Paper ECE] {method}: mean regression ECE = {reg_avg:.6f}")
        print(f"[Paper ECE] Saved: {cls_csv}")
        print(f"[Paper ECE] Saved: {reg_csv}")
        print(f"[Paper ECE] Saved: {summary_json}")
        return summary
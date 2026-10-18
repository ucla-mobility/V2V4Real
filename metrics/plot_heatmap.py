#!/usr/bin/env python3
"""
Bird's-eye-view uncertainty heatmaps for any number of methods on the SAME frame.

Input: one standardized .npz per method (written by inference_vn.py,
inference_laplace.py, inference_mc.py), e.g.
    --npz <vn_dir>/visualization_npz_vn/00123.npz      --label "VN-ES"
    --npz <laplace_dir>/visualization_npz_laplace/00123.npz --label "Baseline + Laplace"
    --npz <laplace_dir>/visualization_npz_laplace/00123.npz --label "Laplace (epistemic)" \
          --key epistemic_reg_variance

What is plotted
- Regression: the box-centre std in metres, sqrt(d^2 (var_x + var_y) + h^2 var_z)
  (d = anchor diagonal, h = anchor height), i.e. the same quantity used for
  AUSE / Pearson / Spearman (metrics.uncertainty_matching_utils.centre_variance_m2).
  Each BEV cell shows the mean over the anchor orientations at that cell.
  Because every method is in metres, one shared colour scale is meaningful.
- Classification (--kind classification): the chosen dense classification
  uncertainty (e.g. mutual_information), averaged over anchors.
- The map is placed with the frame's own cav_lidar_range, so it lines up
  with the LiDAR points and boxes.

The script does not alter or recompute any model uncertainty.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter

try:
    from metrics.uncertainty_matching_utils import centre_variance_m2
except ModuleNotFoundError:
    # Allows: python metrics/plot_heatmap.py ... from the project root.
    from uncertainty_matching_utils import centre_variance_m2


REG_KEYS = ["reg_variance", "regression_uncertainty", "rm_var", "var_rm",
            "variance"]
CLS_KEYS = ["mutual_information", "classification_uncertainty",
            "cls_variance", "predictive_entropy", "var_psm"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", action="append", required=True,
                   help="Per-frame .npz of one method. Repeat per panel.")
    p.add_argument("--label", action="append", required=True,
                   help="Panel title for each --npz.")
    p.add_argument("--key", action="append", default=None,
                   help="Array to plot for each --npz ('auto' = default "
                        "variance key). Repeat once per --npz if used.")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", action="append",
                   choices=["regression", "classification"], default=None,
                   help="Given once: applies to all panels. Or once per --npz "
                        "to mix regression and classification panels.")
    p.add_argument("--ncols", type=int, default=1,
                   help="Panels per row (e.g. 2 for a compact grid).")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--scale", choices=["shared", "own"], default="shared",
                   help="'shared': one colour scale for all panels (valid "
                        "for regression, all in metres). 'own': per panel.")
    p.add_argument("--anchor_l", type=float, default=3.9)
    p.add_argument("--anchor_w", type=float, default=1.6)
    p.add_argument("--anchor_h", type=float, default=1.56)
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   help="Zoom in x (m). Default: full cav_lidar_range.")
    p.add_argument("--ylim", type=float, nargs=2, default=None)
    p.add_argument("--heat-alpha", type=float, default=0.9)
    p.add_argument("--point-size", type=float, default=0.05)
    p.add_argument("--cmap", default="Blues",
                   help="Colour map (default 'Blues': no green, so the boxes stay visible).")
    p.add_argument("--simple", type=int, default=1, choices=[0, 1],
                   help="1 (default): panel title = method name only, one colour bar "
                        "for all panels. 0: also show the plotted array name.")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()
    if len(a.npz) != len(a.label):
        p.error("--npz and --label must be given the same number of times.")
    if a.key is None:
        a.key = ["auto"] * len(a.npz)
    if len(a.key) != len(a.npz):
        p.error("--key must be given once per --npz (use 'auto').")
    if a.kind is None:
        a.kind = ["regression"]
    if len(a.kind) == 1:
        a.kind = a.kind * len(a.npz)
    if len(a.kind) != len(a.npz):
        p.error("--kind must be given once, or once per --npz.")
    if a.scale == "shared" and len(set(a.kind)) > 1:
        p.error("--scale shared needs all panels of the same --kind.")
    return a


def pick(d, key, defaults):
    keys = defaults if key == "auto" else [key]
    for k in keys:
        if k in d.files:
            return np.asarray(d[k], dtype=np.float64), k
    raise KeyError("None of these keys found: " + ", ".join(keys)
                   + "\nAvailable keys: " + ", ".join(d.files))


def uncertainty_map(d, key, kind, a):
    """Returns (H, W) map and a description of what it shows."""
    if kind == "regression":
        x, k = pick(d, key, REG_KEYS)
        x = np.squeeze(x, axis=0) if x.ndim == 4 else x       # [7A, H, W]
        n_anchor = x.shape[0] // 7
        x = x.reshape(n_anchor, 7, *x.shape[1:])
        anchor_args = {"l": a.anchor_l, "w": a.anchor_w, "h": a.anchor_h}
        var_m2 = centre_variance_m2(x, anchor_args, dim_axis=1)  # [A, H, W]
        return np.sqrt(var_m2.mean(axis=0)), f"{k} -> centre std (m)"
    x, k = pick(d, key, CLS_KEYS)
    x = np.squeeze(x)
    if x.ndim == 3:
        x = x.mean(axis=0)
    return x, k


def footprint(box):
    xy = np.asarray(box)[:, :2]
    # remove duplicate top/bottom corners in BEV
    _, idx = np.unique(np.round(xy, 5), axis=0, return_index=True)
    xy = xy[np.sort(idx)]
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    xy = xy[np.argsort(ang)]
    return np.vstack([xy, xy[0]])


def draw_boxes(ax, arr, color, linestyle, label):
    arr = np.asarray(arr)
    if arr.size == 0:
        return
    for i, b in enumerate(arr):
        f = footprint(b)
        ax.plot(f[:, 0], f[:, 1], color=color, linestyle=linestyle,
                linewidth=1.0, label=label if i == 0 else None)


def main():
    a = parse_args()
    ds = [np.load(p, allow_pickle=True) for p in a.npz]
    maps, descs = zip(*[uncertainty_map(d, k, kind, a)
                        for d, k, kind in zip(ds, a.key, a.kind)])

    norm_shared = None
    if a.scale == "shared":
        pooled = np.concatenate([m[np.isfinite(m) & (m > 0)].ravel()
                                 for m in maps])
        lo, hi = np.percentile(pooled, [1, 99])
        norm_shared = LogNorm(vmin=lo, vmax=hi)

    xmin, ymin, _, xmax, ymax, _ = np.asarray(ds[0]["cav_lidar_range"])
    x0, x1 = a.xlim if a.xlim else (xmin, xmax)
    y0, y1 = a.ylim if a.ylim else (ymin, ymax)
    # Figure width follows the plotted aspect ratio so zoomed views fill it.
    panel_h = 2.9
    n = len(ds)
    ncols = max(1, min(a.ncols, n))
    nrows = int(np.ceil(n / ncols))
    aspect = (x1 - x0) / (y1 - y0)
    if ncols == 1:
        width = min(14.0, panel_h * aspect + 2.5)
    else:
        width = ncols * (0.85 * panel_h * aspect + 1.4)
    fig, grid = plt.subplots(nrows, ncols,
                             figsize=(width, panel_h * nrows + 0.6),
                             constrained_layout=True, squeeze=False)
    axs = grid.ravel()
    for ax in axs[n:]:
        ax.axis("off")

    for i, (ax, d, m, name, desc, kind) in enumerate(
            zip(axs, ds, maps, a.label, descs, a.kind)):
        if norm_shared is not None:
            norm = norm_shared
        elif a.simple:
            # own linear scale per panel, 2nd-98th percentile: easy to read in metres
            v = m[np.isfinite(m) & (m > 0)]
            norm = Normalize(*np.percentile(v, [2, 98]))
        else:
            v = m[np.isfinite(m) & (m > 0)]
            norm = LogNorm(*np.percentile(v, [1, 99]))
        im = ax.imshow(m, origin="lower", extent=[xmin, xmax, ymin, ymax],
                       cmap=a.cmap, norm=norm, interpolation="nearest",
                       alpha=a.heat_alpha, aspect="equal")
        if "pcd" in d.files:
            pc = np.asarray(d["pcd"])
            if pc.ndim == 2 and pc.shape[1] >= 2:
                ax.scatter(pc[:, 0], pc[:, 1], s=a.point_size, c="0.35",
                           alpha=0.5, linewidths=0, rasterized=True)
        if "gt_boxes" in d.files:
            draw_boxes(ax, d["gt_boxes"], "black", "--", "Ground truth (real car)")
        if "pred_boxes" in d.files:
            draw_boxes(ax, d["pred_boxes"], "#e00000", "-", "Detection")

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(name if a.simple else f"{name}\n[{desc}]", fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel("y (m)")
        if i >= n - ncols:
            ax.set_xlabel("x (m)")
        if a.simple and norm_shared is None:
            fmt = "%.3f" if (norm.vmax - norm.vmin) < 0.1 else "%.2f"
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01,
                              format=FormatStrFormatter(fmt))
            cb.set_label("m" if kind == "regression" else "nats")
            cb.ax.tick_params(labelsize=7)
        elif not a.simple:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
            cb.set_label("centre std (m)" if kind == "regression"
                         else "uncertainty (nats)")
    if a.simple and norm_shared is not None:
        cb = fig.colorbar(im, ax=list(axs[:n]), fraction=0.025, pad=0.01)
        cb.set_label("Uncertainty of the box centre (m)\nlight = sure, dark = unsure"
                     if set(a.kind) == {"regression"} else "Uncertainty")
    axs[0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    if a.title:
        title = a.title
    elif set(a.kind) == {"regression"}:
        title = "Regression uncertainty (box-centre std, m)"
    elif set(a.kind) == {"classification"}:
        title = "Classification uncertainty"
    else:
        title = "Uncertainty comparison"
    fig.suptitle(title, fontsize=12)
    fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight")
    print("Saved:", a.output)
    for name, desc in zip(a.label, descs):
        print(f"  {name}: {desc}")


if __name__ == "__main__":
    main()

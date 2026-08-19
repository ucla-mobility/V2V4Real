#!/usr/bin/env python3
"""
Compare uncertainty heatmaps for:
  1) MC Dropout
  2) VN-NLL
  3) VN-Energy Score
  4) Laplace

Input: one standardized .npz per method for the SAME frame.
Required keys:
  pred_boxes, gt_boxes
Recommended:
  pcd
  regression_uncertainty OR one native key:
    MC:      reg_variance / var_rm
    VN:      regression_uncertainty / reg_variance / rm_var / variance
    Laplace: reg_variance
Optional classification keys:
    classification_uncertainty / mutual_information /
    cls_variance / predictive_entropy / var_psm

Important:
- The script does NOT alter or recompute uncertainty.
- For fair spatial comparison it defaults to regression uncertainty.
- Each method gets its own color scale by default because MC/VN/Laplace
  uncertainty magnitudes are not guaranteed to have the same units/scale.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mc", required=True)
    p.add_argument("--vn_nll", required=True)
    p.add_argument("--vn_energy", required=True)
    p.add_argument("--laplace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["regression", "classification"],
                   default="regression")
    p.add_argument("--xmin", type=float, default=-50)
    p.add_argument("--xmax", type=float, default=50)
    p.add_argument("--ymin", type=float, default=-50)
    p.add_argument("--ymax", type=float, default=50)
    p.add_argument("--heat-alpha", type=float, default=0.72)
    p.add_argument("--point-size", type=float, default=0.12)
    p.add_argument("--shared-scale", action="store_true",
                   help="Use only if all four uncertainty maps have comparable units/scales.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def pick(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k]), k
    raise KeyError("None of these uncertainty keys found: " + ", ".join(keys)
                   + "\nAvailable keys: " + ", ".join(d.files))


def bev_reduce(x):
    x = np.squeeze(np.asarray(x))
    if x.ndim == 2:
        return x.astype(float)
    if x.ndim < 2:
        raise ValueError(f"Cannot form BEV heatmap from shape {x.shape}")
    # OpenCOOD dense heads use final H,W spatial dimensions.
    return x.mean(axis=tuple(range(x.ndim - 2))).astype(float)


def uncertainty_map(d, kind):
    if kind == "regression":
        x, key = pick(d, [
            "regression_uncertainty", "reg_variance",
            "var_rm", "rm_var", "variance"
        ])
    else:
        x, key = pick(d, [
            "classification_uncertainty", "mutual_information",
            "cls_variance", "predictive_entropy", "var_psm"
        ])
    return bev_reduce(x), key


def footprint(box):
    xy = np.asarray(box)[:, :2]
    # remove duplicate top/bottom corners in BEV
    _, idx = np.unique(np.round(xy, 5), axis=0, return_index=True)
    xy = xy[np.sort(idx)]
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1]-c[1], xy[:, 0]-c[0])
    xy = xy[np.argsort(ang)]
    return np.vstack([xy, xy[0]])


def boxes(ax, arr, linestyle, label):
    arr = np.asarray(arr)
    if arr.size == 0:
        return
    first = True
    for b in arr:
        f = footprint(b)
        ax.plot(f[:,0], f[:,1], linestyle=linestyle, linewidth=1.1,
                label=label if first else None)
        first = False


def main():
    a = args()
    paths = [a.mc, a.vn_nll, a.vn_energy, a.laplace]
    names = ["MC Dropout", "VN-NLL", "VN-Energy", "Laplace"]
    ds = [np.load(p, allow_pickle=True) for p in paths]

    maps, used_keys = [], []
    for d in ds:
        m, k = uncertainty_map(d, a.kind)
        maps.append(m)
        used_keys.append(k)

    shared_vmin = shared_vmax = None
    if a.shared_scale:
        shared_vmin = min(float(np.nanmin(m)) for m in maps)
        shared_vmax = max(float(np.nanmax(m)) for m in maps)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)

    for ax, d, m, name, key in zip(axs, ds, maps, names, used_keys):
        if "pcd" in d:
            pc = np.squeeze(np.asarray(d["pcd"]))
            if pc.ndim == 2 and pc.shape[1] >= 2:
                ax.scatter(pc[:,0], pc[:,1], s=a.point_size,
                           alpha=0.38, rasterized=True)

        kwargs = {}
        if a.shared_scale:
            kwargs["vmin"] = shared_vmin
            kwargs["vmax"] = shared_vmax

        im = ax.imshow(
            m, origin="lower",
            extent=[a.xmin, a.xmax, a.ymin, a.ymax],
            interpolation="bilinear", alpha=a.heat_alpha,
            aspect="equal", **kwargs
        )

        if "gt_boxes" in d:
            boxes(ax, d["gt_boxes"], "--", "GT")
        if "pred_boxes" in d:
            boxes(ax, d["pred_boxes"], "-", "Prediction")

        ax.set_xlim(a.xmin, a.xmax)
        ax.set_ylim(a.ymin, a.ymax)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\n{key}")
        ax.set_xlabel("x (m)")
        if ax is axs[0]:
            ax.set_ylabel("y (m)")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Uncertainty")

    fig.suptitle(
        "Regression uncertainty comparison" if a.kind == "regression"
        else "Classification uncertainty comparison",
        fontsize=14
    )
    fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight")
    print("Saved:", a.output)
    print("Uncertainty keys used:")
    for n, k in zip(names, used_keys):
        print(f"  {n}: {k}")


if __name__ == "__main__":
    main()
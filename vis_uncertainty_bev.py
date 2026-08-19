"""
vis_uncertainty_bev.py

Fig-10-style qualitative uncertainty visualization: a top-down (BEV) LiDAR
plot with predicted 3D boxes drawn as rotated 2D rectangles, colored by
their predicted uncertainty instead of by TP/FP. Ground-truth boxes are
drawn as thin green outlines for reference, exactly like the paper figure.

Drop this file into opencood/visualization/. It has no dependency on the
rest of opencood -- it only needs plain numpy arrays, so it can be called
from inference_vn.py / inference_mc.py right where pred_box_tensor,
pred_var_tensor and gt_box_tensor are already available (see the bottom of
this file for the exact call to add).

Design choice: this uses matplotlib instead of the existing Open3D-based
vis_utils.py. Open3D's bbx2linset() only supports one fixed color per
whole set of boxes (all-red preds / all-green GT), so per-box coloring
would mean rewriting the Open3D pipeline and its offscreen-rendering setup.
A 2D matplotlib BEV plot gives full per-box color/label control, a
colorbar, and easy multi-scene grid layouts (like Fig. 10's 2-column
grid) with much less code, and is the standard way these qualitative
uncertainty figures are made in the uncertainty-estimation literature.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


def _box_to_2d_corners(box_corners_3d):
    """
    box_corners_3d: (8, 3) array, the 8 corners of one 3D box (as returned
    by box_utils.boxes_to_corners_3d, same format used everywhere else in
    this codebase for pred_box_tensor / gt_box_tensor).

    Returns the 4 BEV (x, y) corners in the order needed to draw a closed
    rectangle, i.e. the bottom face of the box viewed from above.
    """
    xy = box_corners_3d[:4, :2]  # bottom face, 4 corners, already ordered
    return xy


def _variance_to_scalar(var_vec, dims=(0, 1)):
    """
    var_vec: (7,) array [x, y, z, h, w, l, yaw] predicted variance for one
    box (this is exactly one row of pred_var_tensor from
    post_process_with_variance()).

    Reduces to a single scalar for coloring. Default: mean of the x, y
    (position-only) variance, matching the same fix recommended for
    error_vs_variance.csv in the code audit (don't blend position, size,
    and yaw variance -- they're different units).
    """
    return float(np.mean(var_vec[list(dims)]))


def plot_bev_uncertainty(ax,
                          lidar_xy,
                          pred_boxes_2d,
                          pred_uncertainty,
                          gt_boxes_2d=None,
                          score_labels=None,
                          cmap_name="RdYlGn_r",
                          vmin=None,
                          vmax=None,
                          point_size=0.3,
                          xlim=None,
                          ylim=None,
                          title=None):
    """
    Draw one BEV panel (one "scene", matching one cell of Fig. 10's grid)
    onto a given matplotlib Axes.

    lidar_xy          : (N, 2) array of LiDAR point x,y (already in ego
                        frame -- this is origin_lidar[:, :2] from the same
                        batch_data['ego']['origin_lidar'] used elsewhere
                        in inference_vn.py / infrence_utils.py).
    pred_boxes_2d     : list of (4, 2) arrays, one per predicted box
                        (output of _box_to_2d_corners for each surviving
                        box in pred_box_tensor).
    pred_uncertainty  : (M,) array, one scalar uncertainty per predicted
                        box, aligned 1:1 with pred_boxes_2d (output of
                        _variance_to_scalar for each row of
                        pred_var_tensor).
    gt_boxes_2d       : optional list of (4, 2) arrays for ground truth.
    score_labels      : optional (M,) array of detection confidence, drawn
                        as small text next to each box like the paper
                        figure.
    """
    # background point cloud, dim so the boxes stand out
    ax.scatter(lidar_xy[:, 0], lidar_xy[:, 1], s=point_size,
               c="#cfcfcf", linewidths=0, alpha=0.6)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    if gt_boxes_2d is not None:
        for gt in gt_boxes_2d:
            poly = Polygon(gt, closed=True, fill=False,
                            edgecolor="#00c853", linewidth=1.1, zorder=3)
            ax.add_patch(poly)

    for i, box in enumerate(pred_boxes_2d):
        color = cmap(norm(pred_uncertainty[i]))
        poly = Polygon(box, closed=True, fill=False,
                        edgecolor=color, linewidth=1.6, zorder=4)
        ax.add_patch(poly)
        if score_labels is not None:
            cx, cy = box.mean(axis=0)
            ax.text(cx, cy + 1.0, f"{score_labels[i]:.2f}",
                    color=color, fontsize=5, ha="center", zorder=5)

    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=9)

    return cmap, norm


def make_uncertainty_grid_figure(scenes, out_path,
                                  row_labels=None, col_labels=None,
                                  cmap_name="RdYlGn_r",
                                  figsize_per_cell=(3.2, 2.6)):
    """
    scenes: 2D list [row][col] of dicts, each with keys
        'lidar_xy', 'pred_boxes_2d', 'pred_uncertainty',
        'gt_boxes_2d' (optional), 'score_labels' (optional)
    exactly like Fig. 10: rows = methods (No Fusion / MC Dropout /
    NLL-VN / ES-VN / ...), columns = scenes (Scene 1 / Scene 2 / ...).

    Uses one shared color scale across every panel so uncertainty is
    visually comparable across methods -- the whole point of the figure.
    """
    n_rows = len(scenes)
    n_cols = len(scenes[0])

    all_unc = np.concatenate([
        np.asarray(cell["pred_uncertainty"])
        for row in scenes for cell in row
        if len(cell["pred_uncertainty"]) > 0
    ]) if any(len(cell["pred_uncertainty"]) for row in scenes for cell in row) else np.array([0.0, 1.0])
    vmin, vmax = float(all_unc.min()), float(all_unc.max())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False
    )

    cmap = last_norm = None
    for r in range(n_rows):
        for c in range(n_cols):
            cell = scenes[r][c]
            title = col_labels[c] if (row_labels is None or r == 0) and col_labels else None
            cmap, last_norm = plot_bev_uncertainty(
                axes[r][c],
                np.asarray(cell["lidar_xy"]),
                cell["pred_boxes_2d"],
                np.asarray(cell["pred_uncertainty"]),
                gt_boxes_2d=cell.get("gt_boxes_2d"),
                score_labels=cell.get("score_labels"),
                cmap_name=cmap_name, vmin=vmin, vmax=vmax,
                title=title,
            )
        if row_labels:
            axes[r][0].set_ylabel(row_labels[r], fontsize=9, rotation=90)
            axes[r][0].yaxis.label.set_visible(True)

    fig.subplots_adjust(right=0.90, wspace=0.02, hspace=0.05)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = cm.ScalarMappable(norm=last_norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Predicted uncertainty (var, x/y)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Wiring into inference_vn.py / inference_mc.py (reference, not executed
# here): inside the per-frame loop, right after
#   pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor = \
#       infrence_utils.inference_intermediate_fusion_with_variance(...)
# add:
#
#   from opencood.visualization.vis_uncertainty_bev import (
#       _box_to_2d_corners, _variance_to_scalar)
#
#   lidar_xy = batch_data['ego']['origin_lidar'][0][:, :2].cpu().numpy()
#   pred_boxes_2d = [_box_to_2d_corners(b.cpu().numpy())
#                     for b in pred_box_tensor] if pred_box_tensor is not None else []
#   pred_unc = [_variance_to_scalar(v.cpu().numpy())
#                for v in pred_var_tensor] if pred_var_tensor is not None else []
#   gt_boxes_2d = [_box_to_2d_corners(b.cpu().numpy())
#                   for b in gt_box_tensor] if gt_box_tensor is not None else []
#
#   frame_cache.append(dict(lidar_xy=lidar_xy, pred_boxes_2d=pred_boxes_2d,
#                            pred_uncertainty=pred_unc, gt_boxes_2d=gt_boxes_2d,
#                            score_labels=pred_score.cpu().numpy()
#                                if pred_score is not None else None))
#
# Then, once you've hand-picked which frame indices are "Scene 1" /
# "Scene 2" (the paper does the same -- these are curated, not random),
# and run this once per method (baseline/MC/NLL-VN/ES-VN) to fill in each
# row, call:
#
#   make_uncertainty_grid_figure(
#       scenes=[[frame_cache_baseline[scene1_idx], frame_cache_baseline[scene2_idx]],
#               [frame_cache_mc[scene1_idx],       frame_cache_mc[scene2_idx]],
#               [frame_cache_nll[scene1_idx],       frame_cache_nll[scene2_idx]],
#               [frame_cache_es[scene1_idx],        frame_cache_es[scene2_idx]]],
#       out_path="uncertainty_qualitative.png",
#       row_labels=["Baseline", "MC Dropout", "NLL-VN", "ES-VN"],
#       col_labels=["Scene 1", "Scene 2"],
#   )
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # ---- self-test / demo with synthetic data, no repo/dataset needed ----
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_demo_out")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)

    def fake_scene(n_points=4000, n_boxes=6, unc_scale=1.0, seed=0):
        r = np.random.default_rng(seed)
        pts = r.normal(0, 25, size=(n_points, 2))
        boxes, unc, scores = [], [], []
        for _ in range(n_boxes):
            cx, cy = r.uniform(-30, 30, size=2)
            w, l = 1.8, 4.2
            ang = r.uniform(0, np.pi)
            local = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            box = local @ rot.T + np.array([cx, cy])
            boxes.append(box)
            unc.append(r.uniform(0, unc_scale))
            scores.append(r.uniform(0.3, 0.95))
        gt = [b + r.normal(0, 0.3, size=b.shape) for b in boxes[:4]]
        return dict(lidar_xy=pts, pred_boxes_2d=boxes, pred_uncertainty=unc,
                    gt_boxes_2d=gt, score_labels=scores)

    scenes = [
        [fake_scene(unc_scale=0.3, seed=1), fake_scene(unc_scale=0.3, seed=2)],
        [fake_scene(unc_scale=1.0, seed=3), fake_scene(unc_scale=1.0, seed=4)],
        [fake_scene(unc_scale=2.0, seed=5), fake_scene(unc_scale=2.0, seed=6)],
    ]
    make_uncertainty_grid_figure(
        scenes, os.path.join(out_dir, "demo_uncertainty_grid.png"),
        row_labels=["Baseline (low unc.)", "NLL-VN (medium)", "ES-VN (higher, uncalibrated)"],
        col_labels=["Scene 1", "Scene 2"],
    )

"""
plot_method_comparison_frame.py

Reproduces the paper's Figure-13 style grid (methods as rows/cols, GT boxes
in green, predicted boxes in red, over a BEV lidar point cloud) but for
uncertainty methods instead of domain adaptation: columns are
Baseline / MC-Dropout / VN-NLL / VN-ES, rows are the real frame(s) you pick.

Reads the .npy files each inference script already writes with --save_npy:
    NNNN_pcd.npy   (P, 4)   lidar points, xyzi
    NNNN_gt.npy    (G, 8, 3) GT box corners
    NNNN_pred.npy  (N, 8, 3) predicted box corners
No re-running the model, no GPU needed -- just reads what's already on disk.

Usage
-----
One frame, four methods (order of --npy_dir matches order of --label):
    python plot_method_comparison_frame.py \
        --npy_dir baseline_model_dir/npy   --label "Baseline" \
        --npy_dir mc_model_dir/npy_mc      --label "MC-Dropout" \
        --npy_dir nll_vn_model_dir/npy_vn  --label "VN-NLL" \
        --npy_dir es_vn_model_dir/npy_vn   --label "VN-ES" \
        --frame_id 7 \
        --out method_comparison_frame7.png

Multiple frames (one row per frame_id, repeat --frame_id):
        --frame_id 7 --frame_id 42 --frame_id 88

Notes
-----
- Every --npy_dir must contain the SAME frame_id(s) -- only true if all runs
  share the identical, ordered test dataloader (shuffle=False,
  drop_last=False), which every inference_*.py script here already sets.
- A method missing a frame is skipped with a warning, not a crash; that
  panel is left blank so the grid still renders for the other methods.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def _corners_to_bev(corners_3d):
    """(8,3) box corners -> (4,2) BEV rectangle, bottom face."""
    return corners_3d[:4, :2]


def load_frame_boxes(npy_dir, frame_id):
    """
    Reads one frame's pcd/gt/pred .npy files for a single method.
    Returns None if the pcd file (the minimum required) is missing.
    """
    tag = "%04d" % frame_id
    pcd_path = os.path.join(npy_dir, f"{tag}_pcd.npy")
    gt_path = os.path.join(npy_dir, f"{tag}_gt.npy")
    pred_path = os.path.join(npy_dir, f"{tag}_pred.npy")

    if not os.path.exists(pcd_path):
        return None

    pcd = np.load(pcd_path)
    gt = np.load(gt_path) if os.path.exists(gt_path) else np.zeros((0, 8, 3))
    pred = np.load(pred_path) if os.path.exists(pred_path) else np.zeros((0, 8, 3))

    gt_boxes_2d = [_corners_to_bev(b) for b in gt]
    pred_boxes_2d = [_corners_to_bev(b) for b in pred]

    return dict(lidar_xy=pcd[:, :2], gt_boxes_2d=gt_boxes_2d,
               pred_boxes_2d=pred_boxes_2d)


def draw_panel(ax, cell, title=None, row_label=None):
    ax.set_facecolor("black")

    if cell is None:
        ax.text(0.5, 0.5, "no data", color="white", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold")
        if row_label:
            ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
        return

    ax.scatter(cell["lidar_xy"][:, 0], cell["lidar_xy"][:, 1],
              s=0.3, c="white", alpha=0.5, linewidths=0)

    for box in cell["gt_boxes_2d"]:
        ax.add_patch(Polygon(box, closed=True, fill=False,
                             edgecolor="limegreen", linewidth=1.2))

    for box in cell["pred_boxes_2d"]:
        ax.add_patch(Polygon(box, closed=True, fill=False,
                             edgecolor="red", linewidth=1.2))

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if row_label:
        ax.set_ylabel(row_label, fontsize=10, fontweight="bold", rotation=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", action="append", required=True,
                        help="One npy folder per method column, e.g. "
                             "model_dir/npy, model_dir/npy_mc, model_dir/npy_vn "
                             "(repeat --npy_dir/--label together, same order)")
    parser.add_argument("--label", action="append", required=True,
                        help="Column label for each --npy_dir, in the same "
                             "order (e.g. 'Baseline', 'MC-Dropout', 'VN-NLL', 'VN-ES')")
    parser.add_argument("--frame_id", type=int, action="append", required=True,
                        help="Frame index (repeat for multiple rows)")
    parser.add_argument("--out", type=str, default="method_comparison_frame.png",
                        help="Output PNG path")
    args = parser.parse_args()

    if len(args.npy_dir) != len(args.label):
        raise ValueError("Number of --npy_dir and --label arguments must match")

    n_rows = len(args.frame_id)
    n_cols = len(args.npy_dir)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for r, frame_id in enumerate(args.frame_id):
        for c, (npy_dir, label) in enumerate(zip(args.npy_dir, args.label)):
            ax = axes[r][c]
            cell = load_frame_boxes(npy_dir, frame_id)
            if cell is None:
                print(f"[warn] no frame {frame_id:04d} found in {npy_dir} "
                      f"-- leaving that panel blank")
            title = label if r == 0 else None
            row_label = f"Frame {frame_id}" if c == 0 else None
            draw_panel(ax, cell, title=title, row_label=row_label)

    # legend, matching the "Green/red = GT/prediction" caption style
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="limegreen", lw=1.5, label="Ground truth"),
        Line2D([0], [0], color="red", lw=1.5, label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
              frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.subplots_adjust(bottom=0.05 + 0.03 / n_rows)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
"""
Scene-level uncertainty figure (bird's-eye view of one V2V4Real frame).

LiDAR points of both CAVs (coloured per vehicle, in the ego frame), the
ground-truth boxes, and every detection coloured by its predicted box-centre
uncertainty sigma_c = sqrt(sigma_x^2 + sigma_y^2) in metres. One panel per
method, all on the same colour scale, so the methods can be compared side by
side on the same scene.

Per-box variances come from the same inference paths as the evaluation:
VN and Laplace through visualize_object_uncertainty.run_method, MC-dropout
as in inference_mc.py (T passes averaged before NMS, sample variance plus the
observation noise written in the MC evaluation's infer.log).

Methods are repeated --method "key=value,..." strings, as in
visualize_object_uncertainty.py, plus type=mc with:
    model_dir=...       MC model folder (config with the V2V4Real test set)
    noise_log=...       infer.log of the MC evaluation (observation noise)
    passes=20           MC forward passes
"""

import argparse
import ast
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.inference_laplace import move_to_device
from opencood.tools.inference_mc import enable_mc_dropout, probability_to_logit
from opencood.tools.visualize_object_uncertainty import (
    _np,
    load_model,
    metric_sigmas,
    polygon,
    run_method,
)

CAV_COLOURS = ["#c0504d", "#4f9a4a"]  # ego, other CAV (as in imgs/scene1.png)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--method", action="append", required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--output_dir", default="results/scene_uncertainty")
    p.add_argument("--x_range", type=float, nargs=2, default=[-70.0, 70.0])
    p.add_argument("--y_range", type=float, nargs=2, default=[-38.4, 38.4])
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--model_name", default="F-Cooper")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_method(text):
    spec = dict(kv.split("=", 1) for kv in text.split(","))
    for key in ("label", "type", "model_dir"):
        if key not in spec:
            raise ValueError(f"--method needs {key}=...: {text}")
    if spec["type"] == "laplace":
        spec.setdefault("scope", "heads")
        spec.setdefault("part", "total")
        spec["num_samples"] = int(spec.get("num_samples", 200))
    elif spec["type"] == "vn":
        spec.setdefault("part", "aleatoric")
    elif spec["type"] == "mc":
        spec["passes"] = int(spec.get("passes", 20))
        if "noise_log" not in spec:
            raise ValueError(f"mc method needs noise_log=...: {text}")
    return spec


def mc_noise_var(log_path):
    """Observation-noise variance printed by inference_mc.py."""
    with open(log_path) as f:
        for line in f:
            m = re.search(r"\[MC\] Observation noise variance.*?:\s*(\[.*\])",
                          line)
            if m:
                return torch.tensor(ast.literal_eval(m.group(1)))
    raise ValueError(f"no MC observation noise in {log_path}")


def run_mc(spec, frames, dataset, device, seed):
    hypes, model = load_model(spec["model_dir"], device)
    enable_mc_dropout(model)
    noise = mc_noise_var(spec["noise_log"]).to(device)
    results = {}
    for frame in frames:
        torch.manual_seed(seed + frame)
        batch = move_to_device(
            dataset.collate_batch_test([dataset[frame]]), device)
        with torch.no_grad():
            outs = [model(batch["ego"]) for _ in range(spec["passes"])]
            prob = torch.sigmoid(torch.stack([o["psm"] for o in outs]))
            rm = torch.stack([o["rm"] for o in outs])
            var = rm.var(dim=0, unbiased=False) + noise.repeat(
                rm.shape[2] // 7).view(1, -1, 1, 1)
            mean_rm = rm.mean(dim=0)
            output = {"psm": probability_to_logit(prob.mean(dim=0)),
                      "rm": mean_rm, "rm_var": var.expand_as(mean_rm)}
            boxes, scores, var, gt = dataset.post_process_with_variance(
                batch, {"ego": output})
        results[frame] = {"boxes": _np(boxes, (0, 8, 3)),
                          "scores": _np(scores, (0,)),
                          "var": _np(var, (0, 7)),
                          "gt": _np(gt, (0, 8, 3))}
    del model
    torch.cuda.empty_cache()
    return hypes, results


def cav_points(dataset, frame):
    """LiDAR points of every CAV in the ego frame, ego first, and each CAV's
    position in the ego frame."""
    base = dataset.retrieve_base_data(frame, cur_ego_pose_flag=True)
    items = sorted(base.items(), key=lambda kv: not kv[1]["ego"])
    pts, origins = [], []
    for _, cav in items:
        xyz = cav["lidar_np"][:, :3]
        T = np.asarray(cav["params"]["transformation_matrix"])
        pts.append(xyz @ T[:3, :3].T + T[:3, 3])
        origins.append(T[:3, 3])
    return pts, origins


def main():
    args = parse_args()
    specs = [parse_method(m) for m in args.method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    hypes0 = load_yaml(None, argparse.Namespace(model_dir=specs[0]["model_dir"]))
    dataset = build_dataset(hypes0, visualize=True, train=False)
    anchor_args = hypes0["postprocess"]["anchor_args"]

    detections = []
    for spec in specs:
        print(f"[scene-unc] running {spec['label']} ...")
        if spec["type"] == "mc":
            _, det = run_mc(spec, args.frames, dataset, device, args.seed)
        else:
            _, det = run_method(spec, args.frames, dataset, device, args.seed)
        for d in det.values():
            d["sigma_c"] = np.array([
                np.hypot(*(lambda s: (s["x"], s["y"]))(
                    metric_sigmas(v, anchor_args))) for v in d["var"]])
        detections.append(det)

    (x0, x1), (y0, y1) = args.x_range, args.y_range
    for frame in args.frames:
        pts, origins = cav_points(dataset, frame)
        sig_all = np.concatenate([det[frame]["sigma_c"] for det in detections])
        sig_all = sig_all[sig_all > 0]
        norm = LogNorm(vmin=np.percentile(sig_all, 2),
                       vmax=np.percentile(sig_all, 98))
        cmap = plt.get_cmap("plasma")

        ncol = args.ncols
        nrow = int(np.ceil(len(specs) / ncol))
        aspect = (y1 - y0) / (x1 - x0)
        fig, axs = plt.subplots(nrow, ncol,
                                figsize=(6.0 * ncol, 6.0 * aspect * nrow + 1.2),
                                squeeze=False, constrained_layout=True)
        for ax in axs.ravel()[len(specs):]:
            ax.axis("off")

        for ax, spec, det in zip(axs.ravel(), specs, detections):
            d = det[frame]
            for k, p in enumerate(pts):
                m = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
                ax.scatter(p[m, 0], p[m, 1], s=0.15, c=CAV_COLOURS[k % 2],
                           linewidths=0, alpha=0.5, zorder=1,
                           label="LiDAR, ego car" if k == 0 else "LiDAR, other car")
            for k, o in enumerate(origins):
                ax.plot(o[0], o[1], "*", color=CAV_COLOURS[k % 2], ms=11,
                        mec="black", mew=0.6, zorder=6)
            for i, g in enumerate(d["gt"]):
                gp = polygon(g)
                ax.plot(gp[:, 0], gp[:, 1], "--", color="black", lw=0.9,
                        zorder=3, label="Ground truth" if i == 0 else None)
            order = np.argsort(d["sigma_c"])
            for i in order:
                bp = polygon(d["boxes"][i])
                ax.plot(bp[:, 0], bp[:, 1], "-", lw=1.8, zorder=4,
                        color=cmap(norm(max(d["sigma_c"][i], norm.vmin))))
            med = np.median(d["sigma_c"]) if len(d["sigma_c"]) else float("nan")
            ax.set_title(f"{spec['label']}   ({len(d['boxes'])} detections, "
                         f"median $\\sigma_c$ {med:.2f} m)", fontsize=10)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            ax.set_xlabel("x (m)", fontsize=8)
            ax.set_ylabel("y (m)", fontsize=8)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        leg = axs[0, 0].legend(handles, labels, loc="upper left", fontsize=7,
                               framealpha=0.9, markerscale=12)
        leg.set_zorder(10)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axs.ravel().tolist(), orientation="horizontal",
                          fraction=0.03, pad=0.02, aspect=60)
        cb.set_label("Predicted box-centre uncertainty $\\sigma_c$ (m), "
                     "same scale in every panel", fontsize=9)
        fig.suptitle(f"{args.model_name} on V2V4Real test, frame {frame}: detections "
                     "coloured by uncertainty (dashed = ground truth, "
                     "stars = the two cars)", fontsize=12)
        out = os.path.join(args.output_dir, f"scene_uncertainty_frame{frame:05d}.png")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        print(f"[scene-unc] saved {out}")


if __name__ == "__main__":
    main()

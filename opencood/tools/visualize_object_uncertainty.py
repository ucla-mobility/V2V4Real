"""
Object-centric spatial uncertainty (objectness-probability contours).

For a few detected objects, show how each method's predictive box
distribution spreads around the object in bird's-eye view, as in Fig. 3 of
Yun & Liu, "Laplace Approximation Based Epistemic Uncertainty Estimation in
3D Object Detection", CoRL 2022 (following Wang et al., "Inferring Spatial
Uncertainty in Object Detection", IROS 2020).

Each detected box is a distribution: mean = the predicted box, per-box
variance of the encoded targets [x, y, z, h, w, l, yaw] from the method's own
inference path (same code as the evaluation scripts; the variance of the
anchor that produced the box, carried through post_process_with_variance).
Boxes are sampled in metres:
    centre   + N(0, (d * sigma_x)^2), N(0, (d * sigma_y)^2)  d = anchor diagonal
    length   * exp(N(0, sigma_l^2)),  width * exp(N(0, sigma_w^2))
    yaw      + N(0, sigma_yaw^2)
and P(point inside the object) is the fraction of sampled boxes containing a
BEV point. Contours of P are drawn with the LiDAR points, the ground-truth box
and the predicted box.

Methods are given as repeated --method "key=value,..." strings:
    label=...           panel title
    type=vn|laplace     inference path
    model_dir=...       trained model folder (config with the V2V4Real test set)
    laplace_dir=...     (laplace) folder with laplace_diag.pt
    scope=heads|last_module|full   (laplace) must match compute_laplace.py
    part=aleatoric|epistemic|total (vn: aleatoric; laplace: epistemic or total)
    num_samples=200     (laplace, scopes other than heads) JVP samples

Objects are chosen automatically per frame from the --reference method
(nearest accurate, farthest, and worst-localised true positive; IoU >= 0.5),
or given with --object FRAME:X:Y. Each method shows its own detection nearest
to that position (within --match_radius m).
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics.uncertainty_matching_utils import bev_iou_matrix
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.inference_laplace import (
    aleatoric_variance,
    build_bayesian_output,
    laplace_forward_glm,
    laplace_forward_linearized,
    move_to_device,
)
from opencood.uncertainty.diagonal_laplace import DiagonalLaplace

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--method", action="append", required=True,
                   help="key=value,... (see module docstring). Repeat.")
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--object", action="append", default=[],
                   help="FRAME:X:Y — use this object instead of automatic choice.")
    p.add_argument("--reference", type=int, default=0,
                   help="Index of the --method used to choose objects.")
    p.add_argument("--output_dir", default="results/object_uncertainty")
    p.add_argument("--window", type=float, default=12.0,
                   help="Side length (m) of the square shown around the object.")
    p.add_argument("--res", type=float, default=0.05, help="Grid resolution (m).")
    p.add_argument("--box_samples", type=int, default=2000)
    p.add_argument("--match_radius", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_method(text):
    spec = dict(kv.split("=", 1) for kv in text.split(","))
    for key in ("label", "type", "model_dir", "part"):
        if key not in spec:
            raise ValueError(f"--method needs {key}=...: {text}")
    if spec["type"] == "laplace":
        spec.setdefault("scope", "heads")
        spec["num_samples"] = int(spec.get("num_samples", 200))
        if "laplace_dir" not in spec:
            raise ValueError(f"laplace method needs laplace_dir=...: {text}")
    return spec


def load_model(model_dir, device):
    hypes = load_yaml(None, argparse.Namespace(model_dir=model_dir))
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(model_dir, model)
    return hypes, model.to(device).eval()


# ---------------------------------------------------------------------------
# Per-method detections with per-box variance [N, 7] (encoded units)
# ---------------------------------------------------------------------------

def run_method(spec, frames, dataset, device, seed):
    hypes, model = load_model(spec["model_dir"], device)
    loss_args = hypes.get("loss", {}).get("args", {})
    lv_min = loss_args.get("log_var_min", -7.0)
    lv_max = loss_args.get("log_var_max", 7.0)

    laplace = None
    if spec["type"] == "laplace":
        laplace = DiagonalLaplace(model, param_scope=spec["scope"], device=device)
        laplace.load(os.path.join(spec["laplace_dir"], "laplace_diag.pt"))
        laplace.set_posterior(alpha=1.0,
                              prior_precision=laplace.optimize_prior_precision())

    results = {}
    for frame in frames:
        batch = move_to_device(
            dataset.collate_batch_test([dataset[frame]]), device)
        ego = batch["ego"]

        if spec["type"] == "vn":
            if spec["part"] != "aleatoric":
                raise ValueError("vn methods only have part=aleatoric")
            with torch.no_grad():
                out = model(ego)
            if "rm_logvar" not in out:
                raise ValueError(f"{spec['label']}: model has no variance head")
            rm_var = torch.exp(out["rm_logvar"].clamp(lv_min, lv_max))
            output = {"psm": out["psm"], "rm": out["rm"], "rm_var": rm_var}
        else:
            generator = torch.Generator(device=device).manual_seed(seed + frame)
            if spec["scope"] == "heads":
                result = laplace_forward_glm(laplace, ego)
            else:
                result = laplace_forward_linearized(
                    laplace, ego, spec["num_samples"], generator)
            rm_var = result["regression"]["variance"]
            if spec["part"] == "total":
                aleatoric = aleatoric_variance(
                    result, laplace, rm_var.shape, lv_min, lv_max)
                rm_var = rm_var + aleatoric
            elif spec["part"] != "epistemic":
                raise ValueError("laplace methods: part=epistemic or total")
            output = dict(build_bayesian_output(result), rm_var=rm_var)

        with torch.no_grad():
            boxes, scores, var, gt = dataset.post_process_with_variance(
                batch, {"ego": output})
        pcd = ego["origin_lidar"][0]
        results[frame] = {
            "boxes": _np(boxes, (0, 8, 3)),
            "scores": _np(scores, (0,)),
            "var": _np(var, (0, 7)),
            "gt": _np(gt, (0, 8, 3)),
            "pcd": pcd.cpu().numpy() if torch.is_tensor(pcd) else np.asarray(pcd),
        }

    del laplace, model
    torch.cuda.empty_cache()
    return hypes, results


def _np(x, empty_shape):
    if x is None:
        return np.zeros(empty_shape, dtype=np.float32)
    return x.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _bev_corners(box):
    """The 4 BEV corners of an (8, 3) corner box, ordered around the centre."""
    xy = box[:4, :2].astype(np.float64)
    c = xy.mean(axis=0)
    return xy[np.argsort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))], c


def footprint(box):
    """BEV footprint of an (8, 3) corner box: centre, unit long axis, L, W."""
    xy, c = _bev_corners(box)
    e1, e2 = xy[1] - xy[0], xy[3] - xy[0]
    if np.linalg.norm(e1) < np.linalg.norm(e2):
        e1, e2 = e2, e1
    length, width = np.linalg.norm(e1), np.linalg.norm(e2)
    return c, e1 / length, length, width


def polygon(box):
    xy, _ = _bev_corners(box)
    return np.vstack([xy, xy[:1]])


def metric_sigmas(var7, anchor_args):
    """Encoded per-box variance -> std of centre (m), log-size, yaw (rad)."""
    d = np.hypot(anchor_args["l"], anchor_args["w"])
    sd = np.sqrt(np.maximum(var7, 0.0))
    return {"x": d * sd[0], "y": d * sd[1], "w": sd[4], "l": sd[5], "yaw": sd[6]}


def objectness_map(box, sig, gx, gy, n_samples, rng, device):
    """P(point inside the object) on the grid (gx, gy) by box sampling."""
    c, u, length, width = footprint(box)
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1),
                       dtype=torch.float32, device=device)
    inside = torch.zeros(pts.shape[0], device=device)
    chunk = 250
    for start in range(0, n_samples, chunk):
        n = min(chunk, n_samples - start)
        e = torch.tensor(rng.standard_normal((n, 5)), dtype=torch.float32,
                         device=device)
        cx = c[0] + sig["x"] * e[:, 0]
        cy = c[1] + sig["y"] * e[:, 1]
        half_l = 0.5 * length * torch.exp(sig["l"] * e[:, 2])
        half_w = 0.5 * width * torch.exp(sig["w"] * e[:, 3])
        ang = float(np.arctan2(u[1], u[0])) + sig["yaw"] * e[:, 4]
        ca, sa = torch.cos(ang), torch.sin(ang)
        dx = pts[None, :, 0] - cx[:, None]
        dy = pts[None, :, 1] - cy[:, None]
        along = dx * ca[:, None] + dy * sa[:, None]
        across = -dx * sa[:, None] + dy * ca[:, None]
        inside += ((along.abs() <= half_l[:, None])
                   & (across.abs() <= half_w[:, None])).float().sum(0)
    return (inside / n_samples).cpu().numpy().reshape(gx.shape)


# ---------------------------------------------------------------------------
# Object selection
# ---------------------------------------------------------------------------

def true_positives(det):
    """Greedy AP-style matching (IoU >= 0.5): list of (pred_idx, gt_idx)."""
    if len(det["boxes"]) == 0 or len(det["gt"]) == 0:
        return []
    ious = bev_iou_matrix(det["boxes"], det["gt"])
    used, pairs = set(), []
    for i in np.argsort(-det["scores"]):
        free = [j for j in range(len(det["gt"])) if j not in used]
        if not free:
            break
        j = free[int(np.argmax(ious[i, free]))]
        if ious[i, j] >= 0.5:
            used.add(j)
            pairs.append((int(i), int(j)))
    return pairs


def choose_objects(det, others=(), radius=2.0):
    """Nearest accurate, farthest, and worst-localised true positive of the
    reference method, preferring objects that every other method also
    detected (a detection within `radius` m), so that no panel is empty."""
    rows = []
    for i, j in true_positives(det):
        c = footprint(det["boxes"][i])[0]
        g = footprint(det["gt"][j])[0]
        if np.linalg.norm(c) < 3.0:
            # the ego vehicle itself (labelled by the other CAV) — skip
            continue
        rows.append((i, float(np.linalg.norm(c)), float(np.linalg.norm(c - g))))
    common = [r for r in rows
              if all(nearest(o["boxes"], footprint(det["boxes"][r[0]])[0],
                             radius) is not None for o in others)]
    rows = common or rows
    if not rows:
        return []
    near = [r for r in rows if r[1] < 25.0] or rows
    picks = [("near, accurate", min(near, key=lambda r: r[2])),
             ("far", max(rows, key=lambda r: r[1])),
             ("worst localised", max(rows, key=lambda r: r[2]))]
    chosen, seen = [], set()
    for name, (i, _, _) in picks:
        if i not in seen:
            seen.add(i)
            chosen.append((name, footprint(det["boxes"][i])[0]))
    return chosen


def nearest(boxes, xy, radius):
    if len(boxes) == 0:
        return None
    dist = [np.linalg.norm(footprint(b)[0] - xy) for b in boxes]
    k = int(np.argmin(dist))
    return k if dist[k] <= radius else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print(f"[obj-unc] running {spec['label']} ...")
        hypes, det = run_method(spec, args.frames, dataset, device, args.seed)
        a = hypes["postprocess"]["anchor_args"]
        if (a["l"], a["w"], a["h"]) != (anchor_args["l"], anchor_args["w"],
                                         anchor_args["h"]):
            raise ValueError(f"{spec['label']}: different anchor sizes")
        detections.append(det)

    manual = {}
    for text in args.object:
        f, x, y = text.split(":")
        manual.setdefault(int(f), []).append(
            ("manual", np.array([float(x), float(y)])))

    half = args.window / 2
    offs = np.arange(-half, half + 1e-9, args.res)
    csv_rows = []

    for frame in args.frames:
        ref = detections[args.reference][frame]
        others = [det[frame] for det in detections]
        objects = manual.get(frame) or choose_objects(ref, others,
                                                      args.match_radius)
        if not objects:
            print(f"[obj-unc] frame {frame}: no true positive to show")
            continue

        nrow, ncol = len(objects), len(specs)
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.4 * nrow),
                                squeeze=False, constrained_layout=True)
        mappable = None
        for r, (obj_name, xy) in enumerate(objects):
            gx, gy = np.meshgrid(xy[0] + offs, xy[1] + offs)
            dist_ego = float(np.linalg.norm(xy))
            for col, (spec, det) in enumerate(zip(specs, detections)):
                ax = axs[r, col]
                d = det[frame]
                pcd = d["pcd"]
                m = (np.abs(pcd[:, 0] - xy[0]) < half) & (np.abs(pcd[:, 1] - xy[1]) < half)
                ax.scatter(pcd[m, 0], pcd[m, 1], s=1.0, c="0.55", linewidths=0,
                           zorder=1)
                g = nearest(d["gt"], xy, args.match_radius + 2)
                if g is not None:
                    gp = polygon(d["gt"][g])
                    ax.plot(gp[:, 0], gp[:, 1], "--", color="black", lw=1.6,
                            zorder=4, label="Ground truth (real car)")
                k = nearest(d["boxes"], xy, args.match_radius)
                if k is None:
                    ax.text(0.5, 0.5, "no detection", transform=ax.transAxes,
                            ha="center", va="center")
                else:
                    sig = metric_sigmas(d["var"][k], anchor_args)
                    rng = np.random.default_rng(args.seed + 1000 * frame + r)
                    prob = objectness_map(d["boxes"][k], sig, gx, gy,
                                          args.box_samples, rng, device)
                    # light sky blue (unlikely) -> navy (certainly the car); no green, so the
                    # black ground-truth and red detection boxes stay visible
                    mappable = ax.contourf(gx, gy, prob, levels=[0.02] + LEVELS + [1.0],
                                           colors=["#bfe3ff", "#8cc8f5", "#5aa7e6",
                                                   "#3182cc", "#1a5ca8", "#0b3175"],
                                           alpha=0.9, zorder=2)
                    bp = polygon(d["boxes"][k])
                    ax.plot(bp[:, 0], bp[:, 1], "-", color="#e00000", lw=1.4,
                            zorder=5, label="Detection")
                    err = (float(np.linalg.norm(footprint(d["boxes"][k])[0]
                                                - footprint(d["gt"][g])[0]))
                           if g is not None else float("nan"))
                    csv_rows.append({
                        "frame": frame, "object": obj_name,
                        "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
                        "distance_m": round(dist_ego, 1), "method": spec["label"],
                        "score": round(float(d["scores"][k]), 3),
                        "centre_error_m": round(err, 3),
                        "sigma_x_m": round(sig["x"], 4), "sigma_y_m": round(sig["y"], 4),
                        "sigma_logl": round(sig["l"], 4), "sigma_logw": round(sig["w"], 4),
                        "sigma_yaw_deg": round(float(np.degrees(sig["yaw"])), 3),
                        "area_p_ge_0.1_m2": round(float((prob >= 0.1).sum() * args.res ** 2), 2),
                        "area_p_ge_0.9_m2": round(float((prob >= 0.9).sum() * args.res ** 2), 2),
                    })
                ax.set_xlim(xy[0] - half, xy[0] + half)
                ax.set_ylim(xy[1] - half, xy[1] + half)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(spec["label"], fontsize=9)
                if col == 0:
                    simple_name = {"near, accurate": "Near car", "far": "Far car",
                                   "worst localised": "Worst-placed car"}.get(obj_name, obj_name)
                    ax.set_ylabel(f"{simple_name} ({dist_ego:.0f} m away)\ny (m)",
                                  fontsize=8)
                if r == nrow - 1:
                    ax.set_xlabel("x (m)", fontsize=8)
        axs[0, 0].legend(loc="upper left", fontsize=6, framealpha=0.9)
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=axs, fraction=0.02, pad=0.01)
            cb.set_label("Probability that this spot belongs to the car")
        fig.suptitle(f"V2V4Real test frame {frame:05d}: where each method thinks the car could be\n"
                     f"(dark blue = sure it is the car; a wide light-blue area = unsure about the car's position and size)",
                     fontsize=11)
        out = os.path.join(args.output_dir, f"object_uncertainty_frame{frame:05d}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[obj-unc] saved {out}")

    if csv_rows:
        out = os.path.join(args.output_dir, "object_uncertainty_summary.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[obj-unc] saved {out}")


if __name__ == "__main__":
    main()

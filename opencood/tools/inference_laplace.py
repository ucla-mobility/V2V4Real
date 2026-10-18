"""
Bayesian inference for OpenCOOD using a diagonal Laplace approximation.

Workflow
--------
1. Load trained MAP OpenCOOD detector.
2. Load previously computed diagonal curvature.
3. Construct the Laplace posterior.
4. Draw posterior weight samples.
5. Run the detector for each sampled parameter vector.
6. Aggregate:
       - classification probabilities
       - classification epistemic uncertainty
       - regression predictions
       - regression uncertainty
7. Feed posterior-mean predictions into the normal OpenCOOD
   post-processing/evaluation pipeline.
"""

import argparse
import contextlib
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.uncertainty.diagonal_laplace import DiagonalLaplace
from opencood.models.sub_modules import fuse_utils
from opencood.models.sub_modules import torch_transformation_utils
from einops import rearrange
from metrics.calibration_utils import PaperCalibrationAccumulator


from metrics.uncertainty_matching_utils import match_error_uncertainty_pairs

def parse_args():

    parser = argparse.ArgumentParser(
        description="OpenCOOD inference with diagonal Laplace uncertainty."
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Directory containing the trained OpenCOOD checkpoint.",
    )

    parser.add_argument(
        "--laplace_dir",
        default=None,
        type=str,
        help=(
            "Directory containing Laplace artifacts and receiving Laplace "
            "inference outputs. Default: <model_dir>_laplace"
        ),
    )

    parser.add_argument(
        "--laplace_path",
        default=None,
        type=str,
        help=(
            "Saved curvature file. "
            "Default: <laplace_dir>/laplace_diag.pt"
        ),
    )

    parser.add_argument(
        "--param_scope",
        default="heads",
        choices=["heads", "last_module", "full"],
        help=(
            "'heads': cls_head/reg_head (last layer). 'last_module': last "
            "block before the heads + heads (F-Cooper / V2VNet / V2X-ViT: "
            "shrink_conv; AttFuse: backbone.deblocks). 'full': the complete "
            "detector. BatchNorm is never included. Must match compute_laplace.py."
        ),
    )
    parser.add_argument(
        "--scope_modules",
        default=None,
        type=str,
        help=(
            "Comma-separated module prefixes that override --param_scope. "
            "Must match compute_laplace.py."
        ),
    )

    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help=(
            "Laplace posterior covariance scaling (temperature). "
            "1.0 = standard Laplace posterior."
        ),
    )

    parser.add_argument(
        "--prior_precision",
        default="auto",
        type=str,
        help=(
            "Gaussian prior precision, or 'auto' to choose it by maximising "
            "the Laplace marginal likelihood on the training curvature."
        ),
    )

    parser.add_argument(
        "--prior_per_module",
        default=1,
        type=int,
        choices=[0, 1],
        help=(
            "With --prior_precision auto: 1 (default) = one prior precision "
            "per module (layer-wise), 0 = one scalar for all parameters."
        ),
    )

    parser.add_argument(
        "--regression_variance",
        default="total",
        choices=["total", "epistemic"],
        help=(
            "'total': epistemic (Laplace) + aleatoric variance. Aleatoric is "
            "exp(rm_logvar) for variance-network models, otherwise the "
            "training-set observation noise. 'epistemic': Laplace only."
        ),
    )

    parser.add_argument(
        "--predictive",
        default=None,
        choices=["glm", "linearized", "sample"],
        help=(
            "'glm': exact closed-form linearised predictive, only for "
            "--param_scope heads (outputs are linear in the head weights); "
            "default for heads. 'linearized': linearised (GLM) predictive "
            "for deeper scopes: mean = MAP output, variance from J*dtheta "
            "(forward-mode JVP) over --num_samples posterior weight samples; "
            "default for last_module and full (for full, each JVP runs "
            "through the whole detector). 'sample': plain Monte Carlo through the "
            "nonlinear network (weights perturbed in the forward pass)."
        ),
    )

    parser.add_argument(
        "--num_samples",
        default=20,
        type=int,
        help="Number of posterior Monte Carlo samples ('sample' mode only).",
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for the posterior weight samples ('sample' mode).",
    )


    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="cuda, cuda:0, cpu, etc.",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    return parser.parse_args()


def resolve_device(device_argument):

    if device_argument is not None:
        return torch.device(device_argument)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def move_to_device(data, device):

    if torch.is_tensor(data):
        return data.to(
            device,
            non_blocking=True,
        )

    if isinstance(data, dict):
        return {
            key: move_to_device(value, device)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            move_to_device(value, device)
            for value in data
        ]

    if isinstance(data, tuple):
        return tuple(
            move_to_device(value, device)
            for value in data
        )

    return data


# ==============================================================
# Classification uncertainty helpers
# ==============================================================

def binary_entropy(probability, eps=1e-8):

    p = probability.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return -(
        p * torch.log(p)
        +
        (1.0 - p)
        * torch.log(1.0 - p)
    )


def compute_classification_uncertainty(
    probability_samples,
):

    # probability_samples:
    #
    # [T, B, anchors, H, W]
    #
    # T = posterior samples

    mean_probability = probability_samples.mean(
        dim=0
    )

    probability_variance = probability_samples.var(
        dim=0,
        unbiased=False,
    )

    predictive_entropy = binary_entropy(
        mean_probability
    )

    expected_entropy = binary_entropy(
        probability_samples
    ).mean(
        dim=0
    )

    # Epistemic classification uncertainty.
    mutual_information = (
        predictive_entropy
        - expected_entropy
    ).clamp_min(
        0.0
    )

    return {
        "mean_probability":
            mean_probability,

        "variance":
            probability_variance,

        "predictive_entropy":
            predictive_entropy,

        "expected_entropy":
            expected_entropy,

        "mutual_information":
            mutual_information,
    }


# ==============================================================
# Regression uncertainty
# ==============================================================

def compute_regression_uncertainty(
    regression_samples,
):

    mean_regression = regression_samples.mean(
        dim=0
    )

    regression_variance = regression_samples.var(
        dim=0,
        unbiased=False,
    )

    regression_std = torch.sqrt(
        regression_variance.clamp_min(0.0)
    )

    return {
        "mean":
            mean_regression,

        "variance":
            regression_variance,

        "std":
            regression_std,
    }


# ==============================================================
# Probability -> logit
# ==============================================================

def probability_to_logit(
    probability,
    eps=1e-6,
):

    p = probability.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return torch.log(
        p / (1.0 - p)
    )


# ==============================================================
# Laplace posterior inference
# ==============================================================

@torch.no_grad()
def laplace_forward(
    laplace_model,
    ego_data,
    num_samples,
):

    if num_samples < 1:
        raise ValueError(
            "num_samples must be >= 1."
        )

    classification_samples = []
    regression_samples = []
    logvar_samples = []

    try:

        for sample_idx in range(
            num_samples
        ):

            # --------------------------------------------------
            # Draw theta ~ q(theta | D)
            # --------------------------------------------------

            laplace_model.sample_weights()

            # --------------------------------------------------
            # Detector forward pass
            # --------------------------------------------------

            output_dict = laplace_model.model(
                ego_data
            )

            classification_logits = (
                output_dict["psm"]
            )

            regression_output = (
                output_dict["rm"]
            )

            # --------------------------------------------------
            # Convert logits to probability before averaging.
            #
            # IMPORTANT:
            #
            #     E[sigmoid(z)] != sigmoid(E[z])
            #
            # so Bayesian classification should average the
            # probabilities from individual posterior samples.
            # --------------------------------------------------

            classification_probability = (
                torch.sigmoid(
                    classification_logits
                )
            )

            classification_samples.append(
                classification_probability.detach()
            )

            regression_samples.append(
                regression_output.detach()
            )

            # Variance-network models also predict aleatoric variance.
            if "rm_logvar" in output_dict:
                logvar_samples.append(
                    output_dict["rm_logvar"].detach()
                )

    finally:

        # Critical:
        #
        # Never leave a posterior sample installed in the detector.
        laplace_model.restore_map_weights()

    classification_samples = torch.stack(
        classification_samples,
        dim=0,
    )

    regression_samples = torch.stack(
        regression_samples,
        dim=0,
    )

    classification = (
        compute_classification_uncertainty(
            classification_samples
        )
    )

    regression = (
        compute_regression_uncertainty(
            regression_samples
        )
    )

    return {
        "classification":
            classification,

        "regression":
            regression,

        "logvar_samples":
            torch.stack(logvar_samples, dim=0) if logvar_samples else None,

        "classification_samples":
            classification_samples,

        "regression_samples":
            regression_samples,
    }


def laplace_forward_glm(
    laplace_model,
    ego_data,
):
    """
    Exact predictive for a diagonal Laplace posterior on the 1x1-conv heads.

    With MAP weights W, b and posterior std s_W, s_b, a head output
    y = W f + b (f = fused feature at one BEV cell) is Gaussian with
        mean     = W f + b             (the MAP output)
        variance = sum_k f_k^2 s_Wk^2 + s_b^2,
    i.e. a 1x1 convolution of f^2 with s_W^2. This is exact because the heads
    are linear in their weights (GLM predictive, Laplace Redux 2021), so it
    replaces weight sampling without any approximation and without sampling
    noise.

    Classification (all deterministic, from the exact logit variance v):
    - mean probability: probit approximation sigmoid(mu / sqrt(1 + pi/8 v))
      (MacKay 1992);
    - probability variance: delta method (p (1 - p))^2 v, p = sigmoid(mu);
    - mutual information (epistemic): second-order expansion
      MI = H(E[p]) - E[H(p)] ~= 0.5 p (1 - p) v  (nats). Checked against
      4e6-sample Monte Carlo: within 0.5% for v <= 0.01, 5% at v = 0.1.
    """
    model = laplace_model.model
    sigma = laplace_model.sigma
    for name in ("cls_head.weight", "cls_head.bias",
                 "reg_head.weight", "reg_head.bias"):
        if name not in sigma:
            raise ValueError(
                f"GLM predictive needs a posterior on {name} "
                "(use --param_scope heads)."
            )

    laplace_model.restore_map_weights()

    captured = {}

    def hook(_module, inputs, _output):
        captured["feature"] = inputs[0].detach()

    handle = model.cls_head.register_forward_hook(hook)
    try:
        output_dict = model(ego_data)
    finally:
        handle.remove()

    feature_sq = captured["feature"] ** 2

    def head_variance(prefix):
        var_w = sigma[prefix + ".weight"] ** 2
        var_b = sigma[prefix + ".bias"] ** 2
        return F.conv2d(feature_sq, var_w) + var_b.view(1, -1, 1, 1)

    logit_mean = output_dict["psm"].detach()
    logit_var = head_variance("cls_head")
    reg_mean = output_dict["rm"].detach()
    reg_var = head_variance("reg_head")

    classification = _classification_from_logit_moments(logit_mean, logit_var)

    logvar = output_dict.get("rm_logvar")

    return {
        "classification": classification,
        "regression": {
            "mean": reg_mean,
            "variance": reg_var,
            "std": reg_var.clamp_min(0.0).sqrt(),
        },
        "logvar_samples":
            logvar.detach().unsqueeze(0) if logvar is not None else None,
        "classification_samples": None,
        "regression_samples": None,
    }


def _classification_from_logit_moments(logit_mean, logit_var):
    """Deterministic classification uncertainty from Gaussian logits
    (same formulas as laplace_forward_glm)."""
    mean_probability = torch.sigmoid(
        logit_mean / torch.sqrt(1.0 + (np.pi / 8.0) * logit_var)
    )
    p_map = torch.sigmoid(logit_mean)
    bernoulli_var = p_map * (1.0 - p_map)
    predictive_entropy = binary_entropy(mean_probability)
    mutual_information = 0.5 * bernoulli_var * logit_var
    return {
        "mean_probability": mean_probability,
        "variance": bernoulli_var ** 2 * logit_var,
        "logit_variance": logit_var,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": predictive_entropy - mutual_information,
        "mutual_information": mutual_information,
    }


def laplace_forward_linearized(
    laplace_model,
    ego_data,
    num_samples,
    generator,
):
    """
    Linearised (GLM) predictive for a Laplace posterior on
    shrink_conv + heads (--param_scope last_module) of F-Cooper. Any other
    scope or architecture is passed on to laplace_forward_linearized_full.

    The network is linearised in the selected weights around the MAP,
        f_lin(theta) = f(theta_MAP) + J (theta - theta_MAP),
    so the predictive mean is exactly the MAP output (detections and AP are
    unchanged) and the output variance is diag(J Sigma J^T). It is estimated
    as mean_s (J dtheta_s)^2 with dtheta_s ~ N(0, Sigma), where each J dtheta_s
    is one forward-mode JVP through the network tail only
    (shrink_conv -> fusion -> heads); the backbone runs once.

    Plain weight sampling through the nonlinear network ('sample') breaks
    here: ~95% of the shrink_conv weights are ~0 and weakly determined by the
    data, so their posterior is the prior (std ~0.05); perturbing all of them
    at once corrupts the features (AP@0.5 0.462 -> 0.363). The linearised
    predictive is the standard remedy (Immer et al., AISTATS 2021;
    Daxberger et al., "Laplace Redux", NeurIPS 2021).
    """
    model = laplace_model.model
    names = laplace_model.param_names
    tail_modules = ("shrink_conv", "cls_head", "reg_head")
    fast_tail = (
        getattr(model, "shrink_flag", False)
        and not getattr(model, "compression", False)
        and type(getattr(model, "fusion_net", None)).__name__ == "SpatialFusion"
    )
    if not fast_tail or any(n.split(".")[0] not in tail_modules for n in names):
        # Not an F-Cooper tail scope (scope reaches into the backbone /
        # pillar encoder, e.g. --param_scope full, or a different
        # architecture such as AttFuse / V2VNet / V2X-ViT): linearise the
        # whole detector. Same estimator, one full forward per sample.
        return laplace_forward_linearized_full(
            laplace_model, ego_data, num_samples, generator)
    for n in names:
        if n.split(".")[0] not in tail_modules:
            raise ValueError(
                f"linearized predictive supports scopes within {tail_modules}; "
                f"got {n}"
            )

    laplace_model.restore_map_weights()

    captured = {}
    handle = model.shrink_conv.register_forward_hook(
        lambda _m, inputs, _o: captured.__setitem__("x", inputs[0].detach())
    )
    try:
        with torch.no_grad():
            output_dict = model(ego_data)
    finally:
        handle.remove()

    x = captured["x"]
    if type(model.fusion_net).__name__ != "SpatialFusion":
        raise ValueError("linearized predictive implements F-Cooper max fusion only")
    # Same as SpatialFusion.forward (max over each sample's CAVs); integer
    # split sizes because tensor_split with a tensor of indices has no
    # forward-mode AD support.
    cav_counts = [int(n) for n in ego_data["record_len"].tolist()]

    def max_fusion(feat):
        return torch.cat([xx.max(dim=0, keepdim=True)[0]
                          for xx in torch.split(feat, cav_counts)], dim=0)
    params = dict(model.named_parameters())
    primals = {n: params[n].detach() for n in names}

    def sub(prefix, p):
        return {k[len(prefix) + 1:]: v for k, v in p.items()
                if k.startswith(prefix + ".")}

    def tail(p):
        feat = torch.func.functional_call(
            model.shrink_conv, sub("shrink_conv", p), (x,))
        fused = max_fusion(feat)
        psm = torch.func.functional_call(model.cls_head, sub("cls_head", p), (fused,))
        rm = torch.func.functional_call(model.reg_head, sub("reg_head", p), (fused,))
        return psm, rm

    return _linearized_result(
        tail, primals, laplace_model, output_dict, num_samples, generator)


# Set once an op without forward-mode AD support is hit (e.g. in a fusion
# module of a new architecture); later JVPs then use the double-backward trick.
_FORWARD_AD_UNSUPPORTED = False

_FORWARD_AD_ERROR_HINTS = ("forward ad", "forward-mode", "forward mode", "jvp",
                           "dual", "tangent", "functorch", "data-dependent")


def _is_forward_ad_error(err):
    if isinstance(err, NotImplementedError):
        return True
    message = str(err).lower()
    return any(hint in message for hint in _FORWARD_AD_ERROR_HINTS)


def _jvp_double_backward(fn, primals, tangents):
    """
    J·tangent of fn (params dict -> (psm, rm)) at primals with reverse-mode
    AD only ("double-backward trick"): g(u) = J^T u is linear in u, so
    d<g(u), v>/du = J v. Works for every op that has a backward; gives the
    same values as torch.func.jvp, at the cost of a forward and two
    backward passes.
    """
    names = list(primals)
    with torch.enable_grad():
        inputs = [primals[n].detach().requires_grad_(True) for n in names]
        outputs = fn(dict(zip(names, inputs)))
        dummies = [torch.zeros_like(o, requires_grad=True) for o in outputs]
        grads = torch.autograd.grad(outputs, inputs, grad_outputs=dummies,
                                    create_graph=True, allow_unused=True)
        used = [(g, tangents[n]) for g, n in zip(grads, names) if g is not None]
        jv = torch.autograd.grad([g for g, _ in used], dummies,
                                 grad_outputs=[t for _, t in used],
                                 allow_unused=True)
    return tuple(torch.zeros_like(o).detach() if j is None else j.detach()
                 for o, j in zip(outputs, jv))


def _linearized_result(fn, primals, laplace_model, output_dict, num_samples,
                       generator):
    """
    Shared part of the linearised predictive: output variance
    diag(J Sigma J^T) = E[(J dtheta)^2], dtheta ~ N(0, Sigma), estimated with
    num_samples forward-mode JVPs of fn (params -> (psm, rm)) at the MAP.
    The predictive mean is the MAP output (output_dict).
    """
    global _FORWARD_AD_UNSUPPORTED
    names = laplace_model.param_names
    sigma = laplace_model.sigma
    logit_var = torch.zeros_like(output_dict["psm"])
    reg_var = torch.zeros_like(output_dict["rm"])
    with torch.no_grad():
        for _ in range(num_samples):
            tangents = {
                n: sigma[n] * torch.randn(
                    primals[n].shape, device=primals[n].device,
                    generator=generator)
                for n in names
            }
            if not _FORWARD_AD_UNSUPPORTED:
                try:
                    _, (t_psm, t_rm) = torch.func.jvp(fn, (primals,), (tangents,))
                except (NotImplementedError, RuntimeError) as err:
                    if not _is_forward_ad_error(err):
                        raise
                    _FORWARD_AD_UNSUPPORTED = True
                    print(
                        "[Laplace] An op in this model has no forward-mode AD "
                        f"support ({type(err).__name__}: {str(err)[:200]}). "
                        "Using the double-backward JVP instead (same values, "
                        "slower)."
                    )
            if _FORWARD_AD_UNSUPPORTED:
                t_psm, t_rm = _jvp_double_backward(fn, primals, tangents)
            logit_var += t_psm.pow(2)
            reg_var += t_rm.pow(2)
    logit_var /= num_samples
    reg_var /= num_samples

    logvar = output_dict.get("rm_logvar")
    return {
        "classification": _classification_from_logit_moments(
            output_dict["psm"].detach(), logit_var),
        "regression": {
            "mean": output_dict["rm"].detach(),
            "variance": reg_var,
            "std": reg_var.sqrt(),
        },
        "logvar_samples":
            logvar.detach().unsqueeze(0) if logvar is not None else None,
        "classification_samples": None,
        "regression_samples": None,
    }


class _GridSampleFixedGrid(torch.autograd.Function):
    """
    F.grid_sample(x, grid) as a function of x only. The sampling grid comes
    from the agent poses (no parameters, no tangent), so the map is linear in
    x: J·t = grid_sample(t, grid), and the backward is its adjoint. PyTorch
    has neither a forward-mode rule nor a double backward for grid_sampler_2d
    (V2X-ViT STTF / V2VNet warp_affine); this gives both.
    """

    @staticmethod
    def forward(x, grid, kwargs):
        return F.grid_sample(x, grid, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, grid, kwargs = inputs
        ctx.save_for_backward(grid)
        ctx.save_for_forward(grid)
        ctx.x_shape = x.shape
        ctx.kwargs = kwargs

    @staticmethod
    def backward(ctx, grad_out):
        grid, = ctx.saved_tensors
        return (_GridSampleAdjoint.apply(grad_out, grid, ctx.x_shape,
                                         ctx.kwargs), None, None)

    @staticmethod
    def jvp(ctx, x_tangent, grid_tangent, _):
        # grid is detached by the caller: its tangent is None or zero.
        grid, = ctx.saved_tensors
        return F.grid_sample(x_tangent, grid, **ctx.kwargs)


class _GridSampleAdjoint(torch.autograd.Function):
    """Adjoint (transpose) of _GridSampleFixedGrid; also linear."""

    @staticmethod
    def forward(grad_out, grid, x_shape, kwargs):
        with torch.enable_grad():
            x0 = grad_out.new_zeros(x_shape, requires_grad=True)
            out = F.grid_sample(x0, grid.detach(), **kwargs)
            grad_x, = torch.autograd.grad(out, x0, grad_out.detach())
        return grad_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, grid, x_shape, kwargs = inputs
        ctx.save_for_backward(grid)
        ctx.save_for_forward(grid)
        ctx.x_shape = x_shape
        ctx.kwargs = kwargs

    @staticmethod
    def backward(ctx, grad_grad):
        grid, = ctx.saved_tensors
        return (_GridSampleFixedGrid.apply(grad_grad, grid, ctx.kwargs),
                None, None, None)

    @staticmethod
    def jvp(ctx, g_tangent, grid_tangent, _shape, _kwargs):
        grid, = ctx.saved_tensors
        return _GridSampleAdjoint.apply(g_tangent, grid, ctx.x_shape,
                                        ctx.kwargs)


class _FunctionalWithLinearGridSample:
    """torch.nn.functional with grid_sample replaced by _GridSampleFixedGrid."""

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                    align_corners=None):
        kwargs = dict(mode=mode, padding_mode=padding_mode,
                      align_corners=align_corners)
        return _GridSampleFixedGrid.apply(x, grid.detach(), kwargs)


@contextlib.contextmanager
def _forward_ad_compatible(model, cav_counts):
    """
    Temporarily swaps steps that have no forward-mode AD support
    (torch.func.jvp) for mathematically identical versions. Model files are
    not changed; the originals are restored on exit.

    - PointPillarScatter: the in-place write spatial_feature[:, idx] = pillars
      becomes the out-of-place zeros(...).index_copy(1, idx, pillars).
    - regroup of SpatialFusion (F-Cooper), AttFusion (AttFuse, one per
      backbone scale) and V2VNetFusion (V2VNet): tensor_split with a tensor of indices becomes
      torch.split with the integer CAV counts of this batch (cav_counts,
      read from record_len before the transform; reading tensor values
      inside torch.func.jvp is not allowed).
    - fuse_utils.regroup (module-level function used by CoBEVT and V2X-ViT to
      pad every sample to max_cav agents): same split with integer CAV counts,
      zero padding and mask as the original.
    - F.grid_sample inside torch_transformation_utils.warp_affine (V2X-ViT
      STTF, V2VNet): _GridSampleFixedGrid, same values, with a forward-mode
      rule and a double backward.
    Any other unsupported op makes torch.func.jvp raise, and
    _linearized_result then falls back to the double-backward JVP.
    """
    scatter = getattr(model, "scatter", None)
    if type(scatter).__name__ != "PointPillarScatter":
        scatter = None
    fusions = [m for m in model.modules()
               if type(m).__name__ in ("SpatialFusion", "AttFusion", "V2VNetFusion")
               and hasattr(m, "regroup")]

    def scatter_forward(batch_dict):
        pillar_features = batch_dict["pillar_features"]
        coords = batch_dict["voxel_coords"]
        batch_size = coords[:, 0].max().int().item() + 1
        batch_spatial_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = (this_coords[:, 1]
                       + this_coords[:, 2] * scatter.nx
                       + this_coords[:, 3]).type(torch.long)
            pillars = pillar_features[batch_mask, :].t()
            spatial_feature = torch.zeros(
                scatter.num_bev_features,
                scatter.nz * scatter.nx * scatter.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device,
            ).index_copy(1, indices, pillars)
            batch_spatial_features.append(spatial_feature)
        batch_dict["spatial_features"] = torch.stack(
            batch_spatial_features, 0).view(
            batch_size, scatter.num_bev_features * scatter.nz,
            scatter.ny, scatter.nx)
        return batch_dict

    def regroup(x, record_len):
        assert len(cav_counts) == len(record_len)
        return torch.split(x, cav_counts)

    model_module = sys.modules[type(model).__module__]
    original_padded_regroup = getattr(model_module, "regroup", None)
    patch_padded = original_padded_regroup is fuse_utils.regroup

    def padded_regroup(dense_feature, record_len, max_len):
        # Same as fuse_utils.regroup: split per sample, zero-pad to max_len
        # agents, stack as (B, L, C, H, W), mask 1 = real agent.
        assert len(cav_counts) == len(record_len)
        features, mask = [], []
        for feature in torch.split(dense_feature, cav_counts):
            padding = max_len - feature.shape[0]
            mask.append([1] * feature.shape[0] + [0] * padding)
            feature = torch.cat(
                [feature, feature.new_zeros((padding,) + tuple(feature.shape[1:]))], dim=0)
            features.append(feature.reshape(-1, feature.shape[2], feature.shape[3]).unsqueeze(0))
        features = rearrange(torch.cat(features, dim=0), "b (l c) h w -> b l c h w", l=max_len)
        return features, torch.tensor(mask, device=features.device)

    if scatter is not None:
        scatter.forward = scatter_forward
    for fusion in fusions:
        fusion.regroup = regroup
    if patch_padded:
        model_module.regroup = padded_regroup
    torch_transformation_utils.F = _FunctionalWithLinearGridSample()
    try:
        yield
    finally:
        if scatter is not None:
            del scatter.forward
        for fusion in fusions:
            del fusion.regroup
        if patch_padded:
            model_module.regroup = original_padded_regroup
        torch_transformation_utils.F = F


def laplace_forward_linearized_full(
    laplace_model,
    ego_data,
    num_samples,
    generator,
):
    """
    Linearised (GLM) predictive when the Laplace scope reaches into the
    backbone / pillar encoder (--param_scope full). Same estimator as
    laplace_forward_linearized, but each JVP runs through the whole detector
    (F-Cooper: pillar encoder -> scatter -> backbone -> shrink_conv -> fusion
    -> heads; any other architecture: its own complete forward pass),
    so it costs about one extra forward pass per sample.
    """
    model = laplace_model.model
    laplace_model.restore_map_weights()
    with torch.no_grad():
        output_dict = model(ego_data)

    names = laplace_model.param_names
    params = dict(model.named_parameters())
    primals = {n: params[n].detach() for n in names}

    def full(p):
        out = torch.func.functional_call(model, p, (ego_data,))
        return out["psm"], out["rm"]

    cav_counts = [int(n) for n in ego_data["record_len"].tolist()]
    with _forward_ad_compatible(model, cav_counts):
        return _linearized_result(
            full, primals, laplace_model, output_dict, num_samples, generator)


# ==============================================================
# Aleatoric (observation-noise) variance
# ==============================================================

def aleatoric_variance(laplace_result, laplace_model, reg_shape,
                       logvar_min, logvar_max):
    """
    Aleatoric variance in the dense rm layout [B, 7*A, H, W].

    Variance-network model: mean over posterior samples of exp(rm_logvar),
    clamped exactly like in training. Otherwise: the homoscedastic
    per-dimension noise estimated from MAP training residuals.
    Returns None if neither is available.
    """
    logvar_samples = laplace_result["logvar_samples"]
    if logvar_samples is not None:
        return torch.exp(
            logvar_samples.clamp(logvar_min, logvar_max)
        ).mean(dim=0)

    noise_var = laplace_model.noise_var
    if noise_var is None:
        return None

    # rm channels are anchor-major: channel = anchor * 7 + dim
    n_anchor = reg_shape[1] // 7
    return noise_var.repeat(n_anchor).view(1, -1, 1, 1).expand(reg_shape)


# ==============================================================
# Build OpenCOOD-compatible posterior-mean output
# ==============================================================

def build_bayesian_output(
    laplace_result,
):

    mean_probability = (
        laplace_result[
            "classification"
        ][
            "mean_probability"
        ]
    )

    mean_regression = (
        laplace_result[
            "regression"
        ][
            "mean"
        ]
    )

    # OpenCOOD postprocessor expects psm to contain logits.
    #
    # We have the posterior predictive probability:
    #
    #     p_bar = E_q[sigmoid(z_theta)]
    #
    # Therefore construct an equivalent logit:
    #
    #     logit(p_bar)
    #
    # such that:
    #
    #     sigmoid(logit(p_bar)) = p_bar

    bayesian_psm = probability_to_logit(
        mean_probability
    )

    return {
        "psm":
            bayesian_psm,

        "rm":
            mean_regression,
    }


# ==============================================================
# Evaluation helpers (kept aligned with MC/VN protocol)
# ==============================================================


def box_centers(box_tensor):
    """Center of OpenCOOD corner boxes [N,8,3]."""
    return box_tensor.mean(dim=1).detach().cpu().numpy()


# ==============================================================
# Main
# ==============================================================

def main():

    args = parse_args()

    # The original training directory is only used to load config/MAP weights.
    # Laplace artifacts live in a separate sibling directory.
    if args.laplace_dir is None:
        args.laplace_dir = args.model_dir.rstrip(os.sep) + "_laplace"

    args.laplace_dir = os.path.abspath(args.laplace_dir)
    os.makedirs(args.laplace_dir, exist_ok=True)

    # Create standard locations now so later evaluation/visualization code can
    # write there without polluting the original training directory.
    for subdir in ("results", "uncertainty", "npy"):
        os.makedirs(os.path.join(args.laplace_dir, subdir), exist_ok=True)

    print(f"[Laplace] Source model directory: {os.path.abspath(args.model_dir)}")
    print(f"[Laplace] Laplace directory:      {args.laplace_dir}")

    device = resolve_device(
        args.device
    )

    print(
        f"\n[Laplace] Device: {device}"
    )

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    hypes = load_yaml(
        None,
        args
    )

    # ----------------------------------------------------------
    # Test dataset
    # ----------------------------------------------------------

    print(
        "\n[Laplace] Building evaluation dataset..."
    )

    test_dataset = build_dataset(
        hypes,
        visualize=True,
        train=False,
    )

    if hasattr(
        test_dataset,
        "collate_batch_test",
    ):

        collate_fn = (
            test_dataset.collate_batch_test
        )

    else:

        collate_fn = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    # ----------------------------------------------------------
    # MAP detector
    # ----------------------------------------------------------

    print(
        "\n[Laplace] Loading MAP detector..."
    )

    model = train_utils.create_model(
        hypes
    )

    model = model.to(
        device
    )

    _, model = train_utils.load_saved_model(
        args.model_dir,
        model,
    )

    model = model.to(
        device
    )

    model.eval()

    # ----------------------------------------------------------
    # Parameter scope
    # ----------------------------------------------------------

    param_scope = args.param_scope
    heads_only = param_scope == "heads" and not args.scope_modules
    if args.predictive is None:
        args.predictive = "glm" if heads_only else "linearized"
    if args.predictive == "glm" and not heads_only:
        raise ValueError(
            "--predictive glm is exact only for --param_scope heads; "
            "use --predictive sample."
        )
    torch.manual_seed(args.seed)

    # ----------------------------------------------------------
    # Laplace object
    # ----------------------------------------------------------

    laplace_model = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device,
        scope_modules=args.scope_modules,
    )

    # ----------------------------------------------------------
    # Load curvature
    # ----------------------------------------------------------

    if args.laplace_path is None:

        laplace_path = os.path.join(
            args.laplace_dir,
            "laplace_diag.pt",
        )

    else:

        laplace_path = (
            args.laplace_path
        )

    print(
        "\n[Laplace] Loading curvature:"
    )

    print(
        laplace_path
    )

    laplace_model.load(
        laplace_path
    )

    # ----------------------------------------------------------
    # Construct posterior
    # ----------------------------------------------------------

    if args.prior_precision == "auto":
        prior_precision = laplace_model.optimize_prior_precision(
            alpha=args.alpha,
            per_module=bool(args.prior_per_module),
        )
    else:
        prior_precision = float(args.prior_precision)

    laplace_model.set_posterior(
        alpha=args.alpha,
        prior_precision=prior_precision,
    )

    loss_args = hypes.get("loss", {}).get("args", {})
    logvar_min = loss_args.get("log_var_min", -7.0)
    logvar_max = loss_args.get("log_var_max", 7.0)
    warned_no_aleatoric = False

    print(
        "\n[Laplace] Posterior ready."
    )

    if args.predictive == "glm":
        print("[Laplace] Predictive: exact closed-form (GLM) for the heads")
    elif args.predictive == "linearized":
        print(
            f"[Laplace] Predictive: linearised (GLM), variance from "
            f"{args.num_samples} JVP samples; mean = MAP"
        )
        jvp_generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        print(
            f"[Laplace] Predictive: Monte Carlo, {args.num_samples} samples"
        )

    # ----------------------------------------------------------
    # Evaluation state
    # ----------------------------------------------------------

    cav_lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
    result_stat = {0.5: {"tp": [], "fp": [], "gt": 0},
                   0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_short = {0.5: {"tp": [], "fp": [], "gt": 0},
                         0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_middle = {0.5: {"tp": [], "fp": [], "gt": 0},
                          0.3: {"tp": [], "fp": [], "gt": 0}}
    result_stat_long = {0.5: {"tp": [], "fp": [], "gt": 0},
                        0.3: {"tp": [], "fp": [], "gt": 0}}

    prediction_errors = []
    prediction_variances = []
    # Feng et al. paper-style dense classification + marginal regression ECE.
    paper_calibration = PaperCalibrationAccumulator(n_cls_bins=10, n_reg_levels=10)
    visualization_npz_dir = os.path.join(args.laplace_dir, "visualization_npz_laplace")
    os.makedirs(visualization_npz_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    for batch_idx, batch_data in enumerate(
        test_loader
    ):

        batch_data = move_to_device(
            batch_data,
            device,
        )

        ego_data = batch_data[
            "ego"
        ]

        # ------------------------------------------------------
        # Bayesian forward pass
        # ------------------------------------------------------

        if args.predictive == "glm":
            result = laplace_forward_glm(
                laplace_model,
                ego_data,
            )
        elif args.predictive == "linearized":
            result = laplace_forward_linearized(
                laplace_model,
                ego_data,
                args.num_samples,
                jvp_generator,
            )
        else:
            result = laplace_forward(
                laplace_model,
                ego_data,
                args.num_samples,
            )

        # ------------------------------------------------------
        # Posterior predictive mean in normal OpenCOOD format
        # ------------------------------------------------------

        bayesian_output = (
            build_bayesian_output(
                result
            )
        )

        # ------------------------------------------------------
        # Uncertainty tensors
        # ------------------------------------------------------

        cls_variance = result[
            "classification"
        ][
            "variance"
        ]

        predictive_entropy = result[
            "classification"
        ][
            "predictive_entropy"
        ]

        mutual_information = result[
            "classification"
        ][
            "mutual_information"
        ]

        epistemic_reg_variance = result[
            "regression"
        ][
            "variance"
        ]

        reg_variance = epistemic_reg_variance
        if args.regression_variance == "total":
            aleatoric = aleatoric_variance(
                result, laplace_model, epistemic_reg_variance.shape,
                logvar_min, logvar_max,
            )
            if aleatoric is not None:
                reg_variance = epistemic_reg_variance + aleatoric
            elif not warned_no_aleatoric:
                print(
                    "[Laplace] WARNING: no aleatoric variance available "
                    "(old curvature file) -> using epistemic only."
                )
                warned_no_aleatoric = True

        # ------------------------------------------------------
        # Paper-style calibration: dense classification labels and
        # positive-anchor encoded regression targets. No 2 m matching.
        # ------------------------------------------------------
        paper_calibration.update(
            cls_prob=result["classification"]["mean_probability"],
            reg_mean=result["regression"]["mean"],
            reg_var=reg_variance,
            batch_data=batch_data,
        )

        # ------------------------------------------------------
        # Standard OpenCOOD post-processing of posterior mean
        # ------------------------------------------------------
        # The total regression variance is carried through the same
        # thresholding / NMS / range filter as the boxes, so each final box
        # gets the variance of the anchor that produced it (same protocol as
        # inference_vn.py). Boxes and scores are identical to post_process.
        output_dict_mean = {"ego": dict(bayesian_output, rm_var=reg_variance)}
        pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor = \
            test_dataset.post_process_with_variance(batch_data, output_dict_mean)

        # ------------------------------------------------------
        # Regression uncertainty-quality pairs.
        # Shared one-to-one AP-style BEV-IoU matching (IoU >= 0.5).
        # ------------------------------------------------------
        match_result = match_error_uncertainty_pairs(
            pred_box_tensor, pred_score, pred_var_tensor, gt_box_tensor,
            iou_threshold=0.5,
            anchor_args=hypes["postprocess"]["anchor_args"])
        prediction_errors.extend(match_result["errors_m"].tolist())
        prediction_variances.extend(match_result["uncertainties"].tolist())

        # ------------------------------------------------------
        # Standardized per-frame NPZ for plot_heatmap.py
        # ------------------------------------------------------
        pcd = batch_data["ego"].get("origin_lidar", [np.empty((0, 4), dtype=np.float32)])[0]
        if torch.is_tensor(pcd):
            pcd = pcd.detach().cpu().numpy()
        pred_np = (pred_box_tensor.detach().cpu().numpy()
                   if pred_box_tensor is not None else np.empty((0, 8, 3), dtype=np.float32))
        gt_np = (gt_box_tensor.detach().cpu().numpy()
                 if gt_box_tensor is not None else np.empty((0, 8, 3), dtype=np.float32))
        score_np = (pred_score.detach().cpu().numpy()
                    if pred_score is not None else np.empty((0,), dtype=np.float32))
        np.savez_compressed(
            os.path.join(visualization_npz_dir, f"{batch_idx:05d}.npz"),
            method=np.asarray("Diagonal Laplace"),
            pcd=np.asarray(pcd),
            gt_boxes=gt_np,
            pred_boxes=pred_np,
            pred_scores=score_np,
            classification_uncertainty=mutual_information.detach().cpu().numpy(),
            mutual_information=mutual_information.detach().cpu().numpy(),
            predictive_entropy=predictive_entropy.detach().cpu().numpy(),
            cls_variance=cls_variance.detach().cpu().numpy(),
            **({"cls_logit_variance": result["classification"]
                ["logit_variance"].detach().cpu().numpy()}
               if "logit_variance" in result["classification"] else {}),
            regression_uncertainty=reg_variance.detach().cpu().numpy(),
            reg_variance=reg_variance.detach().cpu().numpy(),
            epistemic_reg_variance=epistemic_reg_variance.detach().cpu().numpy(),
            cav_lidar_range=np.asarray(cav_lidar_range, dtype=np.float32),
            num_samples=np.asarray(
                args.num_samples if args.predictive != "glm" else 0,
                dtype=np.int32),
            predictive=np.asarray(args.predictive),
        )

        # ------------------------------------------------------
        # AP evaluation. Use the baseline/VN dictionary convention here:
        # IoU 0.3 and 0.5, including the same distance ranges.
        # ------------------------------------------------------
        for thr in (0.5, 0.3):
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, thr
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_short, thr,
                left_range=0, right_range=30
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_middle, thr,
                left_range=30, right_range=50
            )
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat_long, thr,
                left_range=50, right_range=100
            )

        if batch_idx % 50 == 0:
            print(
                f"[Laplace] Processed {batch_idx + 1}/{len(test_loader)}"
            )

    # ----------------------------------------------------------
    # Save the same metric CSV protocol as MC/VN
    # ----------------------------------------------------------
    err_var_csv = os.path.join(args.laplace_dir, "error_vs_variance_laplace.csv")
    with open(err_var_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_m", "mean_variance"])
        writer.writerows(zip(prediction_errors, prediction_variances))
    print(f"[Laplace] Saved {len(prediction_errors)} matched error/variance pairs: {err_var_csv}")

    # Canonical paper-style classification + regression ECE.
    paper_calibration.save(args.laplace_dir, "diagonal_laplace")

    # Standard OpenCOOD AP summaries, kept inside laplace_dir.
    eval_utils.eval_final_results(result_stat, args.laplace_dir)
    eval_utils.eval_final_results(result_stat_short, args.laplace_dir, "short")
    eval_utils.eval_final_results(result_stat_middle, args.laplace_dir, "middle")
    eval_utils.eval_final_results(result_stat_long, args.laplace_dir, "long")

    print(
        "\n[Laplace] Inference complete."
    )


if __name__ == "__main__":
    main()
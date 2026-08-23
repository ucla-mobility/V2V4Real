"""OpenCOOD <-> laplace-torch adapter for PointPillar detection heads.

This implementation fits ONE diagonal laplace-torch posterior PER ANCHOR
for the classification and regression heads.

Classification:
    Uses valid anchors only:
        positive OR negative anchors.
    Don't-care anchors are excluded.

Regression:
    Uses positive anchors only.
    OpenCOOD regression targets are expected to be packed as A*7 channels,
    for example:
        [B, H, W, 14]
    when A=2.

Laplace:
    Uses official laplace-torch Laplace(...).

Inference:
    Monte-Carlo weight sampling from the fitted posterior.

Alpha scaling:
    theta_alpha =
        theta_MAP + sqrt(alpha) * (theta_sample - theta_MAP)

Therefore:
    Cov(theta_alpha) = alpha * Cov(theta_sample)

alpha = 0:
    deterministic MAP head

alpha = 1:
    native laplace-torch posterior
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from laplace import Laplace


# ================================================================
# Small models used by laplace-torch
# ================================================================

class _TwoClassLinear(nn.Module):
    """Binary objectness represented as a two-class classification model."""

    def __init__(self, in_features: int):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):

        z = self.linear(x).squeeze(-1)

        # Equivalent binary logits represented as two classes.
        return torch.stack(
            (
                -0.5 * z,
                0.5 * z
            ),
            dim=-1
        )


class _RegressionLinear(nn.Module):
    """Linear regression model for one anchor's 7 encoded box parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int = 7
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x):

        return self.linear(x)


# ================================================================
# Validation helpers
# ================================================================

def _require_1x1(
    conv: nn.Conv2d,
    name: str
):

    if tuple(conv.kernel_size) != (1, 1):

        raise ValueError(
            f"{name} has kernel_size={tuple(conv.kernel_size)}. "
            "Per-anchor Laplace requires a 1x1 Conv2d."
        )

    if conv.bias is None:

        raise ValueError(
            f"{name} has bias=None. "
            "This adapter requires a bias."
        )


def _detect_anchor_axis(
    shape,
    num_anchors,
    tensor_name,
    exclude_axes=()
):
    """
    Find an explicit anchor dimension A.

    Used for pos_equal_one and neg_equal_one, which normally have shape:

        [B, H, W, A]
    """

    candidates = [

        ax

        for ax in range(
            1,
            len(shape)
        )

        if (
            shape[ax] == num_anchors
            and ax not in exclude_axes
        )
    ]

    if len(candidates) == 0:

        raise ValueError(
            f"Could not find anchor axis of size "
            f"{num_anchors} in {tensor_name} "
            f"with shape {tuple(shape)}."
        )

    if len(candidates) > 1:

        raise ValueError(
            f"{tensor_name} shape {tuple(shape)} "
            f"has multiple candidate anchor axes "
            f"of size {num_anchors}: {candidates}."
        )

    return candidates[0]


def _detect_regression_channel_axis(
    shape,
    num_anchors,
    tensor_name="targets"
):
    """
    Detect packed OpenCOOD regression channel dimension A*7.

    Example:

        num_anchors = 2

        targets.shape
            = [B, 48, 176, 14]

        14 = 2 anchors * 7 regression parameters

    Returns:
        axis containing A*7.
    """

    packed_size = (
        num_anchors * 7
    )

    candidates = [

        ax

        for ax in range(
            1,
            len(shape)
        )

        if shape[ax] == packed_size
    ]

    if len(candidates) == 0:

        raise ValueError(
            f"Could not find packed regression axis "
            f"of size A*7={packed_size} in "
            f"{tensor_name} with shape {tuple(shape)}."
        )

    if len(candidates) > 1:

        raise ValueError(
            f"{tensor_name} shape {tuple(shape)} "
            f"has multiple candidate packed regression "
            f"axes of size {packed_size}: {candidates}."
        )

    return candidates[0]


def _to_device(
    obj,
    device
):

    if torch.is_tensor(obj):

        return obj.to(
            device
        )

    if isinstance(
        obj,
        dict
    ):

        return {

            k:
                _to_device(
                    v,
                    device
                )

            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        list
    ):

        return [

            _to_device(
                v,
                device
            )

            for v
            in obj
        ]

    if isinstance(
        obj,
        tuple
    ):

        return tuple(

            _to_device(
                v,
                device
            )

            for v
            in obj
        )

    return obj


# ================================================================
# Dataset collection
# ================================================================

@torch.no_grad()
def collect_head_dataset(
    model,
    loader,
    device,
    max_batches=None,
    verbose=True
):
    """
    Run the MAP network on training batches and collect:

        input feature map to cls_head
        input feature map to reg_head
        positive anchor mask
        negative anchor mask
        encoded regression targets

    Data is stored on CPU after each batch.
    """

    cls_head = model.cls_head
    reg_head = model.reg_head

    # ------------------------------------------------------------
    # Check detection-head assumptions
    # ------------------------------------------------------------

    _require_1x1(
        cls_head,
        "model.cls_head"
    )

    _require_1x1(
        reg_head,
        "model.reg_head"
    )

    num_cls_anchors = (
        cls_head.out_channels
    )

    if (
        reg_head.out_channels
        % 7
        != 0
    ):

        raise ValueError(
            "reg_head.out_channels must be divisible by 7. "
            f"Got {reg_head.out_channels}."
        )

    num_reg_anchors = (
        reg_head.out_channels
        // 7
    )

    if (
        num_reg_anchors
        != num_cls_anchors
    ):

        raise ValueError(
            "Classification and regression heads imply "
            "different numbers of anchors: "
            f"cls={num_cls_anchors}, "
            f"reg={num_reg_anchors}."
        )

    num_anchors = (
        num_cls_anchors
    )

    print(
        "\n[Laplace-official] "
        f"Number of anchors detected from heads: "
        f"{num_anchors}"
    )

    print(
        "[Laplace-official] "
        f"cls_head: in={cls_head.in_channels}, "
        f"out={cls_head.out_channels}"
    )

    print(
        "[Laplace-official] "
        f"reg_head: in={reg_head.in_channels}, "
        f"out={reg_head.out_channels}"
    )

    # ------------------------------------------------------------
    # Capture head input feature maps
    # ------------------------------------------------------------

    captured = {}

    def cls_hook(
        _module,
        inputs
    ):

        captured[
            "cls_x"
        ] = (
            inputs[0]
            .detach()
        )

    def reg_hook(
        _module,
        inputs
    ):

        captured[
            "reg_x"
        ] = (
            inputs[0]
            .detach()
        )

    h1 = (
        cls_head
        .register_forward_pre_hook(
            cls_hook
        )
    )

    h2 = (
        reg_head
        .register_forward_pre_hook(
            reg_hook
        )
    )

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    cls_feat_list = []
    reg_feat_list = []

    pos_list = []
    neg_list = []

    reg_tgt_list = []

    axis_pos = None
    axis_neg = None
    axis_reg = None

    model.eval()

    try:

        for i, batch in enumerate(
            loader
        ):

            if (
                max_batches
                is not None
                and i >= max_batches
            ):

                break

            batch = _to_device(
                batch,
                device
            )

            ego = batch[
                "ego"
            ]

            # ----------------------------------------------------
            # MAP forward pass.
            #
            # Hooks capture the tensors immediately before
            # cls_head and reg_head.
            # ----------------------------------------------------

            model(
                ego
            )

            if (
                "cls_x"
                not in captured
                or
                "reg_x"
                not in captured
            ):

                raise RuntimeError(
                    "Failed to capture detection-head input "
                    "features. Check model.cls_head and "
                    "model.reg_head."
                )

            cls_x = (
                captured[
                    "cls_x"
                ]
            )

            reg_x = (
                captured[
                    "reg_x"
                ]
            )

            # ----------------------------------------------------
            # Labels
            # ----------------------------------------------------

            if (
                "label_dict"
                not in ego
            ):

                raise KeyError(
                    "ego['label_dict'] was not found."
                )

            labels = (
                ego[
                    "label_dict"
                ]
            )

            pos = (
                labels[
                    "pos_equal_one"
                ]
                .detach()
            )

            neg = (
                labels[
                    "neg_equal_one"
                ]
                .detach()
            )

            tgt = (
                labels[
                    "targets"
                ]
                .detach()
            )

            B = (
                cls_x.shape[0]
            )

            # ----------------------------------------------------
            # Detect layout ONCE
            # ----------------------------------------------------

            if axis_pos is None:

                axis_pos = (
                    _detect_anchor_axis(
                        pos.shape,
                        num_anchors,
                        "pos_equal_one"
                    )
                )

                axis_neg = (
                    _detect_anchor_axis(
                        neg.shape,
                        num_anchors,
                        "neg_equal_one"
                    )
                )

                # IMPORTANT FIX:
                #
                # OpenCOOD regression targets do NOT necessarily
                # have an explicit dimension of size A.
                #
                # Example:
                #
                # targets =
                #     [B, H, W, A*7]
                #
                # Your observed case:
                #
                #     [2, 48, 176, 14]
                #
                # where:
                #
                #     A = 2
                #     14 = 2*7
                #
                axis_reg = (
                    _detect_regression_channel_axis(
                        tgt.shape,
                        num_anchors,
                        "targets"
                    )
                )

                if verbose:

                    print(
                        "\n"
                        "========================================"
                    )

                    print(
                        "[Laplace-official] "
                        "Detected shapes / axes"
                    )

                    print(
                        "VERIFY THESE ONCE:"
                    )

                    print(
                        "========================================"
                    )

                    print(
                        f"cls_x             : "
                        f"{tuple(cls_x.shape)}"
                    )

                    print(
                        f"reg_x             : "
                        f"{tuple(reg_x.shape)}"
                    )

                    print(
                        f"pos_equal_one     : "
                        f"{tuple(pos.shape)} "
                        f"-> anchor axis {axis_pos}"
                    )

                    print(
                        f"neg_equal_one     : "
                        f"{tuple(neg.shape)} "
                        f"-> anchor axis {axis_neg}"
                    )

                    print(
                        f"targets           : "
                        f"{tuple(tgt.shape)} "
                        f"-> packed A*7 axis {axis_reg}"
                    )

                    print(
                        f"num_anchors       : "
                        f"{num_anchors}"
                    )

                    print(
                        f"regression dims   : "
                        f"{num_anchors} x 7 "
                        f"= {num_anchors * 7}"
                    )

                    print(
                        "========================================\n"
                    )

            # ----------------------------------------------------
            # Positive / negative masks
            #
            # Convert:
            #
            # [B, H, W, A]
            #
            # to:
            #
            # [B, HW, A]
            # ----------------------------------------------------

            pos_c = (
                pos
                .movedim(
                    axis_pos,
                    -1
                )
                .reshape(
                    B,
                    -1,
                    num_anchors
                )
            )

            neg_c = (
                neg
                .movedim(
                    axis_neg,
                    -1
                )
                .reshape(
                    B,
                    -1,
                    num_anchors
                )
            )

            # ----------------------------------------------------
            # Regression targets
            #
            # Convert:
            #
            # [B, H, W, A*7]
            #
            # to:
            #
            # [B, HW, A, 7]
            # ----------------------------------------------------

            tgt_c = (
                tgt
                .movedim(
                    axis_reg,
                    -1
                )
            )

            if (
                tgt_c.shape[-1]
                !=
                num_anchors * 7
            ):

                raise ValueError(
                    "Regression target channel mismatch after "
                    "moving packed axis. "
                    f"Got {tuple(tgt_c.shape)}, expected last "
                    f"dimension {num_anchors * 7}."
                )

            tgt_c = (
                tgt_c
                .reshape(
                    B,
                    -1,
                    num_anchors,
                    7
                )
            )

            # ----------------------------------------------------
            # Detection-head input features
            #
            # [B, C, H, W]
            #
            # ->
            #
            # [B, HW, C]
            # ----------------------------------------------------

            cls_x_c = (
                cls_x
                .permute(
                    0,
                    2,
                    3,
                    1
                )
                .contiguous()
                .reshape(
                    B,
                    -1,
                    cls_x.shape[1]
                )
            )

            reg_x_c = (
                reg_x
                .permute(
                    0,
                    2,
                    3,
                    1
                )
                .contiguous()
                .reshape(
                    B,
                    -1,
                    reg_x.shape[1]
                )
            )

            # ====================================================
            # CRITICAL SANITY CHECKS
            # ====================================================

            if (
                cls_x_c.shape[1]
                !=
                pos_c.shape[1]
            ):

                raise ValueError(
                    "Classification spatial mismatch:\n"
                    f"  features = {tuple(cls_x_c.shape)}\n"
                    f"  labels   = {tuple(pos_c.shape)}"
                )

            if (
                cls_x_c.shape[1]
                !=
                neg_c.shape[1]
            ):

                raise ValueError(
                    "Classification negative-mask spatial "
                    "mismatch:\n"
                    f"  features = {tuple(cls_x_c.shape)}\n"
                    f"  labels   = {tuple(neg_c.shape)}"
                )

            if (
                reg_x_c.shape[1]
                !=
                tgt_c.shape[1]
            ):

                raise ValueError(
                    "Regression spatial mismatch:\n"
                    f"  features = {tuple(reg_x_c.shape)}\n"
                    f"  targets  = {tuple(tgt_c.shape)}"
                )

            if (
                pos_c.shape[2]
                !=
                num_anchors
            ):

                raise ValueError(
                    "Positive-mask anchor mismatch: "
                    f"{tuple(pos_c.shape)}, "
                    f"expected A={num_anchors}."
                )

            if (
                neg_c.shape[2]
                !=
                num_anchors
            ):

                raise ValueError(
                    "Negative-mask anchor mismatch: "
                    f"{tuple(neg_c.shape)}, "
                    f"expected A={num_anchors}."
                )

            if (
                tgt_c.shape[2]
                !=
                num_anchors
            ):

                raise ValueError(
                    "Regression target anchor mismatch: "
                    f"{tuple(tgt_c.shape)}, "
                    f"expected A={num_anchors}."
                )

            if (
                tgt_c.shape[3]
                != 7
            ):

                raise ValueError(
                    "Regression target dimension mismatch: "
                    f"{tuple(tgt_c.shape)}, "
                    "expected 7 regression values per anchor."
                )

            # ----------------------------------------------------
            # Print normalized shapes for first batch
            # ----------------------------------------------------

            if (
                verbose
                and i == 0
            ):

                print(
                    "[Laplace-official] "
                    "Normalized first-batch shapes:"
                )

                print(
                    " cls features :",
                    tuple(
                        cls_x_c.shape
                    )
                )

                print(
                    " reg features :",
                    tuple(
                        reg_x_c.shape
                    )
                )

                print(
                    " pos mask     :",
                    tuple(
                        pos_c.shape
                    )
                )

                print(
                    " neg mask     :",
                    tuple(
                        neg_c.shape
                    )
                )

                print(
                    " reg targets  :",
                    tuple(
                        tgt_c.shape
                    )
                )

                print()

            # ----------------------------------------------------
            # Move collected data to CPU.
            # ----------------------------------------------------

            for b in range(B):

                cls_feat_list.append(
                    cls_x_c[
                        b
                    ]
                    .cpu()
                )

                reg_feat_list.append(
                    reg_x_c[
                        b
                    ]
                    .cpu()
                )

                pos_list.append(
                    pos_c[
                        b
                    ]
                    .cpu()
                )

                neg_list.append(
                    neg_c[
                        b
                    ]
                    .cpu()
                )

                reg_tgt_list.append(
                    tgt_c[
                        b
                    ]
                    .cpu()
                )

            if (
                verbose
                and
                (i + 1) % 50 == 0
            ):

                print(
                    "[Laplace-official] "
                    f"Collected {i + 1} batches"
                )

    finally:

        h1.remove()
        h2.remove()

    if not cls_feat_list:

        raise RuntimeError(
            "No batches were collected "
            "for Laplace fitting."
        )

    print(
        "\n[Laplace-official] "
        "Finished collecting fitting data."
    )

    print(
        "[Laplace-official] "
        f"Collected scene/batch elements: "
        f"{len(cls_feat_list)}"
    )

    return {

        "cls_feat":
            cls_feat_list,

        "reg_feat":
            reg_feat_list,

        "pos":
            pos_list,

        "neg":
            neg_list,

        "reg_tgt":
            reg_tgt_list,

        "num_anchors":
            num_anchors,
    }


# ================================================================
# Build per-anchor datasets
# ================================================================

def build_per_anchor_datasets(
    collected
):
    """
    Build one classification and one regression dataset per anchor.

    Classification:
        valid = positive OR negative

    Regression:
        positive anchors only
    """

    num_anchors = (
        collected[
            "num_anchors"
        ]
    )

    cls_feat = torch.cat(
        collected[
            "cls_feat"
        ],
        dim=0
    )

    reg_feat = torch.cat(
        collected[
            "reg_feat"
        ],
        dim=0
    )

    pos = torch.cat(
        collected[
            "pos"
        ],
        dim=0
    )

    neg = torch.cat(
        collected[
            "neg"
        ],
        dim=0
    )

    reg_tgt = torch.cat(
        collected[
            "reg_tgt"
        ],
        dim=0
    )

    print(
        "\n[Laplace-official] "
        "Combined dataset shapes:"
    )

    print(
        " cls_feat:",
        tuple(
            cls_feat.shape
        )
    )

    print(
        " reg_feat:",
        tuple(
            reg_feat.shape
        )
    )

    print(
        " pos:",
        tuple(
            pos.shape
        )
    )

    print(
        " neg:",
        tuple(
            neg.shape
        )
    )

    print(
        " reg_tgt:",
        tuple(
            reg_tgt.shape
        )
    )

    per_anchor = []

    for a in range(
        num_anchors
    ):

        pos_a = (
            pos[
                :,
                a
            ]
            > 0
        )

        neg_a = (
            neg[
                :,
                a
            ]
            > 0
        )

        # --------------------------------------------------------
        # Classification
        #
        # Positive OR negative = valid.
        #
        # Don't-care anchors are ignored.
        # --------------------------------------------------------

        valid_a = (
            pos_a
            |
            neg_a
        )

        cls_x_a = (
            cls_feat[
                valid_a
            ]
        )

        cls_y_a = (
            pos_a[
                valid_a
            ]
            .long()
        )

        # --------------------------------------------------------
        # Regression
        #
        # ONLY positive anchors contribute to regression curvature.
        # --------------------------------------------------------

        reg_x_a = (
            reg_feat[
                pos_a
            ]
        )

        reg_y_a = (
            reg_tgt[
                pos_a,
                a,
                :
            ]
        )

        n_cls = int(
            valid_a
            .sum()
            .item()
        )

        n_reg = int(
            pos_a
            .sum()
            .item()
        )

        n_pos_cls = int(
            pos_a[
                valid_a
            ]
            .sum()
            .item()
        )

        n_neg_cls = (
            n_cls
            -
            n_pos_cls
        )

        per_anchor.append({

            "cls_x":
                cls_x_a,

            "cls_y":
                cls_y_a,

            "reg_x":
                reg_x_a,

            "reg_y":
                reg_y_a,

            "n_cls":
                n_cls,

            "n_reg":
                n_reg,
        })

        print(
            "\n[Laplace-official] "
            f"anchor {a}:"
        )

        print(
            f"  classification total : "
            f"{n_cls}"
        )

        print(
            f"  classification pos   : "
            f"{n_pos_cls}"
        )

        print(
            f"  classification neg   : "
            f"{n_neg_cls}"
        )

        print(
            f"  regression positives : "
            f"{n_reg}"
        )

        if n_cls == 0:

            raise RuntimeError(
                f"Anchor {a} has zero valid "
                "classification samples."
            )

        if n_reg == 0:

            raise RuntimeError(
                f"Anchor {a} has zero positive "
                "regression samples. "
                "Increase --max_fit_batches "
                "or use more training data."
            )

    return per_anchor


# ================================================================
# Bundle classes
# ================================================================

@dataclass
class _AnchorFit:

    la: object
    channels: List[int]


@dataclass
class PerAnchorLaplaceBundle:

    kind: str
    fits: List[_AnchorFit]

    def to(
        self,
        device
    ):
        """
        Move anchor-specific Laplace models to device.
        """

        for f in self.fits:

            f.la.model.to(
                device
            )

        return self

    @torch.no_grad()
    def sample_into(
        self,
        conv: nn.Conv2d,
        alpha: float = 1.0
    ):
        """
        Draw one posterior sample per anchor and write it into
        the corresponding OpenCOOD 1x1 convolution channels.

        Posterior covariance scaling:

            theta_alpha =
                theta_MAP
                + sqrt(alpha)
                * (theta_sample - theta_MAP)

        Therefore:

            Cov(theta_alpha)
                =
                alpha * Cov(theta_sample)
        """

        if alpha < 0:

            raise ValueError(
                f"alpha must be >= 0, "
                f"got {alpha}"
            )

        scale = (
            float(alpha)
            ** 0.5
        )

        for f in self.fits:

            # ----------------------------------------------------
            # Native sample from laplace-torch
            # ----------------------------------------------------

            sample = (
                f.la.sample(
                    1
                )[0]
            )

            # ----------------------------------------------------
            # MAP parameters of anchor-specific model
            # ----------------------------------------------------

            map_params = torch.cat([

                p.detach()
                .reshape(-1)

                for p
                in f.la.model.parameters()

            ]).to(
                sample.device
            )

            if (
                sample.numel()
                !=
                map_params.numel()
            ):

                raise RuntimeError(
                    "Laplace sample and MAP parameter "
                    "sizes do not match: "
                    f"sample={sample.numel()}, "
                    f"MAP={map_params.numel()}."
                )

            # ----------------------------------------------------
            # Alpha covariance scaling
            # ----------------------------------------------------

            sample = (
                map_params
                +
                scale
                *
                (
                    sample
                    -
                    map_params
                )
            )

            out_ch = (
                f.channels
            )

            n_out = len(
                out_ch
            )

            in_ch = (
                conv.in_channels
            )

            n_w = (
                n_out
                *
                in_ch
            )

            expected = (
                n_w
                +
                n_out
            )

            if (
                sample.numel()
                != expected
            ):

                raise RuntimeError(
                    "Unexpected sampled parameter count "
                    f"for {self.kind}: "
                    f"got {sample.numel()}, "
                    f"expected {expected}."
                )

            # ----------------------------------------------------
            # Weight
            # ----------------------------------------------------

            w = (
                sample[
                    :n_w
                ]
                .reshape(
                    n_out,
                    in_ch,
                    1,
                    1
                )
            )

            # ----------------------------------------------------
            # Bias
            # ----------------------------------------------------

            b = (
                sample[
                    n_w:
                    n_w + n_out
                ]
            )

            conv.weight.data[
                out_ch
            ] = (
                w.to(
                    device=conv.weight.device,
                    dtype=conv.weight.dtype
                )
            )

            conv.bias.data[
                out_ch
            ] = (
                b.to(
                    device=conv.bias.device,
                    dtype=conv.bias.dtype
                )
            )


# ================================================================
# Fit official laplace-torch posterior
# ================================================================

def fit_official_head_laplace(
    model,
    per_anchor_datasets,
    prior_precision=1.0,
    cls_batch_size=4096,
    reg_batch_size=4096
):
    """
    Fit official laplace-torch diagonal posteriors.

    One classification posterior and one regression posterior
    are fitted for each anchor.
    """

    device = next(
        model.parameters()
    ).device

    cls_head = (
        model.cls_head
    )

    reg_head = (
        model.reg_head
    )

    _require_1x1(
        cls_head,
        "model.cls_head"
    )

    _require_1x1(
        reg_head,
        "model.reg_head"
    )

    in_c_cls = (
        cls_head.in_channels
    )

    in_c_reg = (
        reg_head.in_channels
    )

    cls_fits = []
    reg_fits = []

    print(
        "\n========================================"
    )

    print(
        "[Laplace-official] "
        "Starting laplace-torch fitting"
    )

    print(
        f"prior_precision = "
        f"{prior_precision}"
    )

    print(
        "hessian_structure = diag"
    )

    print(
        "parameter scope = detection heads"
    )

    print(
        "========================================\n"
    )

    for a, d in enumerate(
        per_anchor_datasets
    ):

        print(
            "\n----------------------------------------"
        )

        print(
            "[Laplace-official] "
            f"Fitting anchor {a}"
        )

        print(
            "----------------------------------------"
        )

        # ========================================================
        # Classification
        # ========================================================

        cls_model = (
            _TwoClassLinear(
                in_c_cls
            )
            .to(
                device
            )
        )

        # Initialize anchor-specific model at MAP detector weights.
        with torch.no_grad():

            cls_model.linear.weight.copy_(

                cls_head.weight.data[
                    a,
                    :,
                    0,
                    0
                ]
                .reshape(
                    1,
                    -1
                )
            )

            cls_model.linear.bias.copy_(

                cls_head.bias.data[
                    a:
                    a + 1
                ]
            )

        if (
            d[
                "n_cls"
            ]
            == 0
        ):

            raise RuntimeError(
                f"Anchor {a} has zero "
                "classification samples."
            )

        cls_dataset = (
            TensorDataset(
                d[
                    "cls_x"
                ],
                d[
                    "cls_y"
                ]
            )
        )

        cls_loader = (
            DataLoader(

                cls_dataset,

                batch_size=min(
                    cls_batch_size,
                    d[
                        "n_cls"
                    ]
                ),

                shuffle=False
            )
        )

        cls_la = Laplace(

            cls_model,

            "classification",

            subset_of_weights="all",

            hessian_structure="diag",

            prior_precision=(
                prior_precision
            )
        )

        print(
            "[Laplace-official] "
            f"anchor {a}: "
            "fitting classification "
            f"(n={d['n_cls']})..."
        )

        cls_la.fit(
            cls_loader
        )

        cls_fits.append(

            _AnchorFit(

                la=cls_la,

                channels=[
                    a
                ]
            )
        )

        # ========================================================
        # Regression
        # ========================================================

        reg_model = (
            _RegressionLinear(
                in_c_reg,
                7
            )
            .to(
                device
            )
        )

        # Initialize at MAP detector weights.
        with torch.no_grad():

            reg_model.linear.weight.copy_(

                reg_head.weight.data[
                    a * 7:
                    (a + 1) * 7,
                    :,
                    0,
                    0
                ]
            )

            reg_model.linear.bias.copy_(

                reg_head.bias.data[
                    a * 7:
                    (a + 1) * 7
                ]
            )

        if (
            d[
                "n_reg"
            ]
            == 0
        ):

            raise RuntimeError(
                f"Anchor {a} has zero "
                "positive regression samples."
            )

        reg_dataset = (
            TensorDataset(
                d[
                    "reg_x"
                ],
                d[
                    "reg_y"
                ]
            )
        )

        reg_loader = (
            DataLoader(

                reg_dataset,

                batch_size=min(
                    reg_batch_size,
                    d[
                        "n_reg"
                    ]
                ),

                shuffle=False
            )
        )

        reg_la = Laplace(

            reg_model,

            "regression",

            subset_of_weights="all",

            hessian_structure="diag",

            prior_precision=(
                prior_precision
            )
        )

        print(
            "[Laplace-official] "
            f"anchor {a}: "
            "fitting regression "
            f"(n={d['n_reg']})..."
        )

        reg_la.fit(
            reg_loader
        )

        reg_fits.append(

            _AnchorFit(

                la=reg_la,

                channels=list(
                    range(
                        a * 7,
                        (a + 1) * 7
                    )
                )
            )
        )

        print(
            "[Laplace-official] "
            f"anchor {a} fitted successfully: "
            f"cls n={d['n_cls']}, "
            f"reg n={d['n_reg']}"
        )

    print(
        "\n========================================"
    )

    print(
        "[Laplace-official] "
        "All anchor posteriors fitted."
    )

    print(
        "========================================\n"
    )

    return (

        PerAnchorLaplaceBundle(

            kind="classification",

            fits=cls_fits
        ),

        PerAnchorLaplaceBundle(

            kind="regression",

            fits=reg_fits
        )
    )


# ================================================================
# Laplace Monte-Carlo inference
# ================================================================

def predict_with_head_laplace(
    model,
    cls_bundle,
    reg_bundle,
    ego_data,
    T=10,
    alpha=1.0
):
    """
    Monte-Carlo prediction using sampled Laplace detection-head weights.

    Backbone/fusion network remains at MAP.

    Only cls_head and reg_head are sampled.

    Parameters
    ----------
    T:
        Number of posterior weight samples.

    alpha:
        Posterior covariance multiplier.

        alpha=0:
            deterministic MAP

        alpha=1:
            native laplace-torch posterior
    """

    if T < 1:

        raise ValueError(
            f"T must be >= 1, "
            f"got {T}"
        )

    if alpha < 0:

        raise ValueError(
            f"alpha must be >= 0, "
            f"got {alpha}"
        )

    # ------------------------------------------------------------
    # Save MAP detection-head parameters.
    # ------------------------------------------------------------

    cls_map = (
        copy.deepcopy(
            model.cls_head.state_dict()
        )
    )

    reg_map = (
        copy.deepcopy(
            model.reg_head.state_dict()
        )
    )

    psm_samples = []
    rm_samples = []

    try:

        for _ in range(T):

            # ----------------------------------------------------
            # Draw independent head samples.
            # ----------------------------------------------------

            cls_bundle.sample_into(
                model.cls_head,
                alpha=alpha
            )

            reg_bundle.sample_into(
                model.reg_head,
                alpha=alpha
            )

            # ----------------------------------------------------
            # Full OpenCOOD forward pass.
            #
            # Backbone stays MAP;
            # only detection heads have sampled weights.
            # ----------------------------------------------------

            out = model(
                ego_data
            )

            psm_samples.append(

                out[
                    "psm"
                ]
                .detach()
            )

            rm_samples.append(

                out[
                    "rm"
                ]
                .detach()
            )

    finally:

        # --------------------------------------------------------
        # Always restore original MAP heads.
        # --------------------------------------------------------

        model.cls_head.load_state_dict(
            cls_map
        )

        model.reg_head.load_state_dict(
            reg_map
        )

    # ------------------------------------------------------------
    # [T, B, ...]
    # ------------------------------------------------------------

    psm = torch.stack(
        psm_samples,
        dim=0
    )

    rm = torch.stack(
        rm_samples,
        dim=0
    )

    # ------------------------------------------------------------
    # Classification probabilities
    # ------------------------------------------------------------

    probs = torch.sigmoid(
        psm
    )

    mean_prob = (
        probs.mean(
            dim=0
        )
    )

    var_prob = (
        probs.var(
            dim=0,
            unbiased=False
        )
    )

    # ------------------------------------------------------------
    # Classification entropy / mutual information
    # ------------------------------------------------------------

    eps = 1e-8

    def entropy(p):

        p = p.clamp(
            eps,
            1.0 - eps
        )

        return -(

            p
            *
            torch.log(
                p
            )

            +

            (
                1.0
                -
                p
            )
            *
            torch.log(
                1.0
                -
                p
            )
        )

    predictive_entropy = (
        entropy(
            mean_prob
        )
    )

    expected_entropy = (
        entropy(
            probs
        )
        .mean(
            dim=0
        )
    )

    mutual_information = (

        predictive_entropy
        -
        expected_entropy

    ).clamp_min(
        0.0
    )

    # ------------------------------------------------------------
    # Regression predictive moments
    # ------------------------------------------------------------

    mean_reg = (
        rm.mean(
            dim=0
        )
    )

    var_reg = (
        rm.var(
            dim=0,
            unbiased=False
        )
    )

    return {

        "mean_cls_prob":
            mean_prob,

        "var_cls_prob":
            var_prob,

        "predictive_entropy":
            predictive_entropy,

        "mutual_information":
            mutual_information,

        "mean_reg_encoded":
            mean_reg,

        "var_reg_encoded":
            var_reg,

        # Retained in case later analysis
        # needs the individual samples.
        "cls_prob_samples":
            probs,

        "reg_samples":
            rm,
    }


# ================================================================
# Convert Bayesian predictive mean to standard OpenCOOD output
# ================================================================

def make_opencood_output(
    result,
    eps=1e-6
):
    """
    Convert Laplace predictive mean into the ordinary OpenCOOD output:

        {
            "psm": classification logits,
            "rm":  encoded regression output
        }

    This allows the standard OpenCOOD postprocessor/AP code to be reused.
    """

    p = (

        result[
            "mean_cls_prob"
        ]

        .clamp(
            eps,
            1.0 - eps
        )
    )

    # Convert predictive probability back to logit because
    # OpenCOOD post-processing expects psm logits.
    mean_logit = torch.log(

        p
        /
        (
            1.0
            -
            p
        )
    )

    return {

        "psm":
            mean_logit,

        "rm":
            result[
                "mean_reg_encoded"
            ],
    }
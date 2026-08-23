"""Fit PointPillar detection-head Laplace approximations with laplace-torch.

Fits one diagonal laplace-torch posterior PER ANCHOR for cls_head
and reg_head.

This is HEAD / LAST-MODULE Laplace, NOT full-network Laplace.
"""

import argparse
import os

import torch

from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.tools import train_utils

from opencood.data_utils.datasets import (
    build_dataset
)

from opencood.uncertainty.official_laplace_heads import (
    collect_head_dataset,
    build_per_anchor_datasets,
    fit_official_head_laplace,
)


def parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_dir",
        required=True
    )

    p.add_argument(
        "--max_fit_batches",
        type=int,
        default=None,
        help=(
            "Maximum number of training batches "
            "used to fit Laplace. "
            "Default None uses the full training set."
        )
    )

    p.add_argument(
        "--prior_precision",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--output_name",
        default="laplace_torch_heads.pth"
    )

    return p.parse_args()


def main():

    opt = parser()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # ============================================================
    # Configuration
    # ============================================================

    config_path = os.path.join(
        opt.model_dir,
        "config.yaml"
    )

    hypes = yaml_utils.load_yaml(
        config_path,
        opt
    )

    # ============================================================
    # MAP model
    # ============================================================

    model = (
        train_utils
        .create_model(
            hypes
        )
        .to(device)
    )

    epoch, model = (
        train_utils
        .load_saved_model(
            opt.model_dir,
            model
        )
    )

    model.eval()

    print(
        f"Loaded MAP model "
        f"from epoch {epoch}"
    )

    # ============================================================
    # Training dataset
    # ============================================================

    ds = build_dataset(
        hypes,
        visualize=False,
        train=True
    )

    batch_size = (
        hypes[
            "train_params"
        ][
            "batch_size"
        ]
    )

    print(
        "Laplace fitting "
        f"batch size: {batch_size}"
    )

    loader = DataLoader(

        ds,

        batch_size=batch_size,

        num_workers=8,

        collate_fn=(
            ds.collate_batch_train
        ),

        shuffle=False,

        drop_last=True,
    )

    # ============================================================
    # Capture features + labels
    # ============================================================

    print(
        "\n[Laplace-official] "
        "Collecting per-anchor fitting data..."
    )

    collected = (
        collect_head_dataset(
            model,
            loader,
            device,
            opt.max_fit_batches
        )
    )

    # ============================================================
    # Construct correct masked datasets
    # ============================================================

    print(
        "\n[Laplace-official] "
        "Building per-anchor datasets..."
    )

    per_anchor = (
        build_per_anchor_datasets(
            collected
        )
    )

    # ============================================================
    # Official laplace-torch fitting
    # ============================================================

    print(
        "\n[Laplace-official] "
        "Fitting laplace-torch posteriors..."
    )

    cls_la, reg_la = (
        fit_official_head_laplace(

            model,

            per_anchor,

            opt.prior_precision
        )
    )

    # ============================================================
    # Save posterior
    # ============================================================

    path = os.path.join(
        opt.model_dir,
        opt.output_name
    )

    checkpoint = {

        "cls_la":
            cls_la,

        "reg_la":
            reg_la,

        "epoch":
            epoch,

        "prior_precision":
            opt.prior_precision,

        # Metadata
        "backend":
            "laplace-torch",

        "param_scope":
            "heads",

        "hessian_structure":
            "diag",

        "laplace_variant":
            "per_anchor_last_module",

        "num_anchors":
            collected[
                "num_anchors"
            ],
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        "\nSaved official "
        "laplace-torch posterior:"
    )

    print(
        path
    )

    print(
        "\nConfiguration:"
    )

    print(
        " backend          = "
        "laplace-torch"
    )

    print(
        " parameter scope  = heads"
    )

    print(
        " Hessian          = diagonal"
    )

    print(
        " prior precision  =",
        opt.prior_precision
    )

    print(
        " anchors          =",
        collected[
            "num_anchors"
        ]
    )


if __name__ == "__main__":

    main()
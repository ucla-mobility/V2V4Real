import argparse
import os

import torch

from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.tools import train_utils

from opencood.data_utils.datasets import (
    build_dataset
)

from opencood.uncertainty.diagonal_laplace import (
    DiagonalLaplace
)


def parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_dir",
        type=str,
        required=True
    )

    p.add_argument(
        "--param_scope",
        type=str,
        default="heads",
        choices=[
            "heads",
            "full"
        ]
    )

    p.add_argument(
        "--max_fit_batches",
        type=int,
        default=None
    )

    p.add_argument(
        "--output_name",
        type=str,
        default=None
    )

    return p.parse_args()


def main():

    opt = parser()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------
    # Configuration
    # ------------------------------------------------

    config_path = os.path.join(
        opt.model_dir,
        "config.yaml"
    )

    hypes = yaml_utils.load_yaml(
        config_path,
        opt
    )

    # ------------------------------------------------
    # MAP model
    # ------------------------------------------------

    print("\nCreating model...")

    model = train_utils.create_model(
        hypes
    )

    model.to(device)

    epoch_loaded, model = (
        train_utils.load_saved_model(
            opt.model_dir,
            model
        )
    )

    print(
        f"\nLoaded MAP checkpoint: "
        f"epoch {epoch_loaded}"
    )

    model.eval()

    # ------------------------------------------------
    # Training dataset
    # ------------------------------------------------

    print("\nBuilding TRAINING dataset...")

    train_dataset = build_dataset(
        hypes,
        visualize=False,
        train=True
    )

    batch_size = hypes[
        "train_params"
    ]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=(
            train_dataset.collate_batch_train
        ),
        shuffle=True,
        pin_memory=False,
        drop_last=True
    )

    # ------------------------------------------------
    # Same criterion as detector training
    # ------------------------------------------------

    criterion = train_utils.create_loss(
        hypes
    )

    # ------------------------------------------------
    # Scope
    # ------------------------------------------------

    if opt.param_scope == "heads":

        param_scope = (
            "cls_head",
            "reg_head"
        )

    else:

        param_scope = "full"

    # ------------------------------------------------
    # Laplace
    # ------------------------------------------------

    laplace = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device
    )

    laplace.fit_empirical_diag(
        train_loader=train_loader,
        criterion=criterion,
        to_device_fn=lambda batch:
            train_utils.to_device(
                batch,
                device
            ),
        max_batches=opt.max_fit_batches
    )

    # IMPORTANT:
    # Do not permanently choose alpha here.
    # Fisher should be reusable for many alpha values.

    if opt.output_name is None:

        output_name = (
            f"laplace_curvature_"
            f"epoch{epoch_loaded}_"
            f"{opt.param_scope}.pth"
        )

    else:

        output_name = opt.output_name

    save_path = os.path.join(
        opt.model_dir,
        output_name
    )

    laplace.save(save_path)

    print("\n================================")
    print("Laplace curvature computation done")
    print("================================")
    print("Saved:", save_path)


if __name__ == "__main__":

    main()
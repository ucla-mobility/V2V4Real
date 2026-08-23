"""
Compute diagonal Laplace curvature for a trained OpenCOOD detector.

This script:

1. Loads an already-trained OpenCOOD model / MAP checkpoint.
2. Builds the training dataset and dataloader.
3. Selects the parameters that receive the Laplace approximation.
4. Computes an empirical diagonal curvature estimate using the native
   OpenCOOD detection loss.
5. Saves MAP parameters + curvature for later posterior construction
   and uncertainty inference.

Important
---------
The curvature in this implementation is estimated from squared gradients
of the OpenCOOD detector training objective:

    F_diag ~= E[(dL / dtheta)^2]

laplace-torch is subsequently used by DiagonalLaplace for posterior
construction and posterior parameter sampling.

Therefore this script should NOT be described as using laplace-torch's
native GGN/Hessian backend for curvature.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Make project root importable when this script is executed directly.
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.uncertainty.diagonal_laplace import DiagonalLaplace


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical diagonal curvature for "
            "Laplace approximation of an OpenCOOD detector."
        )
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help=(
            "Directory containing the trained OpenCOOD model "
            "and configuration."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "Output path for the saved Laplace curvature. "
            "Default: <model_dir>/laplace_diag.pt"
        ),
    )

    parser.add_argument(
        "--param_scope",
        default="heads",
        choices=[
            "heads",
            "full",
        ],
        help=(
            "Parameters included in the Laplace approximation. "
            "'heads' selects cls_head and reg_head. "
            "'full' selects the entire detector."
        ),
    )

    parser.add_argument(
        "--max_batches",
        default=None,
        type=int,
        help=(
            "Maximum number of training batches used for "
            "curvature estimation. By default all batches are used."
        ),
    )

    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help=(
            "Optional batch-size override. "
            "If omitted, use the value from the YAML configuration."
        ),
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help=(
            "Device, e.g. 'cuda', 'cuda:0', or 'cpu'. "
            "Default: CUDA when available."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------

def resolve_device(device_argument):

    if device_argument is not None:
        return torch.device(device_argument)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Move nested OpenCOOD batch to device
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    device = resolve_device(
        args.device
    )

    print(
        f"\n[Laplace] Device: {device}"
    )

    # -------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------

    hypes = load_yaml(
        None,
        args
    )

    # -------------------------------------------------------------
    # Build TRAINING dataset
    # -------------------------------------------------------------
    #
    # Curvature must be estimated from training data around the
    # already-trained MAP solution.
    # -------------------------------------------------------------

    print(
        "\n[Laplace] Building training dataset..."
    )

    train_dataset = build_dataset(
        hypes,
        visualize=False,
        train=True,
    )

    if args.batch_size is not None:

        batch_size = args.batch_size

    else:

        batch_size = hypes[
            "train_params"
        ].get(
            "batch_size",
            1,
        )

    print(
        f"[Laplace] Batch size: {batch_size}"
    )

    # OpenCOOD datasets provide their own collate function.

    if hasattr(
        train_dataset,
        "collate_batch_train",
    ):

        collate_fn = (
            train_dataset.collate_batch_train
        )

    else:

        collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    print(
        f"[Laplace] Training batches: "
        f"{len(train_loader)}"
    )

    # -------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------

    print(
        "\n[Laplace] Creating model..."
    )

    model = train_utils.create_model(
        hypes
    )

    model = model.to(
        device
    )

    # -------------------------------------------------------------
    # Load trained MAP checkpoint
    # -------------------------------------------------------------

    print(
        "\n[Laplace] Loading trained checkpoint..."
    )

    _, model = train_utils.load_saved_model(
        args.model_dir,
        model,
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "[Laplace] MAP model loaded."
    )

    # -------------------------------------------------------------
    # Create native OpenCOOD criterion
    # -------------------------------------------------------------

    print(
        "\n[Laplace] Creating OpenCOOD loss..."
    )

    criterion = train_utils.create_loss(
        hypes
    )

    # -------------------------------------------------------------
    # Select Laplace parameter scope
    # -------------------------------------------------------------

    if args.param_scope == "heads":

        param_scope = (
            "cls_head",
            "reg_head",
        )

    elif args.param_scope == "full":

        param_scope = "full"

    else:

        raise ValueError(
            f"Unsupported param_scope: "
            f"{args.param_scope}"
        )

    print(
        f"\n[Laplace] Parameter scope: "
        f"{args.param_scope}"
    )

    # -------------------------------------------------------------
    # Construct Laplace object
    # -------------------------------------------------------------

    laplace_model = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device,
    )

    # -------------------------------------------------------------
    # Count selected parameters
    # -------------------------------------------------------------

    selected_parameter_count = sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if name in laplace_model.param_names
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "\n[Laplace] Selected parameters: "
        f"{selected_parameter_count:,}"
    )

    print(
        "[Laplace] Total model parameters: "
        f"{total_parameter_count:,}"
    )

    print(
        "[Laplace] Laplace fraction: "
        f"{100.0 * selected_parameter_count / total_parameter_count:.4f}%"
    )

    # -------------------------------------------------------------
    # Compute empirical diagonal curvature
    # -------------------------------------------------------------

    def to_device_fn(batch):

        return move_to_device(
            batch,
            device
        )

    laplace_model.fit_empirical_diag(
        train_loader=train_loader,
        criterion=criterion,
        to_device_fn=to_device_fn,
        max_batches=args.max_batches,
    )

    # -------------------------------------------------------------
    # Output path
    # -------------------------------------------------------------

    if args.output is None:

        output_path = os.path.join(
            args.model_dir,
            "laplace_diag.pt",
        )

    else:

        output_path = args.output

    output_directory = os.path.dirname(
        os.path.abspath(
            output_path
        )
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    laplace_model.save(
        output_path
    )

    print(
        "\n============================================"
    )

    print(
        "Laplace curvature computation complete."
    )

    print(
        f"Saved to:\n{output_path}"
    )

    print(
        "============================================\n"
    )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
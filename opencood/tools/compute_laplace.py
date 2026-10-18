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
        "--laplace_dir",
        default=None,
        type=str,
        help=(
            "Directory for all Laplace-generated artifacts. "
            "Default: <model_dir>_laplace"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "Output path for the saved Laplace curvature. "
            "Default: <laplace_dir>/laplace_diag.pt"
        ),
    )

    parser.add_argument(
        "--param_scope",
        default="heads",
        choices=[
            "heads",
            "last_module",
            "full",
        ],
        help=(
            "Parameters included in the Laplace approximation "
            "(BatchNorm never included). "
            "'heads': cls_head and reg_head (last layer). "
            "'last_module': last block before the heads + heads "
            "(F-Cooper / V2VNet / V2X-ViT: shrink_conv; AttFuse: "
            "backbone.deblocks). "
            "'full': the entire detector."
        ),
    )
    parser.add_argument(
        "--scope_modules",
        default=None,
        type=str,
        help=(
            "Comma-separated module prefixes that override --param_scope, "
            "e.g. 'backbone.deblocks,backbone.fuse_modules,cls_head,reg_head'. "
            "Use the same value in inference_laplace.py."
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
        "--curvature",
        default=None,
        choices=["ggn", "sampled_fisher", "ef"],
        help=(
            "'ggn': exact diagonal generalized Gauss-Newton, heads only "
            "(default for --param_scope heads). 'sampled_fisher': diagonal "
            "Fisher with labels sampled from the model, any scope (default "
            "otherwise); unbiased estimate of the same GGN. 'ef': empirical "
            "Fisher of the detector loss (collapses with focal loss)."
        ),
    )

    parser.add_argument(
        "--fisher_samples",
        default=1,
        type=int,
        help="Label samples per frame for --curvature sampled_fisher.",
    )

    parser.add_argument(
        "--noise_frames",
        default=500,
        type=int,
        help=(
            "Random training frames for the regression-noise pre-pass "
            "(--curvature sampled_fisher)."
        ),
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for label sampling (--curvature sampled_fisher).",
    )

    parser.add_argument(
        "--train_dataset",
        default="opv2v",
        type=str,
        help=(
            "Value of the config 'dataset' field used for the curvature pass. "
            "The model was trained on OPV2V, but its config.yaml may already "
            "be edited to v2v4real for testing; 'opv2v' switches the dataset "
            "code back to the simulation (training) coordinate handling."
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

    # Keep the trained MAP directory read-only. All post-hoc Laplace
    # artifacts are written to a sibling *_laplace directory by default.
    if args.laplace_dir is None:
        args.laplace_dir = args.model_dir.rstrip(os.sep) + "_laplace"

    args.laplace_dir = os.path.abspath(args.laplace_dir)
    os.makedirs(args.laplace_dir, exist_ok=True)

    print(f"[Laplace] Source model directory: {os.path.abspath(args.model_dir)}")
    print(f"[Laplace] Output directory:       {args.laplace_dir}")

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

    # Curvature is a property of the TRAINING data: make sure the training
    # dataset is read with the training-domain settings.
    if hypes.get("dataset") != args.train_dataset:
        print(
            f"[Laplace] config dataset '{hypes.get('dataset')}' -> "
            f"'{args.train_dataset}' for the curvature pass"
        )
        hypes["dataset"] = args.train_dataset
    print(f"[Laplace] Training data: {hypes['root_dir']}")

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

    # Per-example gradients are required for the empirical Fisher.
    batch_size = 1

    print(
        f"[Laplace] Batch size: {batch_size} (per-example gradients)"
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

    param_scope = args.param_scope
    heads_only = param_scope == "heads" and not args.scope_modules
    if args.curvature is None:
        args.curvature = "ggn" if heads_only else "sampled_fisher"

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
        scope_modules=args.scope_modules,
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

    loss_args = hypes.get("loss", {}).get("args", {})
    print(f"[Laplace] Curvature: {args.curvature}")
    if args.curvature == "ggn":
        if args.param_scope != "heads":
            raise ValueError("--curvature ggn supports --param_scope heads only")
        laplace_model.fit_ggn_diag(
            train_loader=train_loader,
            to_device_fn=to_device_fn,
            max_batches=args.max_batches,
            n_data=len(train_dataset),
            logvar_min=loss_args.get("log_var_min", -7.0),
            logvar_max=loss_args.get("log_var_max", 7.0),
        )
    elif args.curvature == "sampled_fisher":
        laplace_model.fit_sampled_fisher_diag(
            train_loader=train_loader,
            to_device_fn=to_device_fn,
            max_batches=args.max_batches,
            n_data=len(train_dataset),
            n_samples=args.fisher_samples,
            noise_frames=args.noise_frames,
            logvar_min=loss_args.get("log_var_min", -7.0),
            logvar_max=loss_args.get("log_var_max", 7.0),
            seed=args.seed,
        )
    else:
        laplace_model.fit_empirical_diag(
            train_loader=train_loader,
            criterion=criterion,
            to_device_fn=to_device_fn,
            max_batches=args.max_batches,
            n_data=len(train_dataset),
        )

    # -------------------------------------------------------------
    # Output path
    # -------------------------------------------------------------

    if args.output is None:

        output_path = os.path.join(
            args.laplace_dir,
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
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
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.uncertainty.diagonal_laplace import DiagonalLaplace


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
        "--laplace_path",
        default=None,
        type=str,
        help=(
            "Saved curvature file. "
            "Default: <model_dir>/laplace_diag.pt"
        ),
    )

    parser.add_argument(
        "--param_scope",
        default="heads",
        choices=["heads", "full"],
        help=(
            "'heads': Laplace approximation over cls_head/reg_head. "
            "'full': approximation over the complete detector."
        ),
    )

    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="Laplace posterior covariance scaling.",
    )

    parser.add_argument(
        "--prior_precision",
        default=1.0,
        type=float,
        help="Gaussian prior precision.",
    )

    parser.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Number of posterior Monte Carlo samples.",
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

        "classification_samples":
            classification_samples,

        "regression_samples":
            regression_samples,
    }


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
# Main
# ==============================================================

def main():

    args = parse_args()

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
        visualize=False,
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

    if args.param_scope == "heads":

        param_scope = (
            "cls_head",
            "reg_head",
        )

    else:

        param_scope = "full"

    # ----------------------------------------------------------
    # Laplace object
    # ----------------------------------------------------------

    laplace_model = DiagonalLaplace(
        model=model,
        param_scope=param_scope,
        device=device,
    )

    # ----------------------------------------------------------
    # Load curvature
    # ----------------------------------------------------------

    if args.laplace_path is None:

        laplace_path = os.path.join(
            args.model_dir,
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

    laplace_model.set_posterior(
        alpha=args.alpha,
        prior_precision=args.prior_precision,
    )

    print(
        "\n[Laplace] Posterior ready."
    )

    print(
        f"[Laplace] MC samples: "
        f"{args.num_samples}"
    )

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

        reg_variance = result[
            "regression"
        ][
            "variance"
        ]

        # ------------------------------------------------------
        # At this point:
        #
        # bayesian_output["psm"]
        # bayesian_output["rm"]
        #
        # can be passed through your existing OpenCOOD
        # postprocessor.
        #
        # And:
        #
        # cls_variance
        # predictive_entropy
        # mutual_information
        # reg_variance
        #
        # can be fed into your existing uncertainty evaluation /
        # visualization code.
        # ------------------------------------------------------

        if batch_idx % 50 == 0:

            print(
                f"[Laplace] Processed "
                f"{batch_idx + 1}/"
                f"{len(test_loader)}"
            )

    print(
        "\n[Laplace] Inference complete."
    )


if __name__ == "__main__":
    main()
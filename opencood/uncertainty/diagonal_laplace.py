import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters

try:
    from laplace import Laplace as TorchLaplace
except ImportError:  # handled with an actionable error when posterior is built
    TorchLaplace = None


class DiagonalLaplace:
    """
    Model-agnostic diagonal Laplace approximation for OpenCOOD detectors.

    Assumptions
    -----------
    Model output contains:
        output_dict["psm"] : classification logits
        output_dict["rm"]  : encoded box regression

    Typical workflow
    ----------------
    1. Load trained MAP model.
    2. Construct DiagonalLaplace(model).
    3. fit_empirical_diag(...) on TRAINING data.
    4. save Fisher/MAP parameters.
    5. set_posterior(alpha, prior_precision).
    6. Perform MC inference.

    No optimizer.step() is ever called.
    """

    def __init__(
        self,
        model: nn.Module,
        param_scope=("cls_head", "reg_head"),
        device="cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Freeze everything first.
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.param_names = []

        for name, p in self.model.named_parameters():

            if param_scope == "full":
                selected = True
            else:
                selected = any(
                    scope in name for scope in param_scope
                )

            if selected:
                p.requires_grad_(True)
                self.param_names.append(name)

        if len(self.param_names) == 0:
            raise ValueError(
                f"No parameters matched param_scope={param_scope}"
            )

        print("\n[Laplace] Selected parameters:")

        for name in self.param_names:
            print("   ", name)

        self.theta_map = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if name in self.param_names
        }

        self.fisher_diag = {
            name: torch.zeros_like(p)
            for name, p in self.model.named_parameters()
            if name in self.param_names
        }

        self.sigma = None
        self.alpha = None
        self.prior_precision = None
        self._torch_laplace = None

    # ---------------------------------------------------------
    # Fisher / curvature approximation
    # ---------------------------------------------------------

    def fit_empirical_diag(
        self,
        train_loader,
        criterion,
        to_device_fn,
        max_batches=None
    ):
        """
        Squared-gradient diagonal curvature approximation evaluated
        at theta_MAP.

        NOTE:
        For OpenCOOD's PointPillarLoss this uses gradients of the
        detector training objective (focal classification + regression
        loss). Therefore this should be described as a practical
        empirical diagonal curvature approximation rather than an
        exact Fisher of a fully specified likelihood.
        """

        print("\n[Laplace] Computing diagonal curvature...")

        self.model.eval()

        # reset
        for name in self.fisher_diag:
            self.fisher_diag[name].zero_()

        n_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):

            if (
                max_batches is not None
                and batch_idx >= max_batches
            ):
                break

            batch_data = to_device_fn(batch_data)

            ego_data = batch_data["ego"]

            self.model.zero_grad(set_to_none=True)

            output_dict = self.model(ego_data)

            loss = criterion(
                output_dict,
                ego_data["label_dict"]
            )

            loss.backward()

            with torch.no_grad():

                for name, p in self.model.named_parameters():

                    if (
                        name in self.fisher_diag
                        and p.grad is not None
                    ):
                        self.fisher_diag[name].add_(
                            p.grad.detach().pow(2)
                        )

            n_batches += 1

            if n_batches % 50 == 0:
                print(
                    f"[Laplace] batch={n_batches}, "
                    f"loss={loss.item():.6f}"
                )

        if n_batches == 0:
            raise RuntimeError(
                "No batches were processed while computing Fisher."
            )

        # Mean squared-gradient curvature
        for name in self.fisher_diag:
            self.fisher_diag[name].div_(n_batches)

        print(
            f"\n[Laplace] Curvature computed "
            f"from {n_batches} batches."
        )

        self.print_fisher_statistics()

    # ---------------------------------------------------------

    def print_fisher_statistics(self):

        print("\n========== Fisher statistics ==========")

        for name, f in self.fisher_diag.items():

            x = f.detach().flatten()

            print(f"\n{name}")
            print(f" min    : {x.min().item():.6e}")
            print(f" median : {x.median().item():.6e}")
            print(f" mean   : {x.mean().item():.6e}")
            print(f" max    : {x.max().item():.6e}")

        print("=======================================\n")

    # ---------------------------------------------------------
    # Posterior
    # ---------------------------------------------------------

    def set_posterior(
        self,
        alpha=0.01,
        prior_precision=1.0
    ):
        """Construct the diagonal Gaussian posterior with ``laplace-torch``.

        OpenCOOD uses a detector-specific focal + box-regression objective, which is
        not one of laplace-torch's built-in likelihoods. Curvature is therefore
        accumulated in :meth:`fit_empirical_diag` with the native OpenCOOD loss.
        This method hands that diagonal curvature to laplace-torch, which owns the
        posterior precision/covariance and parameter sampling.

        To preserve the historical thesis parameterization exactly,

            variance = alpha / (F + prior_precision),

        we store H = F / alpha and P0 = prior_precision / alpha in laplace-torch.
        """
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0.")
        if TorchLaplace is None:
            raise ImportError(
                "laplace-torch is required. "
                "Install it with `pip install laplace-torch`."
            )

        self.alpha = float(alpha)
        self.prior_precision = float(prior_precision)

        # alpha=0 is the deterministic MAP baseline retained for compatibility.
        if self.alpha == 0:
            self._torch_laplace = None
            self.sigma = {
                name: torch.zeros_like(v)
                for name, v in self.theta_map.items()
            }
            print(
                "\n[Laplace] alpha=0: "
                "deterministic MAP posterior baseline."
            )
            return

        # Only the selected parameters require gradients (set in __init__).
        # DiagLaplace therefore builds its posterior over exactly this
        # parameter set.
        la = TorchLaplace(
            self.model,
            likelihood="regression",
            subset_of_weights="all",
            hessian_structure="diag",
            prior_precision=self.prior_precision / self.alpha,
        )

        fisher_vec = torch.cat([
            self.fisher_diag[name].reshape(-1)
            for name in self.param_names
        ]).to(self.device)

        theta_vec = torch.cat([
            self.theta_map[name].reshape(-1)
            for name in self.param_names
        ]).to(self.device)

        if fisher_vec.numel() != la.n_params:
            raise RuntimeError(
                f"Curvature has {fisher_vec.numel()} entries "
                f"but laplace-torch selected "
                f"{la.n_params} parameters."
            )

        # Inject the detector-loss curvature.
        # Standard la.fit(loader) cannot be used here because OpenCOOD
        # batches/targets and its joint detection loss are not a built-in
        # laplace-torch likelihood. Posterior algebra below is laplace-torch.
        la.H = fisher_vec / self.alpha
        la.mean = theta_vec
        la.n_data = 1
        la.n_outputs = 1
        la.loss = torch.zeros((), device=self.device)

        self._torch_laplace = la

        posterior_std_vec = la.posterior_scale.detach()

        self.sigma = {}
        offset = 0

        for name in self.param_names:
            ref = self.theta_map[name]
            n = ref.numel()

            self.sigma[name] = (
                posterior_std_vec[
                    offset:offset + n
                ]
                .view_as(ref)
                .clone()
            )

            offset += n

        print(
            "\n[Laplace] Posterior constructed "
            "with laplace-torch:"
        )
        print(" alpha           =", alpha)
        print(" prior precision =", prior_precision)

        self.print_sigma_statistics()

    # ---------------------------------------------------------

    def print_sigma_statistics(self):

        print(
            "\n========== Posterior std statistics =========="
        )

        for name, sigma in self.sigma.items():

            x = sigma.detach().flatten()
            theta = self.theta_map[name].detach().flatten()

            print(f"\n{name}")

            print(
                f" parameter std : "
                f"{theta.std().item():.6e}"
            )

            print(
                f" sigma median  : "
                f"{x.median().item():.6e}"
            )

            print(
                f" sigma mean    : "
                f"{x.mean().item():.6e}"
            )

            print(
                f" sigma max     : "
                f"{x.max().item():.6e}"
            )

        print(
            "==============================================\n"
        )

    # ---------------------------------------------------------
    # Weight sampling
    # ---------------------------------------------------------

    @torch.no_grad()
    def sample_weights(self):

        if self.sigma is None:
            raise RuntimeError(
                "Call set_posterior() first."
            )

        if self.alpha == 0:
            self.restore_map_weights()
            return

        if self._torch_laplace is None:
            raise RuntimeError(
                "laplace-torch posterior is not initialized."
            )

        sample = self._torch_laplace.sample(
            n_samples=1
        )[0]

        selected_params = [
            p
            for p in self.model.parameters()
            if p.requires_grad
        ]

        vector_to_parameters(
            sample,
            selected_params
        )

    # ---------------------------------------------------------

    @torch.no_grad()
    def restore_map_weights(self):

        for name, p in self.model.named_parameters():

            if name in self.theta_map:
                p.copy_(
                    self.theta_map[name]
                )

    # ---------------------------------------------------------
    # MC forward
    # ---------------------------------------------------------

    @torch.no_grad()
    def predict_raw(
        self,
        ego_data,
        T=10
    ):

        if T < 1:
            raise ValueError(
                "T must be >= 1."
            )

        if self.sigma is None:
            raise RuntimeError(
                "Posterior not constructed."
            )

        psm_samples = []
        rm_samples = []

        try:

            for _ in range(T):

                self.sample_weights()

                output = self.model(
                    ego_data
                )

                psm_samples.append(
                    output["psm"].detach()
                )

                rm_samples.append(
                    output["rm"].detach()
                )

        finally:

            # Never leave random sampled weights in model.
            self.restore_map_weights()

        psm_stack = torch.stack(
            psm_samples,
            dim=0
        )

        rm_stack = torch.stack(
            rm_samples,
            dim=0
        )

        # -----------------------------------------------
        # Classification
        # -----------------------------------------------

        cls_prob_samples = torch.sigmoid(
            psm_stack
        )

        mean_cls_prob = cls_prob_samples.mean(
            dim=0
        )

        var_cls_prob = cls_prob_samples.var(
            dim=0,
            unbiased=False
        )

        # -----------------------------------------------
        # Predictive entropy
        # -----------------------------------------------

        predictive_entropy = self.binary_entropy(
            mean_cls_prob
        )

        expected_entropy = self.binary_entropy(
            cls_prob_samples
        ).mean(dim=0)

        # Epistemic classification uncertainty
        mutual_information = (
            predictive_entropy
            - expected_entropy
        ).clamp_min(0.0)

        # -----------------------------------------------
        # Regression moment matching
        # -----------------------------------------------

        mean_reg_encoded = rm_stack.mean(
            dim=0
        )

        var_reg_encoded = rm_stack.var(
            dim=0,
            unbiased=False
        )

        return {
            "mean_cls_prob":
                mean_cls_prob,

            "var_cls_prob":
                var_cls_prob,

            "predictive_entropy":
                predictive_entropy,

            "mutual_information":
                mutual_information,

            "mean_reg_encoded":
                mean_reg_encoded,

            "var_reg_encoded":
                var_reg_encoded,

            "cls_prob_samples":
                cls_prob_samples,

            "reg_samples":
                rm_stack
        }

    # ---------------------------------------------------------

    @staticmethod
    def binary_entropy(
        prob,
        eps=1e-8
    ):

        p = prob.clamp(
            min=eps,
            max=1.0 - eps
        )

        return -(
            p * torch.log(p)
            +
            (1.0 - p)
            * torch.log(1.0 - p)
        )

    # ---------------------------------------------------------
    # Convert Bayesian mean probability back to logits
    # ---------------------------------------------------------

    @staticmethod
    def probability_to_logit(
        probability,
        eps=1e-6
    ):
        """
        Allows normal OpenCOOD postprocessor to be reused.

        sigmoid(logit(p)) == p
        """

        p = probability.clamp(
            min=eps,
            max=1.0 - eps
        )

        return torch.log(
            p / (1.0 - p)
        )

    # ---------------------------------------------------------

    def make_opencood_output(
        self,
        result
    ):
        """
        Create ordinary OpenCOOD output dictionary.

        The Bayesian classification mean is a probability.
        OpenCOOD normally expects psm to be logits.

        Therefore convert probability -> equivalent logit.
        """

        bayes_psm = self.probability_to_logit(
            result["mean_cls_prob"]
        )

        return {
            "psm":
                bayes_psm,

            "rm":
                result["mean_reg_encoded"]
        }

    # ---------------------------------------------------------
    # Save/load
    # ---------------------------------------------------------

    def save(
        self,
        path
    ):

        torch.save(
            {
                "theta_map":
                    self.theta_map,

                "fisher_diag":
                    self.fisher_diag,

                "param_names":
                    self.param_names,

                "alpha":
                    self.alpha,

                "prior_precision":
                    self.prior_precision,

                "sigma":
                    self.sigma
            },
            path
        )

        print(
            f"[Laplace] Saved to {path}"
        )

    # ---------------------------------------------------------

    def load(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.theta_map = checkpoint[
            "theta_map"
        ]

        self.fisher_diag = checkpoint[
            "fisher_diag"
        ]

        self.param_names = checkpoint[
            "param_names"
        ]

        self.alpha = checkpoint.get(
            "alpha",
            None
        )

        self.prior_precision = checkpoint.get(
            "prior_precision",
            None
        )

        self.sigma = checkpoint.get(
            "sigma",
            None
        )

        # Move loaded tensors to current device.
        self.theta_map = {
            k: v.to(self.device)
            for k, v in self.theta_map.items()
        }

        self.fisher_diag = {
            k: v.to(self.device)
            for k, v in self.fisher_diag.items()
        }

        if self.sigma is not None:

            self.sigma = {
                k: v.to(self.device)
                for k, v in self.sigma.items()
            }

        self._torch_laplace = None

        self.restore_map_weights()

        print(
            f"[Laplace] Loaded {path}"
        )
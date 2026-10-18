import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters

try:
    from laplace import Laplace as TorchLaplace
except ImportError:  # handled with an actionable error when posterior is built
    TorchLaplace = None


# Named parameter scopes (module-name prefixes). BatchNorm parameters are
# never included. "last_module" = the last block before the heads + heads,
# the "last-module" setting of Yun & Liu, CoRL 2022; which block that is
# depends on the architecture (see last_module_prefixes).
SCOPES = {
    "heads": ("cls_head", "reg_head"),
    "last_module": ("shrink_conv", "cls_head", "reg_head"),
    "full": "full",
}


def last_module_prefixes(model):
    """
    Module prefixes of the "last module" scope for a given detector.

    - Fusion block with trainable weights between shrink_conv and the heads
      (CoBEVT: FAX swap attention; also V2VNet, V2X-ViT): that fusion block +
      heads — it is the last block before the heads.
    - shrink_conv models whose fusion has no weights (F-Cooper: max fusion):
      shrink_conv + heads (unchanged F-Cooper definition).
    - AttFuse (AttBEVBackbone: attention fusion inside the backbone at every
      scale): the upsampling deblocks after the fusion, which feed the heads.
    Other models: pass the modules explicitly (scope_modules).
    """
    fusion = getattr(model, "fusion_net", None)
    if fusion is not None and any(True for _ in fusion.parameters()):
        return ("fusion_net", "cls_head", "reg_head")
    if getattr(model, "shrink_flag", False):
        return SCOPES["last_module"]
    backbone = getattr(model, "backbone", None)
    if hasattr(backbone, "deblocks") and hasattr(backbone, "fuse_modules"):
        return ("backbone.deblocks", "cls_head", "reg_head")
    raise ValueError(
        f"No 'last_module' definition for {type(model).__name__}; "
        "pass the module prefixes with --scope_modules."
    )


def resolve_scope(model, param_scope, scope_modules=None):
    """
    Module prefixes (tuple) or "full" for a scope name.
    scope_modules (comma-separated string or sequence of prefixes) overrides
    the named scope, e.g. "backbone.deblocks,backbone.fuse_modules,cls_head,reg_head".
    """
    if scope_modules:
        if isinstance(scope_modules, str):
            scope_modules = [s.strip() for s in scope_modules.split(",") if s.strip()]
        return tuple(scope_modules)
    if param_scope == "last_module":
        return last_module_prefixes(model)
    if isinstance(param_scope, str) and param_scope in SCOPES:
        return SCOPES[param_scope]
    return param_scope


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
        device="cuda",
        scope_modules=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        param_scope = resolve_scope(self.model, param_scope, scope_modules)

        # Freeze everything first.
        for p in self.model.parameters():
            p.requires_grad_(False)

        bn_params = {
            f"{mod_name}.{p_name}"
            for mod_name, mod in self.model.named_modules()
            if isinstance(mod, nn.modules.batchnorm._BatchNorm)
            for p_name, _ in mod.named_parameters(recurse=False)
        }

        self.param_names = []

        for name, p in self.model.named_parameters():

            if name in bn_params:
                selected = False
            elif param_scope == "full":
                selected = True
            else:
                selected = any(
                    name.startswith(scope + ".") for scope in param_scope
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

        # Number of training examples the curvature is summed over, and the
        # homoscedastic observation noise per regression dimension
        # [x, y, z, h, w, l, sin(dyaw)] estimated from MAP training residuals.
        self.n_data = None
        self.noise_var = None

    # ---------------------------------------------------------
    # Fisher / curvature approximation
    # ---------------------------------------------------------

    def fit_empirical_diag(
        self,
        train_loader,
        criterion,
        to_device_fn,
        max_batches=None,
        n_data=None
    ):
        """
        Empirical-Fisher diagonal curvature evaluated at theta_MAP:

            F = sum_i (dL_i / dtheta)^2      over all N training examples

        The Laplace posterior precision is F + prior_precision, so F must be
        the SUM of PER-EXAMPLE squared gradients. The loader must therefore
        use batch_size=1. If only a subset of the data is processed
        (max_batches), the sum is extrapolated to n_data examples.

        The same pass also estimates the homoscedastic regression noise
        (mean squared MAP residual on positive anchors, yaw as sin(dyaw)),
        which is the aleatoric part of the predictive variance for a model
        without its own variance head.

        NOTE:
        For OpenCOOD's PointPillarLoss this uses gradients of the
        detector training objective (focal classification + regression
        loss). Therefore this should be described as a practical
        empirical diagonal curvature approximation rather than an
        exact Fisher of a fully specified likelihood.
        """

        if getattr(train_loader, "batch_size", 1) != 1:
            raise ValueError(
                "fit_empirical_diag needs per-example gradients: "
                "use a DataLoader with batch_size=1."
            )

        print("\n[Laplace] Computing diagonal curvature...")

        self.model.eval()

        # reset
        for name in self.fisher_diag:
            self.fisher_diag[name].zero_()

        resid_sq_sum = torch.zeros(7, device=self.device, dtype=torch.float64)
        resid_count = 0

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

                # MAP residuals on positive anchors -> observation noise
                label_dict = ego_data["label_dict"]
                rm = output_dict["rm"].permute(0, 2, 3, 1).reshape(-1, 7)
                targets = label_dict["targets"].reshape(-1, 7)
                pos = label_dict["pos_equal_one"].reshape(-1) > 0
                if pos.any():
                    resid = rm[pos] - targets[pos]
                    resid[:, 6] = torch.sin(resid[:, 6])
                    resid = resid[torch.isfinite(resid).all(dim=1)]
                    resid_sq_sum += resid.double().pow(2).sum(dim=0)
                    resid_count += resid.shape[0]

            n_batches += 1

            if n_batches % 500 == 0:
                print(
                    f"[Laplace] batch={n_batches}, "
                    f"loss={loss.item():.6f}"
                )

        if n_batches == 0:
            raise RuntimeError(
                "No batches were processed while computing Fisher."
            )

        # Sum over the training set (extrapolated if only a subset was used)
        self.n_data = int(n_data) if n_data is not None else n_batches
        scale = self.n_data / n_batches
        for name in self.fisher_diag:
            self.fisher_diag[name].mul_(scale)

        if resid_count > 0:
            self.noise_var = (resid_sq_sum / resid_count).float()
        else:
            self.noise_var = None

        print(
            f"\n[Laplace] Curvature summed over {n_batches} examples "
            f"(scaled to n_data={self.n_data})."
        )
        if self.noise_var is not None:
            print(
                "[Laplace] Observation noise variance "
                "[x,y,z,h,w,l,sin(yaw)]:",
                [round(v, 6) for v in self.noise_var.tolist()]
            )

        self.print_fisher_statistics()

    # ---------------------------------------------------------
    # GGN curvature for the (linear, 1x1-conv) detection heads
    # ---------------------------------------------------------

    @torch.no_grad()
    def fit_ggn_diag(
        self,
        train_loader,
        to_device_fn,
        max_batches=None,
        n_data=None,
        logvar_min=-7.0,
        logvar_max=7.0,
    ):
        """
        Exact diagonal generalized Gauss-Newton (GGN) curvature J^T Lambda J
        for the cls_head / reg_head 1x1 convolutions, summed over the
        training set at theta_MAP.

        For a 1x1 conv the output is linear in the weights, so for output
        channel o and input channel c

            GGN[o, c] = sum_n sum_(h,w) Lambda_o(n,h,w) * x_c(n,h,w)^2
            GGN_bias[o] = sum_n sum_(h,w) Lambda_o(n,h,w)

        Likelihoods
          classification: Bernoulli (sigmoid), Lambda = p (1 - p), on anchors
                          labelled positive or negative (ignored anchors skipped)
          regression:     Gaussian on positive anchors, Lambda = 1 / sigma^2,
                          sigma^2 = exp(rm_logvar) for variance-network models,
                          otherwise the homoscedastic training noise per dim.

        Unlike the empirical Fisher of the focal loss (whose gradients vanish
        at the MAP for the many easy anchors), the GGN reflects how sensitive
        the outputs are to the weights. Recommended by Daxberger et al.,
        "Laplace Redux", NeurIPS 2021.
        """
        head_names = {"cls_head", "reg_head"}
        modules = dict(self.model.named_modules())
        for h in head_names:
            if h not in modules:
                raise ValueError(f"model has no module '{h}'")
            w = modules[h].weight
            if w.dim() != 4 or w.shape[-1] != 1 or w.shape[-2] != 1:
                raise ValueError(f"{h} is not a 1x1 conv; GGN closed form needs it")
        for name in self.param_names:
            if name.split(".")[0] not in head_names:
                raise ValueError("fit_ggn_diag supports param_scope='heads' only")

        feats = {}
        hooks = [
            modules[h].register_forward_hook(
                lambda m, inp, out, h=h: feats.__setitem__(h, inp[0].detach())
            )
            for h in head_names
        ]

        print("\n[Laplace] Computing diagonal GGN curvature...")
        self.model.eval()

        cls_w = torch.zeros_like(self.fisher_diag["cls_head.weight"]).squeeze(-1).squeeze(-1).double()
        cls_b = torch.zeros(cls_w.shape[0], device=self.device, dtype=torch.float64)
        reg_w = torch.zeros_like(self.fisher_diag["reg_head.weight"]).squeeze(-1).squeeze(-1).double()
        reg_b = torch.zeros(reg_w.shape[0], device=self.device, dtype=torch.float64)

        resid_sq_sum = torch.zeros(7, device=self.device, dtype=torch.float64)
        resid_count = 0
        heteroscedastic = None
        n_batches = 0

        try:
            for batch_idx, batch_data in enumerate(train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                ego_data = to_device_fn(batch_data)["ego"]
                out = self.model(ego_data)
                label_dict = ego_data["label_dict"]

                psm = out["psm"]                       # [B, A, H, W]
                rm = out["rm"]                         # [B, 7A, H, W]
                B, A, H, W = psm.shape
                x2_cls = feats["cls_head"].double().pow(2)   # [B, C, H, W]
                x2_reg = feats["reg_head"].double().pow(2)

                pos = label_dict["pos_equal_one"].reshape(B, H, W, A).permute(0, 3, 1, 2) > 0
                neg = label_dict["neg_equal_one"].reshape(B, H, W, A).permute(0, 3, 1, 2) > 0

                # ---- classification: Bernoulli GGN ----
                p = torch.sigmoid(psm).double()
                lam_cls = p * (1 - p) * (pos | neg).double()          # [B, A, H, W]
                cls_w += torch.einsum("bahw,bchw->ac", lam_cls, x2_cls)
                cls_b += lam_cls.sum(dim=(0, 2, 3))

                # ---- regression: Gaussian GGN on positives ----
                pos_o = pos.repeat_interleave(7, dim=1).double()      # [B, 7A, H, W]
                if "rm_logvar" in out:
                    heteroscedastic = True
                    prec = torch.exp(
                        -out["rm_logvar"].clamp(logvar_min, logvar_max)
                    ).double()
                    lam_reg = pos_o * prec
                else:
                    heteroscedastic = False
                    lam_reg = pos_o        # divided by noise variance at the end
                reg_w += torch.einsum("bohw,bchw->oc", lam_reg, x2_reg)
                reg_b += lam_reg.sum(dim=(0, 2, 3))

                # ---- MAP residuals -> homoscedastic observation noise ----
                rm_flat = rm.permute(0, 2, 3, 1).reshape(-1, 7)
                tgt = label_dict["targets"].reshape(-1, 7)
                pmask = label_dict["pos_equal_one"].reshape(-1) > 0
                if pmask.any():
                    resid = rm_flat[pmask] - tgt[pmask]
                    resid[:, 6] = torch.sin(resid[:, 6])
                    resid = resid[torch.isfinite(resid).all(dim=1)]
                    resid_sq_sum += resid.double().pow(2).sum(dim=0)
                    resid_count += resid.shape[0]

                n_batches += 1
                if n_batches % 500 == 0:
                    print(f"[Laplace] GGN batch={n_batches}")
        finally:
            for hk in hooks:
                hk.remove()

        if n_batches == 0:
            raise RuntimeError("No batches were processed while computing GGN.")

        self.noise_var = (resid_sq_sum / max(resid_count, 1)).float()
        if not heteroscedastic:
            inv_noise = (1.0 / self.noise_var.double().clamp_min(1e-8))
            n_anchor = reg_w.shape[0] // 7
            inv_noise = inv_noise.repeat(n_anchor)            # channel = a*7 + d
            reg_w *= inv_noise[:, None]
            reg_b *= inv_noise

        self.n_data = int(n_data) if n_data is not None else n_batches
        scale = self.n_data / n_batches
        shapes = {n: self.fisher_diag[n].shape for n in self.fisher_diag}
        self.fisher_diag = {
            "cls_head.weight": (cls_w * scale).float().view(shapes["cls_head.weight"]),
            "cls_head.bias": (cls_b * scale).float().view(shapes["cls_head.bias"]),
            "reg_head.weight": (reg_w * scale).float().view(shapes["reg_head.weight"]),
            "reg_head.bias": (reg_b * scale).float().view(shapes["reg_head.bias"]),
        }
        self.curvature_type = "ggn"

        print(
            f"\n[Laplace] GGN summed over {n_batches} batches "
            f"(scaled to n_data={self.n_data}); regression noise: "
            f"{'predicted exp(rm_logvar)' if heteroscedastic else 'homoscedastic'}"
        )
        print(
            "[Laplace] Observation noise variance [x,y,z,h,w,l,sin(yaw)]:",
            [round(v, 6) for v in self.noise_var.tolist()]
        )
        self.print_fisher_statistics()

    # ---------------------------------------------------------
    # Sampled (Monte Carlo) diagonal Fisher for any parameter scope
    # ---------------------------------------------------------

    def _estimate_noise_var(self, dataset, collate_fn, to_device_fn,
                            n_frames, seed=0):
        """Homoscedastic regression noise per encoded dim (sin for yaw):
        mean squared MAP residual on positive anchors of a random subset of
        training frames."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g)[:n_frames].tolist()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, idx), batch_size=1,
            shuffle=False, num_workers=4, collate_fn=collate_fn)
        sq = torch.zeros(7, device=self.device, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for batch_data in loader:
                ego = to_device_fn(batch_data)["ego"]
                out = self.model(ego)
                ld = ego["label_dict"]
                rm = out["rm"].permute(0, 2, 3, 1).reshape(-1, 7)
                tgt = ld["targets"].reshape(-1, 7)
                pos = ld["pos_equal_one"].reshape(-1) > 0
                if pos.any():
                    r = rm[pos] - tgt[pos]
                    r[:, 6] = torch.sin(r[:, 6])
                    r = r[torch.isfinite(r).all(dim=1)]
                    sq += r.double().pow(2).sum(dim=0)
                    count += r.shape[0]
        return (sq / max(count, 1)).float()

    def fit_sampled_fisher_diag(
        self,
        train_loader,
        to_device_fn,
        max_batches=None,
        n_data=None,
        n_samples=1,
        noise_frames=500,
        logvar_min=-7.0,
        logvar_max=7.0,
        seed=0,
    ):
        """
        Diagonal Fisher by Monte Carlo label sampling ("standard Fisher" of
        Yun & Liu, CoRL 2022), for ANY parameter scope:

            F = sum_n E_{y ~ p(y | x_n, theta_MAP)} [ (d log p(y | x_n) / d theta)^2 ]

        with the same likelihood as fit_ggn_diag:
          classification: Bernoulli(sigmoid(logit)) on anchors labelled
                          positive or negative (ignored anchors skipped);
          regression:     Gaussian N(rm, sigma^2) on positive anchors,
                          sigma^2 = exp(rm_logvar) for variance-network models,
                          otherwise the homoscedastic training noise per dim.

        Labels are drawn from the model, so the residual (p - y) or
        (rm - y) / sigma^2 has zero mean and is independent across anchors.
        Hence one squared per-frame gradient is an UNBIASED estimate of the
        exact diagonal GGN sum_anchors Lambda (d f / d theta)^2: the cross
        terms between anchors vanish in expectation. Unlike the empirical
        Fisher of the focal loss, it does not collapse at the MAP.
        Cost: one forward and n_samples backward passes per frame.
        """
        if getattr(train_loader, "batch_size", 1) != 1:
            raise ValueError("fit_sampled_fisher_diag needs batch_size=1.")

        self.model.eval()
        gen = torch.Generator(device=self.device).manual_seed(seed)

        # Regression noise is needed before sampling (baseline models).
        noise_var = self._estimate_noise_var(
            train_loader.dataset, train_loader.collate_fn, to_device_fn,
            noise_frames, seed=seed)
        print("[Laplace] Pre-pass noise variance [x,y,z,h,w,l,sin(yaw)] "
              f"({noise_frames} random frames):",
              [round(v, 6) for v in noise_var.tolist()])

        fisher = {n: torch.zeros_like(self.fisher_diag[n], dtype=torch.float64)
                  for n in self.param_names}
        params = dict(self.model.named_parameters())
        resid_sq_sum = torch.zeros(7, device=self.device, dtype=torch.float64)
        resid_count = 0
        heteroscedastic = None
        n_batches = 0

        print(f"\n[Laplace] Computing sampled diagonal Fisher "
              f"({n_samples} label sample(s) per frame)...")

        for batch_idx, batch_data in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            ego = to_device_fn(batch_data)["ego"]
            out = self.model(ego)
            ld = ego["label_dict"]

            psm = out["psm"]                                  # [1, A, H, W]
            rm = out["rm"]                                    # [1, 7A, H, W]
            B, A, H, W = psm.shape
            pos = ld["pos_equal_one"].reshape(B, H, W, A).permute(0, 3, 1, 2) > 0
            neg = ld["neg_equal_one"].reshape(B, H, W, A).permute(0, 3, 1, 2) > 0
            cls_mask = (pos | neg).float()
            pos_o = pos.repeat_interleave(7, dim=1)           # [1, 7A, H, W]

            if "rm_logvar" in out:
                heteroscedastic = True
                var = torch.exp(
                    out["rm_logvar"].detach().clamp(logvar_min, logvar_max))
            else:
                heteroscedastic = False
                var = noise_var.repeat(A).view(1, -1, 1, 1).expand_as(rm)

            prob = torch.sigmoid(psm.detach())
            rm_pos = rm[pos_o]
            std_pos = var[pos_o].sqrt()

            for t in range(n_samples):
                y_cls = torch.bernoulli(prob, generator=gen)
                y_reg = rm_pos.detach() + std_pos * torch.randn(
                    rm_pos.shape, device=self.device, generator=gen)
                nll = (torch.nn.functional.binary_cross_entropy_with_logits(
                           psm, y_cls, reduction="none") * cls_mask).sum()
                nll = nll + ((rm_pos - y_reg).pow(2)
                             / (2.0 * std_pos.pow(2))).sum()
                grads = torch.autograd.grad(
                    nll, [params[n] for n in self.param_names],
                    retain_graph=t < n_samples - 1, allow_unused=True)
                for n, g in zip(self.param_names, grads):
                    if g is not None:
                        fisher[n] += g.detach().double().pow(2) / n_samples

            with torch.no_grad():
                rm_flat = rm.detach().permute(0, 2, 3, 1).reshape(-1, 7)
                tgt = ld["targets"].reshape(-1, 7)
                pmask = ld["pos_equal_one"].reshape(-1) > 0
                if pmask.any():
                    r = rm_flat[pmask] - tgt[pmask]
                    r[:, 6] = torch.sin(r[:, 6])
                    r = r[torch.isfinite(r).all(dim=1)]
                    resid_sq_sum += r.double().pow(2).sum(dim=0)
                    resid_count += r.shape[0]

            n_batches += 1
            if n_batches % 500 == 0:
                print(f"[Laplace] sampled-Fisher batch={n_batches}")

        if n_batches == 0:
            raise RuntimeError("No batches were processed.")

        # Stored noise (used for the aleatoric term at inference) comes from
        # all processed frames; the curvature used the pre-pass estimate.
        self.noise_var = (resid_sq_sum / max(resid_count, 1)).float()
        self.n_data = int(n_data) if n_data is not None else n_batches
        scale = self.n_data / n_batches
        self.fisher_diag = {n: (f * scale).float() for n, f in fisher.items()}
        self.curvature_type = f"sampled_fisher(T={n_samples})"

        print(
            f"\n[Laplace] Sampled Fisher summed over {n_batches} frames "
            f"(scaled to n_data={self.n_data}); regression noise: "
            f"{'predicted exp(rm_logvar)' if heteroscedastic else 'homoscedastic'}"
        )
        print("[Laplace] Observation noise variance [x,y,z,h,w,l,sin(yaw)]:",
              [round(v, 6) for v in self.noise_var.tolist()])
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
    # Prior precision by marginal likelihood (empirical Bayes)
    # ---------------------------------------------------------

    def module_groups(self):
        """Parameter names grouped by the module that owns them
        (e.g. 'reg_head' -> [reg_head.weight, reg_head.bias])."""
        groups = {}
        for name in self.param_names:
            groups.setdefault(name.rsplit(".", 1)[0], []).append(name)
        return groups

    @torch.no_grad()
    def optimize_prior_precision(self, alpha=1.0, grid=None, per_module=True):
        """
        Prior precision by maximising the Laplace log marginal likelihood.

        per_module=True (default): one P0 per module (layer-wise prior,
        Laplace Redux). The objective is a sum over parameters, so it
        separates by module and each module's P0 is optimised on its own.
        Needed when the scope mixes modules of very different size and weight
        scale (e.g. shrink_conv with 1.5M weights and heads with 4k).
        Returns a dict {module: P0}; with a single module this equals the
        scalar result.
        per_module=False: one scalar P0 for all selected parameters.
        """
        if per_module:
            result = {}
            for module, names in self.module_groups().items():
                result[module] = self._optimize_prior_group(names, alpha, grid)
            for module, p0 in result.items():
                print(f"[Laplace] Marginal-likelihood prior precision "
                      f"{module}: {p0:.4e} (alpha={alpha})")
            return result
        best = self._optimize_prior_group(self.param_names, alpha, grid)
        print(f"[Laplace] Marginal-likelihood prior precision: {best:.4e} "
              f"(alpha={alpha})")
        return best

    @torch.no_grad()
    def _optimize_prior_group(self, names, alpha=1.0, grid=None):
        """
        Pick a scalar prior precision P0 by maximising the Laplace log
        marginal likelihood. For a diagonal posterior with H = F / alpha and
        prior precision P0 / alpha (the parameterisation of set_posterior),
        the terms that depend on P0 are

            -0.5 * (P0 / alpha) * ||theta_MAP||^2
            +0.5 * sum_i log( P0 / (F_i + P0) )

        This uses only training data, so no test data is involved.
        """
        F = torch.cat([
            self.fisher_diag[name].reshape(-1).double()
            for name in names
        ])
        theta_sq = sum(
            self.theta_map[name].double().pow(2).sum()
            for name in names
        )

        if grid is None:
            grid = torch.logspace(-6, 8, 561, dtype=torch.float64)
        grid = grid.to(F.device)

        logml = torch.stack([
            -0.5 * (p0 / alpha) * theta_sq
            + 0.5 * torch.log(p0 / (F + p0)).sum()
            for p0 in grid
        ])
        return float(grid[int(torch.argmax(logml))])

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
        p0_values = (list(prior_precision.values())
                     if isinstance(prior_precision, dict)
                     else [prior_precision])
        if min(p0_values) <= 0:
            raise ValueError("prior_precision must be > 0.")
        if TorchLaplace is None:
            raise ImportError(
                "laplace-torch is required. "
                "Install it with `pip install laplace-torch`."
            )

        self.alpha = float(alpha)
        if isinstance(prior_precision, dict):
            # per-module prior -> per-parameter prior vector
            owner = {n: m for m, ns in self.module_groups().items() for n in ns}
            missing = set(owner.values()) - set(prior_precision)
            if missing:
                raise ValueError(f"prior_precision has no value for {missing}")
            self.prior_precision = {m: float(v)
                                    for m, v in prior_precision.items()}
            prior_vec = torch.cat([
                torch.full((self.theta_map[n].numel(),),
                           self.prior_precision[owner[n]],
                           device=self.device)
                for n in self.param_names
            ])
        else:
            self.prior_precision = float(prior_precision)
            prior_vec = self.prior_precision

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
            prior_precision=prior_vec / self.alpha,
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
                    self.sigma,

                "n_data":
                    self.n_data,

                "noise_var":
                    self.noise_var,

                "curvature_type":
                    getattr(self, "curvature_type", "ef")
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

        if list(checkpoint["param_names"]) != list(self.param_names):
            raise ValueError(
                "Curvature file was computed for a different parameter scope:\n"
                f"  file:      {len(checkpoint['param_names'])} tensors, e.g. "
                f"{checkpoint['param_names'][:3]}\n"
                f"  requested: {len(self.param_names)} tensors, e.g. "
                f"{self.param_names[:3]}\n"
                "Use the same --param_scope as in compute_laplace.py."
            )
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

        self.n_data = checkpoint.get("n_data", None)
        self.curvature_type = checkpoint.get("curvature_type", "ef")
        self.noise_var = checkpoint.get("noise_var", None)
        if self.noise_var is not None:
            self.noise_var = self.noise_var.to(self.device)

        if self.n_data is None:
            print(
                "[Laplace] WARNING: curvature file has no n_data -> it was "
                "made by the old batch-averaged Fisher, which is far too "
                "small (posterior = prior). Recompute it with "
                "compute_laplace.py."
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
import torch
import torch.nn as nn
import numpy as np


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise weighted Smooth L1 loss.

    Same idea as the original PointPillar loss:
        input  : predicted box values
        target : target box values
        weights: anchor-wise positive weights

    Returns unreduced loss with shape:
        (B, #anchors, #codes)
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.code_weights = None
        if code_weights is not None:
            self.code_weights = torch.from_numpy(
                np.array(code_weights, dtype=np.float32)
            )

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target

        if self.code_weights is not None:
            code_weights = self.code_weights.to(input.device)
            diff = diff * code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        if weights is not None:
            assert weights.shape[0] == loss.shape[0]
            assert weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarVNLoss(nn.Module):

    def __init__(self, args):
        super(PointPillarVNLoss, self).__init__()

        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']

        # Main probabilistic regression loss.
        # Recommended:
        #   bbox_cov_loss: negative_log_likelihood
        # Optional:
        #   bbox_cov_loss: energy_loss
        # Debug:
        #   bbox_cov_loss: smooth_l1
        self.bbox_cov_loss = args.get('bbox_cov_loss', 'negative_log_likelihood')
        self.num_samples = int(args.get('num_samples', 50))

        # Clamp log variance for numerical stability.
        self.log_var_min = float(args.get('log_var_min', -7.0))
        self.log_var_max = float(args.get('log_var_max', 7.0))

        # Optional warmup/annealing.
        # At the beginning, train like normal PointPillar.
        # Then gradually switch to probabilistic loss.
        self.use_probabilistic_annealing = bool(
            args.get('use_probabilistic_annealing', True)
        )
        self.annealing_steps = int(args.get('annealing_steps', 5000))
        self.current_step = 0

        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        psm = output_dict['psm']
        targets = target_dict['targets']


        if 'rmean' not in output_dict:
            raise KeyError(
                "output_dict does not contain 'rmean'. "
                "For the 4-head VN model, use reg_mean_head output as 'rmean'."
            )

        if 'rlogvar' not in output_dict:
            raise KeyError(
                "output_dict does not contain 'rlogvar'. "
                "For the 4-head VN model, var_head should output log variance as 'rlogvar'."
            )

        rmean = output_dict['rmean']
        rlogvar = output_dict['rlogvar']

        # -------------------------
        # Classification loss
        # -------------------------
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        box_cls_labels_neg = target_dict['neg_equal_one']
        box_cls_labels_neg = box_cls_labels_neg.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels_neg > 0

        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights = reg_weights / torch.clamp(pos_normalizer, min=1.0)
        cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)

        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype,
            device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(
            cls_preds,
            one_hot_targets,
            weights=cls_weights
        )
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # -------------------------
        # Regression mean + log variance
        # -------------------------
        # rmean: [B, A, 7, H, W] -> [B, H, W, A, 7]
        rmean = rmean.permute(0, 3, 4, 1, 2).contiguous()

        # [B, H*W*A, 7]
        rmean = rmean.view(rmean.size(0), -1, 7)

        # rlogvar: [B, A, 7, H, W] -> [B, H, W, A, 7]
        rlogvar = rlogvar.permute(0, 3, 4, 1, 2).contiguous()

        # [B, H*W*A, 7]
        rlogvar = rlogvar.view(rlogvar.size(0), -1, 7)
        rlogvar = torch.clamp(
            rlogvar,
            min=self.log_var_min,
            max=self.log_var_max
        )
        print("rmean:", rmean.shape)
        print("rlogvar:", rlogvar.shape)
        print("targets:", targets.shape)
        print("reg_weights:", reg_weights.shape)

        targets = targets.view(targets.size(0), -1, 7)
        if not hasattr(self, "_printed_target_shapes"):
            print("rmean:", rmean.shape)
            print("rlogvar:", rlogvar.shape)
            print("targets before reshape:", target_dict['targets'].shape)
            print("targets after reshape:", targets.shape)
            print("reg_weights:", reg_weights.shape)
            self._printed_target_shapes = True

        assert rmean.shape == targets.shape
        assert rlogvar.shape == targets.shape

        # Same yaw/sin handling as original PointPillar loss.
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            rmean,
            targets
        )

        # Standard regression loss.
        # This is used for:
        #   1. smooth_l1 mode
        #   2. annealing warmup
        standard_loc_loss_src = self.reg_loss_func(
            box_preds_sin,
            reg_targets_sin,
            weights=reg_weights
        )
        standard_reg_loss = standard_loc_loss_src.sum() / rmean.shape[0]

        # Probabilistic regression loss.
        if self.bbox_cov_loss == 'smooth_l1':
            probabilistic_reg_loss = standard_reg_loss

        elif self.bbox_cov_loss == 'negative_log_likelihood':
            probabilistic_reg_loss = self.nll_regression_loss(
                mean=box_preds_sin,
                target=reg_targets_sin,
                log_var=rlogvar,
                weights=reg_weights
            )

        elif self.bbox_cov_loss == 'energy_loss':
            probabilistic_reg_loss = self.energy_score_regression_loss(
                mean=box_preds_sin,
                target=reg_targets_sin,
                log_var=rlogvar,
                weights=reg_weights
            )

        else:
            raise ValueError(
                f"Unknown bbox_cov_loss={self.bbox_cov_loss}. "
                "Use 'negative_log_likelihood', 'energy_loss', or 'smooth_l1'."
            )

        if self.use_probabilistic_annealing:
            prob_weight = self.get_probabilistic_loss_weight()
            reg_loss = (1.0 - prob_weight) * standard_reg_loss + \
                       prob_weight * probabilistic_reg_loss
        else:
            prob_weight = 1.0
            reg_loss = probabilistic_reg_loss

        reg_loss = reg_loss * self.reg_coe
        total_loss = reg_loss + conf_loss

        self.current_step += 1

        with torch.no_grad():
            var = torch.exp(rlogvar)

            mean_var = var.mean()
            max_var = var.max()
            min_var = var.min()
            std_var = var.std()

        self.loss_dict.update({
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'conf_loss': conf_loss,
            'prob_reg_loss': probabilistic_reg_loss.detach(),
            'standard_reg_loss': standard_reg_loss.detach(),
            'prob_weight': torch.tensor(prob_weight, device=total_loss.device),
            'mean_log_var': rlogvar.detach().mean(),
            'mean_var': mean_var.detach(),
            'max_var': max_var.detach(),
            'min_var': min_var.detach(),
            'std_var': std_var.detach()
        })

        return total_loss

    def nll_regression_loss(self, mean, target, log_var, weights):
        target = torch.where(torch.isnan(target), mean, target)
        diff = mean - target

        loss = 0.5 * torch.exp(-log_var) * (diff ** 2) + 0.5 * log_var
        loss = loss * weights.unsqueeze(-1)

        loss = torch.clamp(loss, min=0.0)

        return loss.sum() / mean.shape[0]

    def energy_score_regression_loss(self, mean, target, log_var, weights):
        """
        Diagonal Gaussian Energy Score loss.

        This samples bbox predictions from:
            Normal(mean, std)

        where:
            std = exp(0.5 * log_var)
        """
        target = torch.where(torch.isnan(target), mean, target)

        std = torch.exp(0.5 * log_var)
        normal_dist = torch.distributions.Normal(mean, std)

        # samples: (num_samples + 1, B, anchors, 7)
        samples = normal_dist.rsample((self.num_samples + 1,))

        samples_1 = samples[:self.num_samples]
        samples_2 = samples[1:self.num_samples + 1]

        target_samples = target.unsqueeze(0).repeat(
            self.num_samples,
            1,
            1,
            1
        )

        sample_weights = weights.unsqueeze(0).repeat(
            self.num_samples,
            1,
            1
        )

        first_term_src = self.reg_loss_func(
            samples_1,
            target_samples,
            weights=sample_weights
        )
        first_term = 2.0 * first_term_src.sum() / self.num_samples

        second_term_src = self.reg_loss_func(
            samples_1,
            samples_2,
            weights=sample_weights
        )
        second_term = -second_term_src.sum() / self.num_samples

        loss = (first_term + second_term) / mean.shape[0]
        return loss

    def get_probabilistic_loss_weight(self):
        """
        Linear warmup from 0 to 1.
        """
        if self.annealing_steps <= 0:
            return 1.0
        return min(1.0, float(self.current_step) / float(self.annealing_steps))

    def cls_loss_func(self,
                      input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor,
                                          target: torch.Tensor):
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        """
        Same direction/yaw treatment as original PointPillarLoss.
        """
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([
            boxes1[..., :dim],
            rad_pred_encoding,
            boxes1[..., dim + 1:]
        ], dim=-1)

        boxes2 = torch.cat([
            boxes2[..., :dim],
            rad_tg_encoding,
            boxes2[..., dim + 1:]
        ], dim=-1)

        return boxes1, boxes2

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        prob_reg_loss = self.loss_dict['prob_reg_loss']
        standard_reg_loss = self.loss_dict['standard_reg_loss']
        prob_weight = self.loss_dict['prob_weight']
        mean_log_var = self.loss_dict['mean_log_var']
        mean_var = self.loss_dict['mean_var']
        max_var = self.loss_dict['max_var']
        min_var = self.loss_dict['min_var']
        std_var = self.loss_dict['std_var']

        msg = (
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f || Prob Loc: %.4f || Std Loc: %.4f"
                " || Max Var: %.4f || Min Var: %.4f || Std Var: %.4f"
                " || Prob Weight: %.4f || Mean LogVar: %.4f || Mean Var: %.4f"
                %
                (
                    epoch,
                    batch_id + 1,
                    batch_len,
                    total_loss.item(),
                    conf_loss.item(),
                    reg_loss.item(),
                    prob_reg_loss.item(),
                    standard_reg_loss.item(),
                    max_var.item(),
                    min_var.item(),
                    std_var.item(),
                    prob_weight.item(),
                    mean_log_var.item(),
                    mean_var.item()
                )
        )

        if pbar is None:
            print(msg)
        else:
            pbar.set_description(msg)

        global_step = epoch * batch_len + batch_id
        writer.add_scalar('Regression_loss', reg_loss.item(), global_step)
        writer.add_scalar('Confidence_loss', conf_loss.item(), global_step)
        writer.add_scalar('Probabilistic_regression_loss',
                          prob_reg_loss.item(), global_step)
        writer.add_scalar('Standard_regression_loss',
                          standard_reg_loss.item(), global_step)
        writer.add_scalar('Probabilistic_loss_weight',
                          prob_weight.item(), global_step)
        writer.add_scalar('Mean_log_variance',
                          mean_log_var.item(), global_step)
        writer.add_scalar('Mean_variance',
                          mean_var.item(), global_step)
        writer.add_scalar("Max_variance",
                          max_var.item(),
                          global_step)

        writer.add_scalar("Min_variance",
                          min_var.item(),
                          global_step)

        writer.add_scalar("Std_variance",
                          std_var.item(),
                          global_step)

    def logging_da(self, epoch, batch_id, batch_len, writer, da_loss, pbar=None):
        """
        Kept for compatibility with DA training.
        """
        self.logging(epoch, batch_id, batch_len, writer, pbar=pbar)

        global_step = epoch * batch_len + batch_id
        writer.add_scalar('da_fea_loss', da_loss['fea_loss'].item(), global_step)
        writer.add_scalar('da_ins_loss', da_loss['ins_loss'].item(), global_step)

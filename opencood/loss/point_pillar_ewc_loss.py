"""Point Pillar EWC Loss"""
import torch.nn as nn
import torch
from opencood.loss.point_pillar_loss import PointPillarLoss

class EWCLoss(nn.Module):
    def __init__(self, model: nn.Module, importance=100000):
        super().__init__()
        self.model = model
        self.importance = importance
        self.params_old = {}
        self.fisher_old = {}

    def forward(self, model: nn.Module):
        loss = torch.tensor(0.0).cuda()
        for name, param in model.named_parameters():
            if name in self.params_old:
                loss += torch.sum(self.fisher_old[name] * (param - self.params_old[name]).pow(2))

        return self.importance * loss

class PointPillarEWCLoss(PointPillarLoss):
    def logging_ewc(self, epoch, batch_id, batch_len, writer, ewc_loss, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                  " || Loc Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), conf_loss.item(), reg_loss.item()))
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f || EWC Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), conf_loss.item(), reg_loss.item(), ewc_loss.item()))

        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch * batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch * batch_len + batch_id)

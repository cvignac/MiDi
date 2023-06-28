import torch
import torch.nn as nn
import time
import wandb

from torchmetrics import MeanSquaredError
from midi.metrics.abstract_metrics import CrossEntropyMetric

class TrainLoss(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.train_pos_mse = MeanSquaredError(sync_on_compute=False, dist_sync_on_step=False)
        self.node_loss = CrossEntropyMetric()
        self.charges_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()

        self.lambda_train = lambda_train

    def forward(self, masked_pred, masked_true, log: bool):
        """ Compute train metrics. Warning: the predictions and the true values are masked, but the relevant entriese
            need to be computed before calculating the loss

            masked_pred, masked_true: placeholders
            log : boolean. """

        node_mask = masked_true.node_mask
        bs, n = node_mask.shape

        true_pos = masked_true.pos[node_mask]       # q x 3
        masked_pred_pos = masked_pred.pos[node_mask]        # q x 3

        true_X = masked_true.X[node_mask]       # q x 4
        masked_pred_X = masked_pred.X[node_mask]        # q x 4

        true_charges = masked_true.charges[node_mask]       # q x 3
        masked_pred_charges = masked_pred.charges[node_mask]        # q x 3

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        masked_pred_E = masked_pred.E[edge_mask]        # r x 5
        true_E = masked_true.E[edge_mask]       # r x 5

        # Check that the masking is correct
        assert (true_X != 0.).any(dim=-1).all()
        assert (true_charges != 0.).any(dim=-1).all()
        assert (true_E != 0.).any(dim=-1).all()

        loss_pos = self.train_pos_mse(masked_pred_pos, true_pos) if true_X.numel() > 0 else 0.0
        loss_X = self.node_loss(masked_pred_X, true_X) if true_X.numel() > 0 else 0.0
        loss_charges = self.charges_loss(masked_pred_charges, true_charges) if true_charges.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(masked_pred.y, masked_true.y) if masked_true.y.numel() > 0 else 0.0

        batch_loss = (self.lambda_train[0] * loss_pos + self.lambda_train[1] * loss_X +
                      self.lambda_train[2] * loss_charges + self.lambda_train[3] * loss_E +
                      self.lambda_train[4] * loss_y)

        to_log = {"train_loss/pos_mse": self.lambda_train[0] * self.train_pos_mse.compute() if true_X.numel() > 0 else -1,
                  "train_loss/X_CE": self.lambda_train[1] * self.node_loss.compute() if true_X.numel() > 0 else -1,
                  "train_loss/charges_CE": self.lambda_train[2] * self.charges_loss.compute() if true_charges.numel() > 0 else -1,
                  "train_loss/E_CE": self.lambda_train[3] * self.edge_loss.compute() if true_E.numel() > 0 else -1.0,
                  "train_loss/y_CE": self.lambda_train[4] * self.y_loss.compute() if masked_true.y.numel() > 0 else -1.0,
                  "train_loss/batch_loss": batch_loss.item()} if log else None

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.train_pos_mse, self.node_loss, self.charges_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_pos_loss = self.train_pos_mse.compute().item() if self.train_pos_mse.total > 0 else -1.0
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_charges_loss = self.charges_loss.compute().item() if self.charges_loss > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0
        epoch_y_loss = self.train_y_loss.compute().item() if self.y_loss.total_samples > 0 else -1.0

        to_log = {"train_epoch/pos_mse": epoch_pos_loss,
                  "train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/charges_CE": epoch_charges_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

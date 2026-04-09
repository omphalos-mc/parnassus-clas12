from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MSELoss(nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        loss_weights: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)
        self.loss_weights = nn.Buffer(
            loss_weights if loss_weights is not None else torch.ones(1),
            persistent=False,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ):
        if mask is not None:
            pred = pred * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)
        mse_loss = (
            self.loss_weights * F.mse_loss(pred, target, reduction="none")
        ).sum()
        if self.reduction == "sum":
            return mse_loss
        elif self.reduction == "mean":
            denominator = (
                np.prod(pred.shape) if mask is None else mask.sum() * pred.shape[-1]
            )
            return mse_loss / denominator


class MSEAndDirectionLoss(nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(
        self,
        cosine_sim_dim: int = 2,
        reduction: Literal["mean", "sum"] = "mean",
        loss_weights: torch.Tensor | list[float] | None = None,
    ):
        super().__init__()
        assert cosine_sim_dim > 0, "cannot be batch dimension"
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.cosine_sim_dim = cosine_sim_dim
        self.reduction = reduction
        if isinstance(loss_weights, list):
            loss_weights = torch.tensor(loss_weights)
        self.loss_weights = nn.Buffer(
            loss_weights if loss_weights is not None else torch.ones(1),
            persistent=False,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ):
        # Adjust cosine similarity dimension for 2D tensors
        cosine_dim = (
            1 if pred.dim() == 2 and self.cosine_sim_dim == 2 else self.cosine_sim_dim
        )

        # Apply mask if provided
        if mask is not None:
            pred_masked = pred * mask.unsqueeze(-1)
            target_masked = target * mask.unsqueeze(-1)
        else:
            pred_masked = pred
            target_masked = target

        # Compute losses
        mse_loss = (
            self.loss_weights * F.mse_loss(pred_masked, target_masked, reduction="none")
        ).sum()
        direction_loss = 1.0 - F.cosine_similarity(
            pred_masked, target_masked, dim=cosine_dim
        )

        if mask is not None:
            direction_loss = direction_loss * mask
        direction_loss = direction_loss.sum()

        # Apply reduction
        if self.reduction == "sum":
            return mse_loss + direction_loss

        # Mean reduction
        mse_denominator = (
            np.prod(pred.shape) if mask is None else mask.sum() * pred.shape[-1]
        )
        cosine_denominator = np.prod(pred.shape[:-1]) if mask is None else mask.sum()

        return mse_loss / mse_denominator + direction_loss / cosine_denominator


class Set2SetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_loss = nn.MSELoss(reduction="none")

    def forward(self, input, target, mask):
        ### new code !!!
        bs = len(input)
        new_mask = mask.unsqueeze(1).expand(-1, target.size(1), -1).cpu()

        new_input = input.cpu().unsqueeze(1).expand(-1, target.size(1), -1, -1)
        new_target = target.cpu().unsqueeze(2).expand(-1, -1, input.size(1), -1)

        pdist = self.regression_loss(new_input, new_target)

        pdist_pt = pdist[..., 0]
        pdist_eta = pdist[..., 1]
        pdist_phi = pdist[..., 2]

        pdist = pdist.mean(-1)
        pdist = pdist * new_mask
        pdist_pt = pdist_pt * new_mask
        pdist_eta = pdist_eta * new_mask
        pdist_phi = pdist_phi * new_mask

        pdist = torch.nan_to_num(pdist, nan=0.0, posinf=0.0, neginf=0.0)

        pdist_ = pdist.detach().numpy()
        indices = np.array(
            [linear_sum_assignment(p) for p in pdist_]
        )  # indices shape (b,2,N)

        spicy_mat = torch.zeros((bs), device=pdist.device)
        spicy_mat_pt = torch.zeros((bs), device=pdist.device)
        spicy_mat_eta = torch.zeros((bs), device=pdist.device)
        spicy_mat_phi = torch.zeros((bs), device=pdist.device)
        for idx_i in range(bs):
            indices_i = indices.shape[2] * indices[idx_i, 0] + indices[idx_i, 1]

            losses = torch.gather(
                pdist[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss = losses.mean(0)

            spicy_mat[idx_i] = total_loss

            losses_pt = torch.gather(
                pdist_pt[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_pt = losses_pt.mean(0)
            spicy_mat_pt[idx_i] = total_loss_pt

            losses_eta = torch.gather(
                pdist_eta[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_eta = losses_eta.mean(0)
            spicy_mat_eta[idx_i] = total_loss_eta

            losses_phi = torch.gather(
                pdist_phi[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_phi = losses_phi.mean(0)
            spicy_mat_phi[idx_i] = total_loss_phi

        return {
            "total_loss": spicy_mat.mean().to(input.device),
            "pt_loss": spicy_mat_pt.mean(),
            "eta_loss": spicy_mat_eta.mean(),
            "phi_loss": spicy_mat_phi.mean(),
        }

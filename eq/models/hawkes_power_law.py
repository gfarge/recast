import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

import eq
from eq.data.batch import get_mask, pad_sequence

from .tpp_model import TPPModel

# To do:
# 1- Implement sampling
# 2- Implement exponential, n-exp


def branching_ratio(k, c, p):
    """
    Compute branching ratio of a power-law Hawkes process.

    Parameters
    ----------
    k : float
        Productivity parameter.
    c : float
        Intercept parameter.
    p : float
        Time decay parameter.

    Returns
    -------
    branching_ratio : float
        Expected number of offspring of an event.

    """
    branching_ratio = k * c**(1-p) / (p-1)

    if branching_ratio > 1:
        print("Branching ratio: ", branching_ratio)
    return branching_ratio


def kernel_int(T1, T2, k, c, p):  # /!\ no productivity
    # ?? Add productivity?
    """Integral of kernel from T1 to T2.

    Used for the finite catalog correction in the productivity estimate (Brodsky 2011).
    """
    if p == 1:
        return np.log(T2 + c) - np.log(T1 + c)
    else:
        return ((T2 + c) ** (1 - p) - (T1 + c) ** (1 - p)) / (1 - p)


def kernel_sample(T1, T2, c, p, size=1, t_max=1e10):
    """Draw sample between T1 and T2 from kernel using inverse transform."""
    # ?? Add productivity? It plays a role only in sampling

    # Draw sampling variable from U[0, 1]
    u = np.random.random(size=size)
    # Define cdf(t), F(t) --- scaled to F(t_max) to not blow up
    F = lambda tau: kernel_int(0, tau, c, p) / kernel_int(0, t_max, c, p)
    # Map u from [0, 1] to the interval [cdf(T1), cdf(T2)]
    u_prime = u * (F(T2) - F(T1)) + F(T1)

    sample = c - (u_prime*kernel_int(0, t_max, c, p)*(1-p) + c**(1-p)) ** (1/(1-p))
#    sample = c - (u_prime*kernel_int(0, t_max, c, p)*(1-p)/k + c**(1-p)) ** (1/(1-p))

    return sample

class Hawkes(TPPModel):
    """Hawkes process, power law: lamda(t | H(t)) = k / (c + t)**p.

    Args:
        p_init: Initial value of p, time decay parameter.
        c_init: Initial value of c, time offset parameter.
        k_init: Initial value of k, productivity parameter.
        mu_init: Initial value of the background (immigrant) intensity.
        report_params: Whether to report the model parameters in the PyTorch Lightning
            progress bar during training.
        learning_rate: Learning rate use for optimization.
    """

    def __init__(
        self,
        p_init: float = 1.1,
        c_init: float = 0.1,
        k_init: float = 0.01,
        mu_init: float = 0.1,
        report_params: bool = True,
        learning_rate: float = 5e-2,
    ):
        super().__init__()
        self.log_p = nn.Parameter(torch.tensor(math.log(p_init)))
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))
        self.log_k = nn.Parameter(torch.tensor(math.log(k_init)))
        self.log_mu = nn.Parameter(torch.tensor(math.log(mu_init)))
        self.report_params = report_params
        self.learning_rate = learning_rate

    @property
    def p(self):
        return torch.exp(self.log_p)

    @property
    def c(self):
        return torch.exp(self.log_c)

    @property
    def mu(self):
        return torch.exp(self.log_mu)

    @property
    def k(self):
        return torch.exp(self.log_k)

    def kernel(self, time):
        """Computes the kernel for the input time"""
        return self.k / (self.c + time) ** (self.p)

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        # --> Extract arrival times from batch
        t = batch.arrival_times
        # t_select - arrival times of events for which intensity must be computed, shape (B, S)
        # (where S = L if t_start == t_nll_start, and S <= L otherwise)
        t_select, intensity_mask = masked_select_per_row(t, batch.mask)

        # --> Compute summed log intensity: sum_[1, N] log λ(ti)
        # inter-times as a tensor: delta_t[0, i, j] = t_i - t_j for seq 0
        delta_t = t_select.unsqueeze(-1) - t.unsqueeze(-2)  # (B, S, L)
        prev_mask = (delta_t > 0).float()  # (B, S, L), to select positive inter-times
                                           # effect of ev i on ev j

        # intensity[0, i, j] = contribution of event t_j on intensity at time t_i
        intensity = (delta_t * prev_mask + self.c).pow(-self.p)  # (B, S, L)
        log_intensity = (
            torch.log(
                (intensity * self.k * prev_mask).sum(-1) + self.mu
            )
            * intensity_mask
        ).sum(-1)

        # --> Compute int_[0, T] λ(u)du (integral from t_i to t_end of the
        # kernel)
        one_minus_p = 1 - self.p
        t_end = batch.t_end.unsqueeze(-1)  # (B, 1)
        t_nll_start = batch.t_nll_start.unsqueeze(-1)  # (B, 1)
        # kernel_int[0, j] = integral of the omori law from max(t_j, t_nll_start) to t_end
        kernel_int = (
            (t_end - t + self.c).pow(one_minus_p)
            - ((t_nll_start - t).clamp_min(0.0) + self.c).pow(one_minus_p)
        ) / one_minus_p  # (B, L)

        survival_mask = get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx)

        integral = (kernel_int * self.k * survival_mask).sum(-1)
        integral += (batch.t_end - batch.t_nll_start) * self.mu

        return (-log_intensity + integral) / (batch.t_end - batch.t_nll_start)  # (B,)

    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        if self.report_params:
            for param_name in ["p", "c", "mu", "k"]:
                self.log(
                    f"params/{param_name}",
                    getattr(self, param_name).item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
        self.logged_train_loss.append(loss.item())
        return loss

def masked_select_per_row(matrix, mask):
    """Perform masked select on each row, and return the result as a padded tensor.

    Args:
        matrix: 2-d tensor from which values must be selected, shape [M, N]
        mask: Boolean matrix indicating what entries must be selected, shape [M, N]

    Returns:
        new_matrix: 2-d tensor, where each row contains the selected entries from the
            respective row of matrix + padding.
        new_mask: Float mask indicating what entries correspond to actual values
            (new_mask[i, j] = 1 => new_matrix[i, j] is not padding).

    Example:
        >>> matrix = torch.tensor([
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
            ])
        >>> mask = torch.tensor([
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
            ])
        >>> selected, new_mask = masked_select_per_row(matrix, mask)
        >>> print(selected)
        tensor([[1, 2, 3],
                [8, 9, 0]])
        >>> print(new_mask)
        tensor([[1., 1., 1.],
                [1., 1., 0.]])
    """
    assert matrix.shape == mask.shape and matrix.ndim == 2
    selected_rows = []
    for matrix_row, mask_row in zip(matrix, mask.bool()):
        selected_rows.append(matrix_row.masked_select(mask_row))

    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(s) for s in selected_rows])
    return new_matrix, new_mask.float()

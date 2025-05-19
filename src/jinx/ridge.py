""" Ridge regression with variational inference. """

from typing import Self
import torch
from torch import nn, Tensor
from jinx import Model

class RidgeSetup(nn.Module):
    """ Ridge regression with variational inference. """

    def __init__(
        self: Self, 
        in_features: int, 
        out_features: int = 1, 
        prior_var: float = 1.0,
        penalty: float = 1.0, 
        lr: float = 1e-2
    ) -> None:
        super().__init__()

        # variational parameters for beta
        self.b_mu = nn.Parameter(torch.randn(in_features, out_features))
        self.b_logvar = nn.Parameter(torch.randn(in_features, out_features))

        # variational parameters for e
        self.e_mu = nn.Parameter(torch.randn(out_features))
        self.e_logvar = nn.Parameter(torch.randn(out_features))

        # noise variance
        self.log_noise_var = nn.Parameter(torch.tensor(0.0))

        # prior variance
        self.prior_var = prior_var
        
        self.penalty = penalty

        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self: Self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ Forward pass of the model. """
        pass

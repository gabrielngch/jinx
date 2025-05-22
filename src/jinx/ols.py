""" OLS with variational inference. """

from typing import Self
import torch
from torch import nn, Tensor
from jinx import Model

class OLSSetup(nn.Module):
    """ Setup of the OLS problem y = Xb + e using pytorch parameters. """

    def __init__(self: Self, in_features: int, out_features: int = 1, prior_var: float = 1.0, lr: float = 1e-2) -> None:
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

        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self: Self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """ 
        Forward pass of model. returns y, b, e where y = Xb + e. 
        
        Args:
            X: The input data tensor.

        Returns:
            y: The output tensor.
            b: The posterior mean of beta.
            e: The posterior mean of e.
        """

        eps_b = torch.rand_like(self.b_mu)
        b = self.b_mu + torch.exp(self.b_logvar) * eps_b

        eps_e = torch.rand_like(self.e_mu)
        e = self.e_mu + torch.exp(self.e_logvar) * eps_e

        y = X.matmul(b) + e

        return y, b, e

    def elbo(self: Self, X: Tensor, y: Tensor, n_samples: int = 1) -> float:
        """
        Compute the ELBO for the OLS problem.

        Args:
            X: The input data tensor.
            y: The output data tensor.
            n_samples: The number of samples to use for the ELBO.

        Returns:
            elbo: The ELBO value.
        """
        N = X.shape[0]
        recon = 0.0
        kl = 0.0
        noise = torch.exp(self.log_noise_var)

        for _ in range(n_samples):
            y_pred, _, _ = self.forward(X)
            recon += -0.5 * N * torch.log(2 * torch.pi * noise) - 0.5 * torch.sum((y - y_pred) ** 2) / noise

            kl_b = 0.5 * torch.sum(
                (torch.exp(self.b_logvar) + self.b_mu.pow(2.0)) / self.prior_var
                - 1.0 * self.b_logvar + torch.log(torch.tensor(self.prior_var))
            )
            kl_e = 0.5 * torch.sum(  
                (torch.exp(self.e_logvar) + self.e_mu.pow(2.0)) / self.prior_var
                - 1.0 - self.e_logvar + torch.log(torch.tensor(self.prior_var))
            )
            kl += kl_b + kl_e

        recon /= n_samples
        kl /= n_samples

        return recon - kl

    def step(self: Self, X: Tensor, y: Tensor, n_samples: int = 1) -> None:
        """
        Perform a single step of the optimisation.

        Args:
            X: The input data tensor.
            y: The output data tensor.
        """
        self.optimiser.zero_grad()
        elbo_val = self.elbo(X, y, n_samples)

        loss = -elbo_val
        loss.backward()
        self.optimiser.step()

        return elbo_val.item()

class LinearRegression(Model):
    """ Linear regression with variational inference. """
    def __init__(
            self: Self, 
            in_features: int, 
            out_features: int = 1, 
            prior_var: float = 1.0, 
            mc_samples: int = 1, 
            lr: float = 1e-2
            ) -> None:
        self.model = OLSSetup(in_features, out_features, prior_var, lr)
        self.mc_samples = mc_samples

    def update(self: Self, X: Tensor, y: Tensor) -> float:
        """
        Perform a single step of the optimisation and return the predicted y.

        Args:
            X: The input data tensor.
            y: The output data tensor.
        """
        self.model.step(X, y, self.mc_samples)
        y_pred, _, _ = self.model.forward(X)
        return y_pred

    @property
    def weight_posterior(self: Self) -> dict[str, Tensor]:
        return {
            "b_mu": self.model.b_mu.detach().clone(),
            "b_logvar": self.model.b_logvar.detach().clone(),
            "e_mu": self.model.e_mu.detach().clone(),
            "e_logvar": self.model.e_logvar.detach().clone(),
            "noise_var": torch.exp(self.model.log_noise_var).detach().clone(),
        }
        


if __name__ == "__main__":
    # synthetic data
    torch.manual_seed(42)
    N, D = 10000, 3
    X = torch.randn(N, D)
    true_b = torch.tensor([[2.0], [-1.0], [0.5]])
    true_a = torch.tensor(0.1)
    y = X.matmul(true_b) + true_a * torch.randn(N, 1)

    model = LinearRegression(D)
    for i in range(10000):
        X_single = X[i, :]
        y_single = y[i, :]
        model.update(X_single, y_single)

    print(model.weight_posterior)


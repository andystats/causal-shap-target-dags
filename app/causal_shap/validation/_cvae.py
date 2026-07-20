"""PyTorch Conditional VAE for learning p(X | Z).

Adapted from the author's Instats workshop (Module 5) R-torch/PyTorch model.
This module imports torch at module load and is therefore imported lazily by
``generators.CVAEGenerator`` — it never enters the deployed app's import path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        h = F.relu(self.fc1(torch.cat([x, z], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(torch.cat([u, z], dim=1)))
        return self.fc2(h)


class ConditionalVAE(nn.Module):
    """Conditional VAE learning p(X | Z) with a standard-scaled covariate space."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = 1
        self.encoder = _Encoder(input_dim, self.condition_dim, latent_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, self.condition_dim, input_dim, hidden_dim)
        self.scaler = StandardScaler()

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        mu, logvar = self.encoder(x, z)
        u = self._reparameterize(mu, logvar)
        return self.decoder(u, z), mu, logvar

    @staticmethod
    def _elbo(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon = F.mse_loss(x_recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld

    def fit(self, X: np.ndarray, Z: np.ndarray, *, epochs: int, lr: float, seed: int) -> "ConditionalVAE":
        """Train with forced determinism so committed weights reproduce exactly."""
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        x_tensor = torch.as_tensor(self.scaler.fit_transform(X), dtype=torch.float32)
        z_tensor = torch.as_tensor(np.asarray(Z, dtype=float), dtype=torch.float32).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            x_recon, mu, logvar = self(x_tensor, z_tensor)
            loss = self._elbo(x_recon, x_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
        self.eval()
        return self

    def sample(self, z_values: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        self.eval()
        n_samples = len(z_values)
        with torch.no_grad():
            u = torch.randn(n_samples, self.latent_dim)
            z_tensor = torch.as_tensor(np.asarray(z_values, dtype=float), dtype=torch.float32).reshape(-1, 1)
            x_scaled = self.decoder(u, z_tensor).numpy()
        return self.scaler.inverse_transform(x_scaled)

    def save_decoder(self, path) -> None:
        """Persist just the decoder weights, which the app can reload without torch training."""
        torch.save(self.decoder.state_dict(), path)

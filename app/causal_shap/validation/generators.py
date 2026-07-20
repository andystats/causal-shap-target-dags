"""Covariate generators for the validation layer.

Three ways to synthesize realistic covariates conditional on treatment. The two
moment/resample generators are pure NumPy and run live in the deployed app; the
CVAE generator (lazy torch) is used only by local build scripts. All redeem the
Module 5 slides' promise of alternative X|Z generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..seeds import SEED_CVAE_TRAINING


@runtime_checkable
class CovariateGenerator(Protocol):
    columns: tuple[str, ...]

    def fit(self, X: pd.DataFrame, z: np.ndarray) -> "CovariateGenerator": ...

    def sample(self, z_values: np.ndarray, seed: int) -> pd.DataFrame: ...


def _split_by_treatment(X: pd.DataFrame, z: np.ndarray) -> dict[int, pd.DataFrame]:
    z = np.asarray(z)
    return {int(level): X.loc[z == level] for level in np.unique(z)}


@dataclass
class MVNGenerator:
    """Moment-matched multivariate normal per treatment arm (the live path)."""

    columns: tuple[str, ...] = ()
    _means: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _chols: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def fit(self, X: pd.DataFrame, z: np.ndarray) -> "MVNGenerator":
        self.columns = tuple(X.columns)
        for level, group in _split_by_treatment(X, z).items():
            values = group.to_numpy(dtype=float)
            cov = np.cov(values, rowvar=False)
            cov = np.atleast_2d(cov) + 1e-6 * np.eye(values.shape[1])
            self._means[level] = values.mean(axis=0)
            self._chols[level] = np.linalg.cholesky(cov)
        return self

    def sample(self, z_values: np.ndarray, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        z_values = np.asarray(z_values)
        out = np.empty((len(z_values), len(self.columns)), dtype=float)
        for level in np.unique(z_values):
            mask = z_values == level
            chol = self._chols[int(level)]
            noise = rng.standard_normal((int(mask.sum()), len(self.columns)))
            out[mask] = self._means[int(level)] + noise @ chol.T
        return pd.DataFrame(out, columns=self.columns)


@dataclass
class BootstrapGenerator:
    """Resample real rows within a treatment arm, with light Gaussian jitter."""

    jitter: float = 0.05
    columns: tuple[str, ...] = ()
    _pools: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _scales: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def fit(self, X: pd.DataFrame, z: np.ndarray) -> "BootstrapGenerator":
        self.columns = tuple(X.columns)
        for level, group in _split_by_treatment(X, z).items():
            values = group.to_numpy(dtype=float)
            self._pools[int(level)] = values
            self._scales[int(level)] = values.std(axis=0)
        return self

    def sample(self, z_values: np.ndarray, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        z_values = np.asarray(z_values)
        out = np.empty((len(z_values), len(self.columns)), dtype=float)
        for level in np.unique(z_values):
            mask = z_values == level
            pool = self._pools[int(level)]
            picks = rng.integers(0, len(pool), size=int(mask.sum()))
            jitter = rng.standard_normal((int(mask.sum()), len(self.columns)))
            out[mask] = pool[picks] + self.jitter * self._scales[int(level)] * jitter
        return pd.DataFrame(out, columns=self.columns)


@dataclass
class CVAEGenerator:
    """Conditional VAE generator (lazy torch; local build scripts only)."""

    latent_dim: int = 16
    epochs: int = 200
    lr: float = 1e-3
    columns: tuple[str, ...] = ()
    _model: object = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, z: np.ndarray) -> "CVAEGenerator":
        from ._cvae import ConditionalVAE

        self.columns = tuple(X.columns)
        model = ConditionalVAE(input_dim=X.shape[1], latent_dim=self.latent_dim)
        model.fit(X.to_numpy(dtype=float), np.asarray(z, dtype=float), epochs=self.epochs, lr=self.lr, seed=SEED_CVAE_TRAINING)
        self._model = model
        return self

    def sample(self, z_values: np.ndarray, seed: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("CVAEGenerator.sample called before fit")
        samples = self._model.sample(np.asarray(z_values, dtype=float), seed=seed)
        return pd.DataFrame(samples, columns=self.columns)

    def save_decoder(self, path: Path) -> None:
        """Persist the trained decoder weights (for provenance / the committed bundle)."""
        if self._model is None:
            raise RuntimeError("CVAEGenerator.save_decoder called before fit")
        self._model.save_decoder(path)

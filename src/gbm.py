"""
Geometric Brownian Motion (GBM) utilities
=========================================

SDE (Itô):
    dS_t = mu * S_t * dt + sigma * S_t * dW_t,  sigma > 0

This module provides exact risk–neutral terminal draws and path simulators
(Euler–Maruyama and exact-discretization in log space), with optional
antithetic variates.

Design goals:
- Pure functions with type hints
- Numpy vectorization (no Python loops over paths)
- Numerically stable (simulate in log-space)
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np

Array = np.ndarray

__all__ = [
    "simulate_gbm_paths",
    "simulate_risk_neutral_paths",
    "euler_paths",
    "sample_terminal",
    "sample_terminal_risk_neutral",
    "gbm_terminal_moments",
]


# ------------------------------ Core helpers ------------------------------ #

def _rng_from(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return a NumPy Generator; create one if *rng* is None."""
    return np.random.default_rng() if rng is None else rng


def _antithetic_expand(Z: Array, antithetic: bool) -> Array:
    """Expand normal draws to include their antithetics if requested.

    Parameters
    ----------
    Z : (n, ...) array
        Base standard-normal draws.
    antithetic : bool
        If True, returns `np.concatenate([Z, -Z], axis=0)`.
    """
    if not antithetic:
        return Z
    return np.concatenate([Z, -Z], axis=0)


# -------------------------- Exact terminal sampling ----------------------- #

def sample_terminal(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    *,
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = False,
    dtype: np.dtype | type = np.float64,
) -> Array:
    """Draw exact terminal prices for GBM.

    Uses the closed form
        S_T = S0 * exp((mu - 0.5*sigma^2) T + sigma * sqrt(T) * Z),  Z~N(0,1).

    Returns
    -------
    ST : ndarray, shape (n_paths,)
    """
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = _rng_from(rng)
    n_half = (n_paths + 1) // 2 if antithetic else n_paths
    Z = rng.standard_normal(n_half, dtype=dtype)
    Z = _antithetic_expand(Z, antithetic)
    Z = Z[:n_paths]  # trim if n_paths odd

    if T == 0.0 or sigma == 0.0:
        return np.full_like(Z, fill_value=S0 * math.exp(mu * T), dtype=dtype)

    drift = (mu - 0.5 * sigma * sigma) * T
    shock = sigma * math.sqrt(T) * Z
    return (S0 * np.exp(drift + shock)).astype(dtype, copy=False)


def sample_terminal_risk_neutral(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    **kwargs,
) -> Array:
    """Convenience: terminal draws under Q with drift *r*."""
    return sample_terminal(S0=S0, mu=r, sigma=sigma, T=T, n_paths=n_paths, **kwargs)


# ------------------------------ Path simulators --------------------------- #

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    *,
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = False,
    dtype: np.dtype | type = np.float64,
) -> Tuple[Array, Array]:
    """Simulate GBM paths with *exact* log-space discretization.

    The scheme applies the identity
        log S_{t+dt} = log S_t + (mu - 0.5*sigma^2) dt + sigma * sqrt(dt) * Z
    with i.i.d. Z ~ N(0,1).

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
        Time grid from 0 to T.
    paths : ndarray, shape (n_paths, n_steps+1)
        Simulated price paths including S0 at column 0.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T < 0:
        raise ValueError("T must be non-negative")

    t = np.linspace(0.0, float(T), int(n_steps) + 1, dtype=dtype)
    dt = t[1] - t[0]

    rng = _rng_from(rng)
    n_half = (n_paths + 1) // 2 if antithetic else n_paths
    Z = rng.standard_normal(size=(n_half, n_steps), dtype=dtype)
    Z = _antithetic_expand(Z, antithetic)[:n_paths]

    if dt == 0.0 or sigma == 0.0:
        # deterministic growth
        growth = math.exp(mu * T)
        paths = np.full((n_paths, n_steps + 1), S0, dtype=dtype)
        paths[:, -1] = S0 * growth
        # Fill intermediate equally-spaced deterministic values for completeness
        if n_steps > 1:
            for k in range(1, n_steps + 1):
                paths[:, k] = S0 * math.exp(mu * (k * dt))
        return t, paths

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    # cumulative log-returns per path
    logS0 = math.log(S0)
    log_increments = drift + vol * Z  # (n_paths, n_steps)
    logS = logS0 + np.cumsum(log_increments, axis=1)

    # prepend the initial state
    paths = np.empty((n_paths, n_steps + 1), dtype=dtype)
    paths[:, 0] = S0
    paths[:, 1:] = np.exp(logS, dtype=dtype)
    return t, paths


def simulate_risk_neutral_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    **kwargs,
) -> Tuple[Array, Array]:
    """Convenience wrapper with drift=r (risk-neutral measure)."""
    return simulate_gbm_paths(S0=S0, mu=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, **kwargs)


def euler_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    *,
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = False,
    dtype: np.dtype | type = np.float64,
) -> Tuple[Array, Array]:
    """Euler–Maruyama paths for GBM (for didactic comparisons).

    Update rule:
        S_{t+dt} = S_t + mu*S_t*dt + sigma*S_t*sqrt(dt)*Z
    This scheme is biasy for GBM but useful for illustrating strong/weak convergence.
    """
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")

    t = np.linspace(0.0, float(T), int(n_steps) + 1, dtype=dtype)
    dt = t[1] - t[0]

    rng = _rng_from(rng)
    n_half = (n_paths + 1) // 2 if antithetic else n_paths
    Z = rng.standard_normal(size=(n_half, n_steps), dtype=dtype)
    Z = _antithetic_expand(Z, antithetic)[:n_paths]

    paths = np.empty((n_paths, n_steps + 1), dtype=dtype)
    paths[:, 0] = S0
    if dt == 0.0:
        paths[:, 1:] = S0
        return t, paths

    sqrt_dt = math.sqrt(dt)
    S = np.full(n_paths, S0, dtype=dtype)
    for k in range(n_steps):
        S = S + mu * S * dt + sigma * S * sqrt_dt * Z[:, k]
        paths[:, k + 1] = S
    return t, paths


# ----------------------------- Distribution facts ------------------------- #

def gbm_terminal_moments(S0: float, mu: float, sigma: float, T: float) -> Tuple[float, float]:
    """Return (mean, variance) of S_T for GBM with drift *mu*.

        E[S_T] = S0 * exp(mu*T)
        Var[S_T] = S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1)
    """
    m = S0 * math.exp(mu * T)
    v = (S0 ** 2) * math.exp(2.0 * mu * T) * (math.exp((sigma ** 2) * T) - 1.0)
    return m, v


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) * math.exp(-r * T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0) * math.exp(-r * T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
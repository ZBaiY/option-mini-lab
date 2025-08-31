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
    "price_barrier_ko_call",
    "barrier_survival_weights_bridge",
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

# ---------------- Barrier option pricing with Brownian-Bridge ---------------- #

def _bridge_hit_prob_segment(
    x0: float,
    x1: float,
    h: float,
    dt: float,
    sigma: float,
    kind: str,
) -> float:
    """
    Conditional crossing probability for a single time step of log-price X_t between t and t+dt,
    given endpoints x0=log S_t and x1=log S_{t+dt}.

    Parameters
    ----------
    x0, x1 : float
        Log-price at the endpoints of the step.
    h : float
        Log-barrier level (ln H for down-and-out; ln U for up-and-out).
    dt : float
        Step size (positive).
    sigma : float
        Volatility (>0).
    kind : {"down", "up"}
        "down"  for down-and-out barrier at level H (knock when X_t ≤ h),
        "up"    for up-and-out   barrier at level U (knock when X_t ≥ h).

    Returns
    -------
    p : float
        Probability that the (conditioned) Brownian bridge crosses the barrier within the step.

    Notes
    -----
    For down barrier:
        if x0 ≤ h or x1 ≤ h → p = 1
        else p = exp( -2 * (x0 - h)*(x1 - h) / (sigma^2 * dt) )
    For up barrier:
        if x0 ≥ h or x1 ≥ h → p = 1
        else p = exp( -2 * (h - x0)*(h - x1) / (sigma^2 * dt) )
    """
    if dt <= 0.0 or sigma <= 0.0:
        return 0.0

    if kind == "down":
        if x0 <= h or x1 <= h:
            return 1.0
        num = -2.0 * (x0 - h) * (x1 - h)
    elif kind == "up":
        if x0 >= h or x1 >= h:
            return 1.0
        num = -2.0 * (h - x0) * (h - x1)
    else:
        raise ValueError("kind must be 'down' or 'up'")

    denom = (sigma * sigma) * dt
    # Clamp for numerical safety
    p = math.exp(min(0.0, num / denom))
    # Guard against denormals outside (0,1]
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p


def barrier_survival_weights_bridge(
    paths: Array,
    *,
    sigma: float,
    T: float,
    barrier: float,
    kind: str = "down",
    dtype: np.dtype | type = np.float64,
) -> Array:
    """
    Compute per-path survival weights via Brownian-bridge crossing probabilities.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps+1)
        Simulated price paths including S0 at column 0 (uniform grid from 0 to T).
    sigma : float
        Volatility used for the simulation (per year, same units as T).
    T : float
        Maturity (years).
    barrier : float
        Barrier level H (down) or U (up), in price units.
    kind : {"down","up"}
        Barrier direction; "down" for down-and-out, "up" for up-and-out.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    w : ndarray, shape (n_paths,)
        Survival probability per path across the whole horizon, i.e. ∏_i (1 - p_i).

    Notes
    -----
    This routine assumes equally spaced steps. It is vectorized across paths and steps.
    """
    if barrier <= 0.0:
        raise ValueError("barrier must be positive (in price units)")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    n_paths, n_cols = paths.shape
    if n_cols < 2:
        return np.ones(n_paths, dtype=dtype)

    # Uniform step size
    n_steps = n_cols - 1
    dt = float(T) / float(n_steps)
    if dt <= 0.0:
        return np.ones(n_paths, dtype=dtype)

    X = np.log(paths, dtype=dtype)  # (n_paths, n_steps+1)
    h = math.log(barrier)

    X0 = X[:, :-1]
    X1 = X[:, 1:]

    if kind == "down":
        ok = (X0 > h) & (X1 > h)
        # default p=1 where either endpoint already knocked
        p = np.ones_like(X0, dtype=dtype)
        # compute only where both endpoints are above
        num = -2.0 * (X0[ok] - h) * (X1[ok] - h)
    elif kind == "up":
        ok = (X0 < h) & (X1 < h)
        p = np.ones_like(X0, dtype=dtype)
        num = -2.0 * (h - X0[ok]) * (h - X1[ok])
    else:
        raise ValueError("kind must be 'down' or 'up'")

    denom = (sigma * sigma) * dt
    # Fill where ok with exp(num/denom) clipped to [0,1]
    pi_ok = np.exp(np.minimum(0.0, num / denom), dtype=dtype)
    # numerical clamps
    pi_ok = np.clip(pi_ok, 0.0, 1.0)
    p[ok] = pi_ok

    # survival per step is (1 - p)
    surv = 1.0 - p
    # product over steps
    w = np.prod(surv, axis=1, dtype=dtype)
    # Guard against tiny negatives due to roundoff
    w = np.clip(w, 0.0, 1.0).astype(dtype, copy=False)
    return w


def price_barrier_ko_call(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    *,
    barrier: float,
    kind: str = "down",
    q: float = 0.0,
    n_steps: int = 64,
    n_paths: int = 100_000,
    method: str = "bridge_weight",
    rng: Optional[np.random.Generator] = None,
    antithetic: bool = True,
    dtype: np.dtype | type = np.float64,
    return_se: bool = True,
) -> Tuple[float, Optional[float]]:
    """
    Monte Carlo price for a knock-out European call using Brownian-bridge correction.

    Parameters
    ----------
    S0, K, r, sigma, T : floats
        Usual Black–Scholes inputs (continuous compounding). No dividends unless *q* is set.
    barrier : float
        Barrier level H (down-and-out) or U (up-and-out).
    kind : {"down","up"}
        Barrier direction; "down" (down-and-out) or "up" (up-and-out).
    q : float, default 0.0
        Continuous dividend yield. The risk-neutral drift used for simulation is r - q.
    n_steps : int, default 64
        Number of time steps for path simulation (Uniform grid).
    n_paths : int, default 100_000
        Number of Monte Carlo paths.
    method : {"bridge_weight","bridge_bernoulli"}
        Estimation approach:
          - "bridge_weight": payoff × survival weight ∏_i (1 - p_i)  (lower variance).
          - "bridge_bernoulli": simulate KO as Bernoulli with prob p_i at each step.
    rng : np.random.Generator, optional
        Random generator.
    antithetic : bool, default True
        Use antithetic variates for variance reduction.
    dtype : numpy dtype, default np.float64
        Numeric dtype.
    return_se : bool, default True
        If True, also return the Monte Carlo standard error.

    Returns
    -------
    price : float
        Discounted Monte Carlo price.
    se : float or None
        Standard error of the estimate if *return_se* is True; otherwise None.
    """
    if T <= 0.0:
        # Immediate maturity: price is intrinsic if not knocked; with instantaneous horizon,
        # hitting probability over (0,T] vanishes in continuous time.
        payoff = max(S0 - K, 0.0)
        price = math.exp(-r * T) * payoff
        return (price, 0.0) if return_se else (price, None)

    # Simulate under Q with drift r - q
    t, paths = simulate_gbm_paths(
        S0=S0,
        mu=r - q,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
        antithetic=antithetic,
        dtype=dtype,
    )

    # Compute Brownian-bridge crossing probabilities → survival weights
    w = barrier_survival_weights_bridge(paths, sigma=sigma, T=T, barrier=barrier, kind=kind, dtype=dtype)

    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0, dtype=dtype)

    if method == "bridge_weight":
        contrib = payoff * w
    elif method == "bridge_bernoulli":
        # Bernoulli knock-out per interval using the same p_i used internally in survival weights.
        # To avoid recomputing p_i, we reconstruct them vectorially (same as in barrier_survival_weights_bridge).
        X = np.log(paths, dtype=dtype)
        h = math.log(barrier)
        n_steps_local = paths.shape[1] - 1
        dt = float(T) / float(n_steps_local)
        X0 = X[:, :-1]
        X1 = X[:, 1:]
        if kind == "down":
            ok = (X0 > h) & (X1 > h)
            p = np.ones_like(X0, dtype=dtype)
            num = -2.0 * (X0[ok] - h) * (X1[ok] - h)
        elif kind == "up":
            ok = (X0 < h) & (X1 < h)
            p = np.ones_like(X0, dtype=dtype)
            num = -2.0 * (h - X0[ok]) * (h - X1[ok])
        else:
            raise ValueError("kind must be 'down' or 'up'")
        denom = (sigma * sigma) * dt
        pi_ok = np.exp(np.minimum(0.0, num / denom), dtype=dtype)
        pi_ok = np.clip(pi_ok, 0.0, 1.0)
        p[ok] = pi_ok

        rng_local = _rng_from(rng)
        U = rng_local.random(size=p.shape, dtype=dtype)
        hit_any = (U < p).any(axis=1)
        alive = ~hit_any
        contrib = payoff * alive.astype(dtype, copy=False)
    else:
        raise ValueError("method must be 'bridge_weight' or 'bridge_bernoulli'")

    disc = math.exp(-r * T)
    priced = disc * contrib

    price = float(np.mean(priced, dtype=np.float64))
    if not return_se:
        return price, None

    # Standard error
    se = float(np.std(priced, ddof=1, dtype=np.float64) / math.sqrt(len(priced)))
    return price, se
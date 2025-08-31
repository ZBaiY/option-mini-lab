# src/ou.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class OUParams:
    kappa: float
    mu: float
    sigma: float

def ou_transition_params(kappa: float, mu: float, sigma: float, dt: float) -> Tuple[float, float, float]:
    """
    Return (a, b, var_eps) for the AR(1) form:
        X_{t+dt} = a + b X_t + eps,  eps ~ N(0, var_eps)
    where b = exp(-kappa dt), a = mu(1 - b), var_eps = (sigma^2/(2kappa)) * (1 - b^2).
    """
    b = np.exp(-kappa * dt)
    a = mu * (1.0 - b)
    var_eps = (sigma**2) * (1.0 - b**2) / (2.0 * kappa)
    return a, b, var_eps

def simulate_ou_exact(n_paths: int, n_steps: int, dt: float, params: OUParams, x0: float,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Exact-discretization simulation using the Gaussian transition.
    Returns array with shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng() if rng is None else rng
    a, b, var_eps = ou_transition_params(params.kappa, params.mu, params.sigma, dt)
    x = np.empty((n_paths, n_steps + 1), dtype=float)
    x[:, 0] = x0
    std = np.sqrt(var_eps)
    for t in range(n_steps):
        eps = rng.normal(0.0, std, size=n_paths)
        x[:, t+1] = a + b * x[:, t] + eps
    return x

def simulate_ou_euler(n_paths: int, n_steps: int, dt: float, params: OUParams, x0: float,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Euler-Maruyama fallback:
        X_{t+dt} = X_t + kappa(mu - X_t) dt + sigma sqrt(dt) Z_t
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.empty((n_paths, n_steps + 1), dtype=float)
    x[:, 0] = x0
    drift = params.kappa * dt
    vol = params.sigma * np.sqrt(dt)
    for t in range(n_steps):
        z = rng.normal(0.0, 1.0, size=n_paths)
        x[:, t+1] = x[:, t] + drift * (params.mu - x[:, t]) + vol * z
    return x

@dataclass
class OUFit:
    kappa: float
    mu: float
    sigma: float
    a: float
    b: float
    sigma_eps: float
    stderr_a: float
    stderr_b: float

def fit_ou_mle(x: np.ndarray, dt: float) -> OUFit:
    """
    Fit OU via AR(1) OLS on consecutive pairs (exact MLE under Gaussian):
        x_{t+1} = a + b x_t + eps
    Recover:
        kappa = -ln b / dt, mu = a / (1 - b), sigma^2 = 2 kappa var_eps / (1 - b^2).
    x: shape (T,) or (T,1). Returns OUFit with parameter SEs for a,b.
    """
    x = np.asarray(x).reshape(-1)
    y = x[1:]
    X = np.column_stack([np.ones_like(x[:-1]), x[:-1]])  # [a, b]
    # OLS estimates
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_hat, b_hat = beta
    resid = y - X @ beta
    T = len(y)
    s2 = (resid @ resid) / (T - 2)  # unbiased residual variance
    # Var-cov of beta = s2 * (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = s2 * XtX_inv
    se_a = np.sqrt(var_beta[0, 0])
    se_b = np.sqrt(var_beta[1, 1])

    # Map back to OU
    if not (0 < b_hat < 1):
        # Clamp for numerical robustness; OU requires b in (0,1).
        b_hat = min(max(b_hat, 1e-8), 1 - 1e-8)
    kappa = -np.log(b_hat) / dt
    mu = a_hat / (1.0 - b_hat)
    var_eps = s2
    sigma = np.sqrt( (2.0 * kappa * var_eps) / (1.0 - b_hat**2) )

    return OUFit(
        kappa=kappa, mu=mu, sigma=sigma,
        a=a_hat, b=b_hat, sigma_eps=np.sqrt(var_eps),
        stderr_a=se_a, stderr_b=se_b
    )

def ou_residuals(x: np.ndarray, fit: OUFit) -> np.ndarray:
    """
    One-step prediction residuals under fitted AR(1) form.
    """
    x = np.asarray(x).reshape(-1)
    y = x[1:]
    X = np.column_stack([np.ones_like(x[:-1]), x[:-1]])
    return y - (fit.a + fit.b * x[:-1])

def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Simple (biased) ACF estimator up to max_lag (inclusive). Returns array of length max_lag+1.
    """
    x = np.asarray(x).reshape(-1)
    x = x - x.mean()
    denom = (x @ x)
    out = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        out[k] = (x[:len(x)-k] @ x[k:]) / denom
    return out

def qq_data(z: np.ndarray, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (theoretical_quantiles, sample_quantiles) vs N(0,1) for QQ plotting.
    Input z should be residuals standardized by their sample std.
    """
    z = np.sort(np.asarray(z).reshape(-1))
    idx = np.linspace(0.5, len(z) - 0.5, n_points)
    sample_q = np.interp(idx, np.arange(1, len(z)+1), z)
    # Theoretical standard normal quantiles at evenly spaced probs
    probs = (idx) / (len(z) + 1.0)
    theo_q = np.sqrt(2) * erfinv(2*probs - 1)
    return theo_q, sample_q

# small, local inv-erf to avoid heavy deps
def erfinv(y: np.ndarray) -> np.ndarray:
    # Winitzki approximation
    a = 0.147
    s = np.sign(y)
    ln = np.log(1 - y**2)
    first = (2/(np.pi*a) + ln/2)
    inner = first**2 - ln/a
    return s * np.sqrt( np.sqrt(inner) - first )

def half_life(kappa: float) -> float:
    """ Theoretical half-life ln(2)/kappa. """
    return np.log(2.0) / kappa
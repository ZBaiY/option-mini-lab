# tests/test_ou.py
import math
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ou import (
    OUParams,
    OUFit,
    ou_transition_params,
    simulate_ou_exact,
    simulate_ou_euler,
    fit_ou_mle,
    ou_residuals,
    acf,
    qq_data,
    half_life,
)


def _theory_mean_var(x0, mu, kappa, sigma, t):
    m = mu + (x0 - mu) * math.exp(-kappa * t)
    v = (sigma ** 2) * (1.0 - math.exp(-2.0 * kappa * t)) / (2.0 * kappa)
    return m, v


def test_transition_params_and_one_step_stats():
    params = OUParams(kappa=1.2, mu=0.7, sigma=0.9)
    dt = 0.05
    a, b, var_eps = ou_transition_params(params.kappa, params.mu, params.sigma, dt)

    # Simulate many independent one-step transitions and compare empirical to theory
    n = 200_000
    x0 = 0.3
    rng = np.random.default_rng(123)
    # Build one-step with exact AR(1) representation
    eps = rng.normal(0.0, math.sqrt(var_eps), size=n)
    x1 = a + b * x0 + eps

    emp_mean = float(x1.mean())
    emp_var = float(x1.var(ddof=0))

    # Theoretical mean/var at dt from continuous-time formula should match AR(1)
    m_theory, v_theory = _theory_mean_var(x0, params.mu, params.kappa, params.sigma, dt)

    assert math.isclose(emp_mean, m_theory, rel_tol=6e-3, abs_tol=6e-3)
    assert math.isclose(emp_var, v_theory, rel_tol=6e-3, abs_tol=6e-3)


@pytest.mark.parametrize("scheme", ["exact", "euler"])  # Euler should be close for small dt
def test_path_mean_var_against_theory(scheme):
    params = OUParams(kappa=0.8, mu=-0.2, sigma=0.5)
    T = 2.0
    n_steps = 400  # small dt -> Euler close to exact
    dt = T / n_steps
    n_paths = 30_000
    x0 = 1.1
    rng = np.random.default_rng(7)

    if scheme == "exact":
        X = simulate_ou_exact(n_paths, n_steps, dt, params, x0, rng=rng)
    else:
        X = simulate_ou_euler(n_paths, n_steps, dt, params, x0, rng=rng)

    XT = X[:, -1]
    m_emp, v_emp = float(XT.mean()), float(XT.var(ddof=0))
    m_th, v_th = _theory_mean_var(x0, params.mu, params.kappa, params.sigma, T)

    # Loose tolerances for Euler; tighter for exact
    rel = 0.01 if scheme == "exact" else 0.03
    assert math.isclose(m_emp, m_th, rel_tol=rel, abs_tol=5e-3)
    assert math.isclose(v_emp, v_th, rel_tol=rel, abs_tol=5e-3)


def test_mle_recovers_params_reasonably():
    # Long single path improves MLE
    true = OUParams(kappa=0.6, mu=0.4, sigma=0.7)
    T = 5000  # time steps
    dt = 0.01
    x0 = -0.8
    rng = np.random.default_rng(12345)
    X = simulate_ou_exact(1, T, dt, true, x0, rng=rng)[0]

    fit = fit_ou_mle(X, dt)

    assert abs(fit.kappa - true.kappa) < 0.05
    assert abs(fit.mu - true.mu) < 0.05
    assert abs(fit.sigma - true.sigma) < 0.05


def test_acf_matches_exp_decay():
    params = OUParams(kappa=1.1, mu=0.0, sigma=0.9)
    dt = 0.02
    n_steps = 12_000
    n_paths = 1
    rng = np.random.default_rng(99)

    # start from stationarity for cleaner ACF
    x0_mean, _ = _theory_mean_var(0.0, params.mu, params.kappa, params.sigma, 10.0)  # big t -> ~mu
    X = simulate_ou_exact(n_paths, n_steps, dt, params, x0=x0_mean, rng=rng)[0]

    rho = acf(X, max_lag=3)
    # Theoretical lag-1 autocorrelation ≈ exp(-kappa*dt)
    rho1_theory = math.exp(-params.kappa * dt)

    assert math.isclose(rho[1], rho1_theory, rel_tol=0.03, abs_tol=0.03)


def test_qq_data_residuals_approximately_normal():
    params = OUParams(kappa=0.9, mu=-0.1, sigma=0.6)
    dt = 0.01
    n_steps = 8000
    rng = np.random.default_rng(2024)
    X = simulate_ou_exact(1, n_steps, dt, params, x0=0.0, rng=rng)[0]

    fit = fit_ou_mle(X, dt)
    res = ou_residuals(X, fit)
    z = (res - res.mean()) / res.std(ddof=1)
    theo, samp = qq_data(z, n_points=200)

    # Fit a line samp ≈ a + b * theo; expect a≈0, b≈1
    A = np.column_stack([np.ones_like(theo), theo])
    beta, *_ = np.linalg.lstsq(A, samp, rcond=None)
    a_hat, b_hat = float(beta[0]), float(beta[1])

    assert abs(a_hat) < 0.05
    assert 0.95 < b_hat < 1.05


def test_half_life_formula():
    kappa = 0.7
    hl = half_life(kappa)
    # After half-life, exp(-kappa*hl) ≈ 0.5
    assert math.isclose(math.exp(-kappa * hl), 0.5, rel_tol=1e-12, abs_tol=1e-12)
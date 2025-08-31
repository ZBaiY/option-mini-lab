# tests/test_ou.py
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ou import OUParams, simulate_ou_exact, fit_ou_mle, acf, half_life

def test_stationary_mean_var_close():
    params = OUParams(kappa=1.2, mu=0.7, sigma=0.8)
    dt, steps, paths, x0 = 0.01, 20000, 200, 0.0
    x = simulate_ou_exact(paths, steps, dt, params, x0, rng=np.random.default_rng(123))
    long_run = x[:, -2000:].reshape(-1)
    mean_theory = params.mu
    var_theory = params.sigma**2 / (2*params.kappa)
    assert abs(long_run.mean() - mean_theory) < 0.05
    assert abs(long_run.var(ddof=1) - var_theory) < 0.05

def test_mle_recovers_params():
    true = OUParams(kappa=0.6, mu=-0.3, sigma=0.5)
    dt, steps, paths, x0 = 0.05, 4000, 1, -0.8
    x = simulate_ou_exact(paths, steps, dt, true, x0, rng=np.random.default_rng(42))[0]
    fit = fit_ou_mle(x, dt)
    # Delta-method SE for kappa: kappa = -ln(b)/dt → d kappa / d b = -1/(dt * b)
    se_kappa = fit.stderr_b / (dt * fit.b)
    # Use a 2.5-sigma band for a single long path (finite-sample variability of AR(1) slope can be nontrivial)
    assert abs(fit.kappa - true.kappa) < 2.5 * se_kappa + 1e-12
    assert abs(fit.mu - true.mu) < 0.10
    assert abs(fit.sigma - true.sigma) < 0.10
    # half-life consistency
    assert abs(half_life(fit.kappa) - np.log(2)/true.kappa) < 0.2

def test_acf_exponential_decay_shape():
    true = OUParams(kappa=0.8, mu=0.0, sigma=1.0)
    dt, steps, x0 = 0.02, 6000, 0.0
    x = simulate_ou_exact(1, steps, dt, true, x0, rng=np.random.default_rng(7))[0]
    r = acf(x, max_lag=20)
    # approximate: successive ratios roughly equal to exp(-kappa dt)
    ratio = r[2:]/r[1:-1]
    target = np.exp(-true.kappa * dt)
    assert np.allclose(ratio, target, atol=0.05)

if __name__ == "__main__":
    test_stationary_mean_var_close()
    test_mle_recovers_params()
    test_acf_exponential_decay_shape()
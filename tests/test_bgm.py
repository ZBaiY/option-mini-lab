import math
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import gbm


def test_terminal_mean_and_variance_match_theory():
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_paths = 200_000

    rng = np.random.default_rng(123)
    ST = gbm.sample_terminal(S0, mu, sigma, T, n_paths, rng=rng)

    m_emp = ST.mean()
    v_emp = ST.var()

    m_theory, v_theory = gbm.gbm_terminal_moments(S0, mu, sigma, T)

    # within 1% relative error
    assert math.isclose(m_emp, m_theory, rel_tol=0.01)
    assert math.isclose(v_emp, v_theory, rel_tol=0.02)


@pytest.mark.parametrize("antithetic", [True, False])
def test_paths_shape_and_terminal_distribution(antithetic):
    S0, mu, sigma, T = 100.0, 0.03, 0.25, 2.0
    n_steps, n_paths = 12, 5000
    rng = np.random.default_rng(42)

    t, paths = gbm.simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths,
                                      rng=rng, antithetic=antithetic)

    assert t.shape == (n_steps + 1,)
    assert paths.shape == (n_paths, n_steps + 1)

    # check first column is S0
    assert np.allclose(paths[:, 0], S0)

    # empirical terminal mean vs theory
    m_emp = paths[:, -1].mean()
    m_theory, _ = gbm.gbm_terminal_moments(S0, mu, sigma, T)
    assert math.isclose(m_emp, m_theory, rel_tol=0.05)


def test_euler_paths_are_nonnegative_with_small_dt():
    S0, mu, sigma, T = 100.0, 0.05, 0.1, 0.5
    n_steps, n_paths = 200, 1000
    rng = np.random.default_rng(7)

    t, paths = gbm.euler_paths(S0, mu, sigma, T, n_steps, n_paths, rng=rng)

    assert (paths >= 0.0).all()
    assert paths.shape == (n_paths, n_steps + 1)


def test_bs_call_and_put_parity():
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.2, 1.0
    call = gbm.bs_call(S0, K, r, sigma, T)
    put = gbm.bs_put(S0, K, r, sigma, T)

    lhs = call - put
    rhs = S0 - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)


def test_zero_volatility_and_zero_maturity():
    S0, K, r, sigma, T = 100.0, 110.0, 0.05, 0.0, 0.0
    call = gbm.bs_call(S0, K, r, sigma, T)
    put = gbm.bs_put(S0, K, r, sigma, T)
    assert call == max(S0 - K, 0.0)
    assert put == max(K - S0, 0.0)
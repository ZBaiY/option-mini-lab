# tests/test_barrier.py
import math
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.barrier import price_barrier_ko_call, bs_call


@pytest.mark.parametrize("antithetic", [True, False])
def test_barrier_price_converges_to_vanilla_when_barrier_is_very_low(antithetic):
    """
    For a down-and-out call, as H -> 0 the KO probability -> 0 and the price -> vanilla call.
    We compare the Monte Carlo estimate with a very low barrier to the BS closed-form price.
    """
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.02, 0.0, 0.2, 1.0
    H_very_low = 1e-8  # effectively unattainable for lognormal S > 0
    n_steps, n_paths = 64, 12_000

    rng = np.random.default_rng(12345)

    price_mc, se_mc = price_barrier_ko_call(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        barrier=H_very_low, kind="down", q=q,
        n_steps=n_steps, n_paths=n_paths,
        method="bridge_weight", rng=rng,
        antithetic=antithetic, return_se=True,
    )

    price_bs = bs_call(S=S0, K=K, r=r - q, sigma=sigma, T=T)

    # Allow either a small absolute tolerance or a few SEs
    tol = max(5e-3, 4.0 * se_mc)
    assert abs(price_mc - price_bs) < tol, f"MC {price_mc:.4f} vs BS {price_bs:.4f} (se={se_mc:.4f})"


def test_barrier_price_monotone_in_barrier_level_down_and_out():
    """
    For down-and-out calls, price should decrease as the barrier increases (more likely to knock out).
    """
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.01, 0.0, 0.25, 1.0
    n_steps, n_paths = 64, 10_000

    rng1 = np.random.default_rng(2024)
    rng2 = np.random.default_rng(2025)
    rng3 = np.random.default_rng(2026)

    H_low, H_mid, H_high = 50.0, 90.0, 99.0

    p_low, _ = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=H_low, kind="down",
        q=q, n_steps=n_steps, n_paths=n_paths,
        method="bridge_weight", rng=rng1,
        antithetic=True, return_se=True
    )
    p_mid, _ = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=H_mid, kind="down",
        q=q, n_steps=n_steps, n_paths=n_paths,
        method="bridge_weight", rng=rng2,
        antithetic=True, return_se=True
    )
    p_high, _ = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=H_high, kind="down",
        q=q, n_steps=n_steps, n_paths=n_paths,
        method="bridge_weight", rng=rng3,
        antithetic=True, return_se=True
    )

    assert p_low >= p_mid >= p_high, f"Monotonicity failed: {p_low:.4f}, {p_mid:.4f}, {p_high:.4f}"


def test_bridge_weight_variance_is_no_worse_than_bernoulli():
    """
    Bridge-weighting is typically lower-variance than Bernoulli KO simulation.
    We check that its SE is not larger (with a small slack for randomness).
    """
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.0, 0.3, 0.75
    n_steps, n_paths = 96, 16_000

    rng_w = np.random.default_rng(7)
    rng_b = np.random.default_rng(8)

    _, se_w = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=95.0, kind="down",
        q=q, n_steps=n_steps, n_paths=n_paths,
        method="bridge_weight", rng=rng_w,
        antithetic=True, return_se=True
    )
    _, se_b = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=95.0, kind="down",
        q=q, n_steps=n_steps, n_paths=n_paths,
        method="bridge_bernoulli", rng=rng_b,
        antithetic=True, return_se=True
    )

    # Allow 10% slack for randomness
    assert se_w <= 1.10 * se_b, f"Expected bridge-weight SE <= Bernoulli SE (got {se_w:.4f} vs {se_b:.4f})"


def test_immediate_maturity_reduces_to_intrinsic():
    """
    With T = 0 the price should equal discounted intrinsic value regardless of barrier.
    """
    S0, K, r, sigma, T = 110.0, 100.0, 0.05, 0.2, 0.0

    price, se = price_barrier_ko_call(
        S0, K, r, sigma, T, barrier=105.0, kind="down",
        q=0.0, n_steps=1, n_paths=1_000,
        method="bridge_weight", rng=np.random.default_rng(42),
        antithetic=True, return_se=True
    )
    # discounted intrinsic with T=0 is exactly payoff
    assert math.isclose(price, max(S0 - K, 0.0), rel_tol=0, abs_tol=1e-12)
    assert se == 0.0
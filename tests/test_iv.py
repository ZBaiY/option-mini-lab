# tests/test_iv_surface.py
import math
import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.iv_surface import (
    bs_price,
    implied_vol,
    Quote,
    IVSurface,
    build_surface,
    price_from_iv_grid,
    butterfly_violations,
    calendar_violations,
)

# ----------------------------- Implied vol solver ----------------------------- #

@pytest.mark.parametrize(
    "S,K,r,q,T,sigma_true,call",
    [
        (100.0, 100.0, 0.01, 0.00, 1.00, 0.20, True),
        (100.0, 90.0,  0.01, 0.00, 0.50, 0.35, True),
        (100.0, 110.0, 0.01, 0.02, 2.00, 0.15, True),
        (50.0,   60.0, 0.00, 0.00, 0.25, 0.40, True),
        # a put to check parity path as well
        (100.0, 100.0, 0.01, 0.00, 1.00, 0.20, False),
    ],
)
def test_implied_vol_roundtrip(S, K, r, q, T, sigma_true, call):
    price = bs_price(S, K, r, q, T, sigma_true, call)
    sigma_est = implied_vol(price, S, K, r, q, T, call)
    assert math.isclose(sigma_est, sigma_true, rel_tol=1e-6, abs_tol=1e-6)

# ---------------------------- Surface building basics ------------------------- #

def test_build_surface_recovers_constant_sigma():
    S, r, q = 100.0, 0.01, 0.00
    Ks = [80, 90, 100, 110, 120]
    Ts = [0.25, 0.5, 1.0, 2.0]
    sigma_true = 0.25
    # calls for all points
    Ks_arr, Ts_arr = np.meshgrid(Ks, Ts, indexing="xy")
    Ks_flat, Ts_flat = Ks_arr.ravel(), Ts_arr.ravel()
    prices = [bs_price(S, float(K), r, q, float(T), sigma_true, True)
              for K, T in zip(Ks_flat, Ts_flat)]

    surf = build_surface(S, r, q, Ks_flat, Ts_flat, prices, calls=True)

    # Mean absolute error should be tiny
    err = np.nanmean(np.abs(surf.iv - sigma_true))
    assert err < 1e-6

# ----------------------------- Interpolation tests ---------------------------- #

def test_iv_interpolation_linear_in_T_and_K():
    """Construct a 2x2 grid where true IV varies linearly in both T and K.
    Build prices from those IVs, then check that `iv_at` returns the bilinear
    value at the midpoint within a small tolerance (solver + interpolation).
    """
    S, r, q = 100.0, 0.005, 0.0
    Ks = [90.0, 110.0]
    Ts = [0.5, 1.5]

    # Define a linear plane: iv(T,K) = a + b*(K-100) + c*(T-1)
    a, b, c = 0.20, -0.0005, 0.02
    iv_grid = np.array([[a + b*(K-100.0) + c*(T-1.0) for K in Ks] for T in Ts])

    quotes = []
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            sig = float(iv_grid[i, j])
            p = bs_price(S, K, r, q, T, sig, True)
            quotes.append(Quote(S=S, K=K, T=T, r=r, q=q, is_call=True, price=p))

    surf = IVSurface.from_quotes(quotes)

    # Midpoint (K=100, T=1.0). For a linear plane, bilinear interp should match exactly.
    iv_mid_true = a + b*(100.0-100.0) + c*(1.0-1.0)
    iv_mid_est = surf.iv_at(100.0, 1.0)
    assert math.isclose(iv_mid_est, iv_mid_true, rel_tol=2e-5, abs_tol=2e-5)

# ------------------------- No-arbitrage (discrete checks) --------------------- #

def test_price_grid_has_no_discrete_arbitrage_for_constant_sigma():
    S, r, q = 100.0, 0.01, 0.0
    Ks = [80, 90, 100, 110, 120]
    Ts = [0.25, 0.5, 1.0, 2.0]
    sigma = 0.22

    # Build consistent prices and surface
    Ks_arr, Ts_arr = np.meshgrid(Ks, Ts, indexing="xy")
    Ks_flat, Ts_flat = Ks_arr.ravel(), Ts_arr.ravel()
    prices = [bs_price(S, float(K), r, q, float(T), sigma, True)
              for K, T in zip(Ks_flat, Ts_flat)]
    surf = build_surface(S, r, q, Ks_flat, Ts_flat, prices, calls=True)

    # Re-price from recovered IV grid
    C = price_from_iv_grid(surf, S=S, r=r, q=q)

    # Discrete butterfly convexity and calendar monotonicity should hold
    bviol = butterfly_violations(C, np.array(Ks, dtype=float))
    cviol = calendar_violations(C, np.array(Ts, dtype=float))
    assert bviol == []
    assert cviol == []

# ------------------------- Quotes with bid/ask mid handling ------------------- #

def test_quotes_with_bid_ask_are_parsed_via_mid():
    S, r, q = 100.0, 0.01, 0.0
    K, T, sigma = 100.0, 1.0, 0.2

    mid = bs_price(S, K, r, q, T, sigma, True)
    # construct a plausible bid/ask around the exact mid
    bid, ask = mid * (1 - 0.01), mid * (1 + 0.01)

    q1 = Quote(S=S, K=K, T=T, r=r, q=q, is_call=True, bid=bid, ask=ask)
    q2 = Quote(S=S, K=K, T=T, r=r, q=q, is_call=True, price=mid)

    surf = IVSurface.from_quotes([q1, q2])

    # The single grid point should be close to the true sigma
    est = float(surf.iv.squeeze())
    assert math.isclose(est, sigma, rel_tol=5e-5, abs_tol=5e-5)
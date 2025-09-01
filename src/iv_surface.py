"""
iv_surface.py — Implied Volatility Surface utilities

This module builds an implied volatility (IV) surface from option quotes,
provides robust implied-vol solvers, simple 2D interpolation, and
ready-to-use plotting helpers (smile/term-structure/surface).


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np

try:
    import matplotlib.pyplot as plt  # plotting is optional at runtime
except Exception:  # pragma: no cover - allow headless envs without pyplot
    plt = None  # type: ignore

###############################################################################
# Black–Scholes pricer (standalone to avoid cross-module assumptions)
###############################################################################

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if isinstance(x, float) else 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return inv_sqrt_2pi * math.exp(-0.5 * x * x) if isinstance(x, float) else inv_sqrt_2pi * np.exp(-0.5 * x * x)


def bs_price(S: float, K: float, r: float, q: float, T: float, sigma: float, call: bool = True) -> float:
    """Black–Scholes price for European call/put on dividend-paying stock.

    Parameters
    ----------
    S, K : spot and strike
    r, q : continuously compounded risk-free rate and dividend yield
    T    : time to maturity (in years)
    sigma: volatility (annualized)
    call : True for call, False for put

    Returns
    -------
    float

    Notes
    -----
    For T=0, returns intrinsic value.
    """
    if T <= 0 or sigma <= 0:
        if call:
            return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if call:
        return S * math.exp(-q * T) * float(_norm_cdf(d1)) - K * math.exp(-r * T) * float(_norm_cdf(d2))
    else:
        return K * math.exp(-r * T) * float(_norm_cdf(-d2)) - S * math.exp(-q * T) * float(_norm_cdf(-d1))


def bs_vega(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """Black–Scholes vega (∂Price/∂σ). Returns *per 1.0 of σ* (not %).
    Safe for small T by guarding sqrt(T).
    """
    if T <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * float(_norm_pdf(d1)) * sqrtT


###############################################################################
# Implied volatility solver
###############################################################################

def implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool = True,
    *,
    sigma_init: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
    bracket: Tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Compute Black–Scholes implied volatility via safeguarded Newton.

    Strategy
    --------
    1) Bracket with [lo, hi]. If model price at bounds does not straddle the
       target price, expand the bracket up to 5.0.
    2) Newton step using vega; if step goes out of bracket or vega small, fall
       back to bisection. Terminate when price error < tol.

    Robust to deep ITM/OTM options and short maturities.

    Examples
    --------
    >>> implied_vol(price=10.4506, S=100, K=100, r=0.01, q=0.0, T=1.0, call=True)
    0.2
    """
    # Guard trivial bounds: intrinsic and upper bound
    intrinsic = max(0.0, (S * math.exp(-q * T) - K * math.exp(-r * T)) if call else (K * math.exp(-r * T) - S * math.exp(-q * T)))
    # If given price below intrinsic (clean price), force tiny vol
    if price <= intrinsic + 1e-12:
        return bracket[0]

    lo, hi = bracket
    plo = bs_price(S, K, r, q, T, lo, call)
    phi = bs_price(S, K, r, q, T, hi, call)

    # Expand hi if needed (up to 5.0)
    while (plo - price) * (phi - price) > 0 and hi < 5.0:
        hi = min(5.0, 2.0 * hi)
        phi = bs_price(S, K, r, q, T, hi, call)

    # If still not bracketing, fallback to Newton from init and clip
    sigma = max(lo, min(hi, sigma_init))

    for _ in range(max_iter):
        model = bs_price(S, K, r, q, T, sigma, call)
        diff = model - price
        if abs(diff) < tol:
            return max(lo, min(hi, sigma))

        v = bs_vega(S, K, r, q, T, sigma)
        if v > 1e-12:
            step = diff / v
            new_sigma = sigma - step
            # If Newton escapes the bracket or becomes non-sensical, bisect
            if new_sigma <= lo or new_sigma >= hi or not (new_sigma == new_sigma):
                new_sigma = 0.5 * (lo + hi)
        else:
            new_sigma = 0.5 * (lo + hi)

        # Update bracket based on sign
        new_model = bs_price(S, K, r, q, T, new_sigma, call)
        if (new_model - price) * (plo - price) < 0:
            hi, phi = new_sigma, new_model
        else:
            lo, plo = new_sigma, new_model

        sigma = new_sigma

    # If not converged, return clipped sigma (best-effort)
    return max(lo, min(hi, sigma))


###############################################################################
# Quotes and Surface container
###############################################################################

@dataclass(frozen=True)
class Quote:
    """Single option quote.

    You can pass either a traded price (mid) or bid/ask to compute mid.
    If `price` is provided it takes precedence.
    """

    S: float
    K: float
    T: float  # years
    r: float  # cc risk-free
    q: float  # cc dividend/borrow
    is_call: bool
    price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def mid(self) -> float:
        if self.price is not None:
            return float(self.price)
        if self.bid is not None and self.ask is not None:
            return 0.5 * (float(self.bid) + float(self.ask))
        raise ValueError("Need either price or both bid & ask to compute mid price")


@dataclass
class IVSurface:
    """Simple rectangular IV surface on a (T × K) grid with separable linear interp.

    Attributes
    ----------
    maturities : np.ndarray, shape (M,)
    strikes    : np.ndarray, shape (N,)
    iv         : np.ndarray, shape (M, N) — NaN allowed for missing points

    Notes
    -----
    - Interpolation `iv_at(K, T)` performs linear interpolation in K within
      the two nearest maturities, then linear interpolation in T (a.k.a.
      *separable bilinear*). If an axis has only one point, it falls back to
      1D linear interpolation or nearest neighbor when outside range.
    - No-arbitrage is NOT enforced here.
    """

    maturities: np.ndarray  # (M,)
    strikes: np.ndarray  # (N,)
    iv: np.ndarray  # (M, N)

    # ------------------------------ building ------------------------------ #
    @staticmethod
    def from_quotes(quotes: Sequence[Quote]) -> "IVSurface":
        """Build a rectangular surface from arbitrary quotes.

        Quotes are grouped by (T, K); if multiple quotes land on the same node,
        their implied vols are averaged.
        """
        if len(quotes) == 0:
            raise ValueError("No quotes provided")

        maturities = np.unique(np.array([q.T for q in quotes], dtype=float))
        strikes = np.unique(np.array([q.K for q in quotes], dtype=float))
        M, N = len(maturities), len(strikes)
        grid = np.full((M, N), np.nan, dtype=float)
        counts = np.zeros_like(grid)

        # Compute IV for each quote and accumulate
        for q in quotes:
            iv = implied_vol(
                price=q.mid(), S=q.S, K=q.K, r=q.r, q=q.q, T=q.T, call=q.is_call
            )
            i = np.searchsorted(maturities, q.T)
            j = np.searchsorted(strikes, q.K)
            # searchsorted returns insertion index; since values exist, move back
            if i == M or maturities[i] != q.T:
                i -= 1
            if j == N or strikes[j] != q.K:
                j -= 1
            grid[i, j] = np.nanmean([grid[i, j], iv]) if not np.isnan(grid[i, j]) else iv
            counts[i, j] += 1

        # If duplicates averaged, ok; leave NaNs as missing (interpolated lazily)
        return IVSurface(maturities=maturities, strikes=strikes, iv=grid)

    # ------------------------------ interpolation ------------------------------ #
    def iv_at(self, K: float, T: float, *, allow_extrapolation: bool = False) -> float:
        """Interpolate the IV at arbitrary (T, K) via separable linear interp.

        If `allow_extrapolation=False`, values outside the convex hull return
        the nearest neighbor on each axis.
        """
        t = float(T)
        k = float(K)
        Ts = self.maturities
        Ks = self.strikes

        # Helper: 1D linear or nearest on a vector x with values y
        def _interp1d(xgrid: np.ndarray, ygrid: np.ndarray, xval: float) -> float:
            if not np.all(np.isfinite(ygrid)):
                # If missing values, use nearest finite value
                idx = np.where(np.isfinite(ygrid))[0]
                if idx.size == 0:
                    return np.nan
                return float(ygrid[idx[np.argmin(np.abs(xgrid[idx] - xval))]])
            if xval <= xgrid[0]:
                return float(ygrid[0]) if not allow_extrapolation else float(np.interp(xval, xgrid, ygrid))
            if xval >= xgrid[-1]:
                return float(ygrid[-1]) if not allow_extrapolation else float(np.interp(xval, xgrid, ygrid))
            return float(np.interp(xval, xgrid, ygrid))

        # If single maturity, do 1D in K
        if Ts.size == 1:
            return _interp1d(Ks, self.iv[0, :], k)
        # If single strike, do 1D in T
        if Ks.size == 1:
            return _interp1d(Ts, self.iv[:, 0], t)

        # Locate bracketing maturities
        i_hi = int(np.searchsorted(Ts, t, side="right"))
        i_lo = max(0, i_hi - 1)
        i_hi = min(i_hi, Ts.size - 1)

        if i_lo == i_hi:
            row = self.iv[i_lo, :]
            return _interp1d(Ks, row, k)

        # Interpolate in strike within each of the two rows
        row_lo = self.iv[i_lo, :]
        row_hi = self.iv[i_hi, :]
        v_lo = _interp1d(Ks, row_lo, k)
        v_hi = _interp1d(Ks, row_hi, k)

        # Then linear in time
        t0, t1 = Ts[i_lo], Ts[i_hi]
        if t <= t0:
            return v_lo if not allow_extrapolation else float(np.interp(t, [t0, t1], [v_lo, v_hi]))
        if t >= t1:
            return v_hi if not allow_extrapolation else float(np.interp(t, [t0, t1], [v_lo, v_hi]))
        w = (t - t0) / (t1 - t0)
        return float((1.0 - w) * v_lo + w * v_hi)

    # ------------------------------ export helpers ------------------------------ #
    def to_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return meshgrids (T_mesh, K_mesh, IV_mesh) for plotting."""
        Tm, Km = np.meshgrid(self.maturities, self.strikes, indexing="ij")
        return Tm, Km, self.iv.copy()

    # ------------------------------ plotting ------------------------------ #
    def plot_smile(self, T: float, ax: Optional["plt.Axes"] = None):
        """Plot IV(K) at a fixed maturity T (nearest maturity if not exact)."""
        if plt is None:
            raise RuntimeError("matplotlib is not available in this environment")
        # Find nearest maturity row
        i = int(np.argmin(np.abs(self.maturities - T)))
        Ks = self.strikes
        ivs = self.iv[i, :]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(Ks, ivs, marker="o")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"Smile at T≈{self.maturities[i]:.3f}y")
        ax.grid(True, ls=":", alpha=0.5)
        return ax

    def plot_term_structure(self, K: float, ax: Optional["plt.Axes"] = None):
        """Plot IV(T) at a fixed strike K (nearest strike if not exact)."""
        if plt is None:
            raise RuntimeError("matplotlib is not available in this environment")
        j = int(np.argmin(np.abs(self.strikes - K)))
        Ts = self.maturities
        ivs = self.iv[:, j]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(Ts, ivs, marker="o")
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"Term structure at K≈{self.strikes[j]:.4g}")
        ax.grid(True, ls=":", alpha=0.5)
        return ax

    def plot_surface(self):
        """Quick 3D wireframe surface of IV(T, K)."""
        if plt is None:
            raise RuntimeError("matplotlib is not available in this environment")
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        Tm, Km, Vm = self.to_mesh()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(Tm, Km, Vm)
        ax.set_xlabel("T (years)")
        ax.set_ylabel("K")
        ax.set_zlabel("IV")
        ax.set_title("Implied Vol Surface")
        return ax


###############################################################################
# Diagnostics & checks (optional helpers for notebooks)
###############################################################################

def price_from_iv_grid(surface: IVSurface, S: float, r: float, q: float) -> np.ndarray:
    """Re-price calls from the surface's IV grid using Black–Scholes.

    Returns a matrix C with the same shape as `surface.iv`.
    """
    Ts, Ks = surface.maturities, surface.strikes
    C = np.empty_like(surface.iv)
    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            sigma = float(surface.iv[i, j])
            C[i, j] = bs_price(S, float(K), r, q, float(T), sigma, True)
    return C


def butterfly_violations(C: np.ndarray, Ks: np.ndarray) -> list[tuple[str, int, int, float]]:
    """Discrete convexity check in strike (butterfly):
    For each maturity row i and interior strike j, checks
        C[i,j-1] + C[i,j+1] - 2*C[i,j] >= 0 .
    Returns a list of violations with tuples ("butterfly", i, j, value).
    """
    viol: list[tuple[str, int, int, float]] = []
    for i in range(C.shape[0]):
        for j in range(1, C.shape[1] - 1):
            lhs = float(C[i, j - 1] + C[i, j + 1] - 2.0 * C[i, j])
            if lhs < -1e-6:
                viol.append(("butterfly", i, j, lhs))
    return viol


def calendar_violations(C: np.ndarray, Ts: np.ndarray) -> list[tuple[str, int, int, float]]:
    """Discrete calendar monotonicity (non-decreasing in maturity):
    For each strike column j and adjacent maturities i -> i+1, checks
        C[i+1, j] - C[i, j] >= 0 .
    Returns a list of violations with tuples ("calendar", i, j, value).
    """
    viol: list[tuple[str, int, int, float]] = []
    for j in range(C.shape[1]):
        for i in range(C.shape[0] - 1):
            diff = float(C[i + 1, j] - C[i, j])
            if diff < -1e-6:
                viol.append(("calendar", i, j, diff))
    return viol


def implied_vol_trace(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    call: bool = True,
    *,
    sigma_init: float = 0.2,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Debug helper: implied-vol solver with a printed iteration trace.

    Mirrors the logic of `implied_vol`, printing each step; returns the final σ.
    Useful in notebooks to diagnose tricky quotes (deep ITM/OTM, tiny T).
    """
    lo, hi = 1e-6, 5.0
    intrinsic = max(0.0, (S * math.exp(-q * T) - K * math.exp(-r * T)) if call else (K * math.exp(-r * T) - S * math.exp(-q * T)))
    if price <= intrinsic + 1e-12:
        print("Price ≤ intrinsic, returning lo")
        return lo

    plo = bs_price(S, K, r, q, T, lo, call)
    phi = bs_price(S, K, r, q, T, hi, call)
    while (plo - price) * (phi - price) > 0 and hi < 5.0:
        hi = min(5.0, 2.0 * hi)
        phi = bs_price(S, K, r, q, T, hi, call)

    sigma = max(lo, min(hi, sigma_init))
    for it in range(max_iter):
        model = bs_price(S, K, r, q, T, sigma, call)
        diff = model - price
        print(f"it={it:02d} sigma={sigma:.10f} model={model:.10f} diff={diff:.3e}")
        if abs(diff) < tol:
            return max(lo, min(hi, sigma))

        v = bs_vega(S, K, r, q, T, sigma)
        if v > 1e-12:
            new_sigma = sigma - diff / v
            if new_sigma <= lo or new_sigma >= hi or not (new_sigma == new_sigma):
                new_sigma = 0.5 * (lo + hi)
        else:
            new_sigma = 0.5 * (lo + hi)

        new_model = bs_price(S, K, r, q, T, new_sigma, call)
        if (new_model - price) * (plo - price) < 0:
            hi, phi = new_sigma, new_model
        else:
            lo, plo = new_sigma, new_model
        sigma = new_sigma

    return max(lo, min(hi, sigma))

###############################################################################
# Convenience function: build from raw arrays
###############################################################################

def build_surface(
    S: float,
    r: float,
    q: float,
    Ks: Sequence[float],
    Ts: Sequence[float],
    prices: Sequence[float],
    calls: Sequence[bool] | bool,
) -> IVSurface:
    """Build an IVSurface from parallel arrays of quotes.

    Parameters
    ----------
    S, r, q : market inputs
    Ks, Ts  : sequences (same length as prices)
    prices  : option mid prices
    calls   : either a single boolean or a sequence of booleans of same length

    Returns
    -------
    IVSurface

    Examples
    --------
    >>> Ks = [90, 100, 110, 90, 100, 110]
    >>> Ts = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    >>> calls = [True]*3 + [True]*3
    >>> # Generate model-consistent prices at 20% vol and recover ~0.2
    >>> prices = [bs_price(100, K, 0.01, 0.0, T, 0.2, True) for K, T in zip(Ks, Ts)]
    >>> surf = build_surface(100, 0.01, 0.0, Ks, Ts, prices, calls)
    >>> float(np.nanmean(np.abs(surf.iv - 0.2)) < 1e-6)
    1.0
    """
    if isinstance(calls, bool):
        calls = [calls] * len(prices)
    if not (len(Ks) == len(Ts) == len(prices) == len(calls)):
        raise ValueError("Ks, Ts, prices, calls must have same length")
    quotes = [
        Quote(S=S, K=float(K), T=float(T), r=float(r), q=float(q), is_call=bool(c), price=float(p))
        for K, T, p, c in zip(Ks, Ts, prices, calls)
    ]
    return IVSurface.from_quotes(quotes)



__all__ = [
    "bs_price",
    "bs_vega",
    "implied_vol",
    "Quote",
    "IVSurface",
    "build_surface",
    "price_from_iv_grid",
    "butterfly_violations",
    "calendar_violations",
    "implied_vol_trace",
]

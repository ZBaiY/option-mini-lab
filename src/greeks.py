import numpy as np
from scipy.stats import norm

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    """
    Black–Scholes Delta.
    option: 'call' or 'put'
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes Gamma.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes Vega.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    """
    Black–Scholes Theta.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def bs_rho(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    """
    Black–Scholes Rho.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# =========================
# Monte Carlo Greek Estimators
# =========================

def _simulate_Z(n_paths: int, seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """
    Generate standard normal draws. If antithetic=True, concatenates Z and -Z to halve variance.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    if antithetic:
        Z = np.concatenate([Z, -Z])
    return Z


def _ST_from_Z(S0: float, r: float, sigma: float, T: float, Z: np.ndarray) -> np.ndarray:
    """
    Map N(0,1) draws Z to terminal GBM prices under risk-neutral dynamics.
    S_T = S0 * exp( (r - 0.5*sigma^2) T + sigma * sqrt(T) * Z )
    """
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    return S0 * np.exp(drift + vol * Z)


def _discount(r: float, T: float) -> float:
    return float(np.exp(-r * T))


def _ci95(mean: float, std_err: float) -> tuple[float, float]:
    """
    95% normal-theory confidence interval.
    """
    half = 1.96 * std_err
    return mean - half, mean + half


def mc_price_euro_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Monte Carlo price for a European call with variance reduction via antithetic variates.

    Returns
    -------
    price : float
    std_err : float
    ci95 : (float, float)
    """
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    ST = _ST_from_Z(S0, r, sigma, T, Z)
    disc = _discount(r, T)
    payoff = np.maximum(ST - K, 0.0)
    disc_payoff = disc * payoff
    price = float(disc_payoff.mean())
    std_err = float(disc_payoff.std(ddof=1) / np.sqrt(disc_payoff.size))
    return price, std_err, _ci95(price, std_err)


def mc_delta_pathwise_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Pathwise (PW) delta estimator for a European call:
        Delta = E[ e^{-rT} * 1_{S_T > K} * (S_T / S0) ].

    This estimator is unbiased for continuously differentiable payoffs; for the call,
    the derivative exists almost everywhere and the PW estimator performs well.

    Returns (est, std_err, ci95)
    """
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    ST = _ST_from_Z(S0, r, sigma, T, Z)
    disc = _discount(r, T)
    indicator = (ST > K).astype(float)
    samples = disc * indicator * (ST / S0)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_vega_pathwise_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Pathwise (PW) vega estimator for a European call using the same Z draws.

    Using S_T = S0 * exp( (r - 0.5*sigma^2)T + sigma*sqrt(T)*Z ),
    dS_T/dsigma = S_T * ( -sigma*T + sqrt(T)*Z ).

        Vega = E[ e^{-rT} * 1_{S_T > K} * dS_T/dsigma ].

    Note: This returns vega in price per 1.0 change of sigma (i.e., not scaled by 0.01).
    """
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    ST = _ST_from_Z(S0, r, sigma, T, Z)
    disc = _discount(r, T)
    dST_dsig = ST * (-sigma * T + np.sqrt(T) * Z)
    samples = disc * (ST > K).astype(float) * dST_dsig
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_delta_LR_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Likelihood-Ratio (LR, aka score function) delta estimator for a European call:

        score_S0 = ∂ log p(S_T; S0) / ∂ S0
                  = [ ln(S_T/S0) - (r - 0.5*sigma^2)T ] / (S0 * sigma^2 * T)

        Delta = E[ e^{-rT} * (S_T - K)^+ * score_S0 ].

    LR works even for discontinuous payoffs (e.g., digitals, barriers).
    """
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    ST = _ST_from_Z(S0, r, sigma, T, Z)
    disc = _discount(r, T)
    payoff = np.maximum(ST - K, 0.0)
    score = (np.log(ST / S0) - (r - 0.5 * sigma**2) * T) / (S0 * sigma**2 * T)
    samples = disc * payoff * score
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_delta_fd_central(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, bump_rel: float = 1e-4,
    antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Central finite-difference (bump-and-revalue) delta with common random numbers (CRN):
        Delta ≈ [C(S0(1+h)) - C(S0(1-h))] / [2*S0*h].

    Uses the same Z draws for both bumped evaluations to reduce variance.
    """
    h = bump_rel
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    disc = _discount(r, T)

    S0p = S0 * (1.0 + h)
    S0m = S0 * (1.0 - h)

    ST_p = _ST_from_Z(S0p, r, sigma, T, Z)
    ST_m = _ST_from_Z(S0m, r, sigma, T, Z)

    Cp = disc * np.maximum(ST_p - K, 0.0)
    Cm = disc * np.maximum(ST_m - K, 0.0)

    samples = (Cp - Cm) / (2.0 * S0 * h)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_gamma_fd_central(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, bump_rel: float = 1e-4,
    antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Central finite-difference gamma with CRN:
        Gamma ≈ [C(S0(1+h)) - 2C(S0) + C(S0(1-h))] / (S0*h)^2.
    """
    h = bump_rel
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    disc = _discount(r, T)

    ST_0 = _ST_from_Z(S0, r, sigma, T, Z)
    ST_p = _ST_from_Z(S0 * (1.0 + h), r, sigma, T, Z)
    ST_m = _ST_from_Z(S0 * (1.0 - h), r, sigma, T, Z)

    C0 = disc * np.maximum(ST_0 - K, 0.0)
    Cp = disc * np.maximum(ST_p - K, 0.0)
    Cm = disc * np.maximum(ST_m - K, 0.0)

    samples = (Cp - 2.0 * C0 + Cm) / ((S0 * h) ** 2)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_vega_fd_central(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, bump_abs: float = 1e-4,
    antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Central finite-difference vega with CRN:
        Vega ≈ [C(sigma + h) - C(sigma - h)] / (2h).
    """
    h = bump_abs
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)
    disc = _discount(r, T)

    ST_p = _ST_from_Z(S0, r, sigma + h, T, Z)
    ST_m = _ST_from_Z(S0, r, sigma - h, T, Z)

    Cp = disc * np.maximum(ST_p - K, 0.0)
    Cm = disc * np.maximum(ST_m - K, 0.0)

    samples = (Cp - Cm) / (2.0 * h)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)

def mc_theta_fd_central(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, bump_abs: float = 1e-4,
    antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Central finite-difference Theta with CRN:
        Theta ≈ [C(T + h) - C(T - h)] / (2h)

    Notes
    -----
    * Uses common Z for T+h and T-h to reduce variance.
    * If T is very small, h is automatically shrunk to avoid negative time.
    * Theta here is dPrice/dT (per year). For per-day, divide by 252 (or 365).
    """
    # Guard for tiny maturities to keep T - h > 0
    h = min(bump_abs, 0.49 * T) if T > 0 else bump_abs

    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)

    # Prices at T+h and T-h (both S_T and discount depend on T)
    ST_p = _ST_from_Z(S0, r, sigma, T + h, Z)
    ST_m = _ST_from_Z(S0, r, sigma, T - h, Z)

    Cp = np.exp(-r * (T + h)) * np.maximum(ST_p - K, 0.0)
    Cm = np.exp(-r * (T - h)) * np.maximum(ST_m - K, 0.0)

    samples = (Cp - Cm) / (2.0 * h)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


def mc_rho_fd_central(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, bump_abs: float = 1e-4,
    antithetic: bool = True, seed: int | None = None
) -> tuple[float, float, tuple[float, float]]:
    """
    Central finite-difference Rho with CRN:
        Rho ≈ [C(r + h) - C(r - h)] / (2h)

    Notes
    -----
    * Uses common Z for r+h and r-h to reduce variance.
    * Both discount and GBM drift depend on r.
    """
    h = bump_abs
    Z = _simulate_Z(n_paths, seed=seed, antithetic=antithetic)

    # r affects both discounting and S_T drift
    ST_p = _ST_from_Z(S0, r + h, sigma, T, Z)
    ST_m = _ST_from_Z(S0, r - h, sigma, T, Z)

    Cp = np.exp(-(r + h) * T) * np.maximum(ST_p - K, 0.0)
    Cm = np.exp(-(r - h) * T) * np.maximum(ST_m - K, 0.0)

    samples = (Cp - Cm) / (2.0 * h)
    est = float(samples.mean())
    std_err = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return est, std_err, _ci95(est, std_err)


# Option Mini Lab

A hands-on project connecting **Itô calculus** with working Python code for option pricing, risk sensitivities, and mean-reverting models.  
The lab bridges **theory ↔ implementation**: SDEs → PDEs → pricing → Greeks → path dependence → calibration.

---

## What's inside

### Core modules (`src/`)
- **`gbm.py`**:  
  Geometric Brownian Motion simulator, Black–Scholes closed-form pricer, and Monte Carlo estimator with confidence intervals.
- **`greeks.py`**:  
  Estimators for Delta, Gamma, Vega, Theta, Rho (pathwise, likelihood-ratio, finite-difference), variance/bias benchmarking.
- **`barrier.py`**:  
  Barrier option pricing (knock-out), Brownian-bridge correction for discrete monitoring bias.
- **`ou.py`**:  
  Ornstein–Uhlenbeck process simulation + exact discretization, MLE calibration, ACF/QQ checks for stationarity.
- **`iv_surface.py`**:  
  Implied volatility surface construction, bid/ask spreads, smile/skew visualization.


### Notebooks (`notebooks/`)
- **`demo_bs.ipynb`**: BS PDE derivation, MC vs closed-form convergence.  
- **`demo_greeks.ipynb`**: Greeks estimation, variance vs bias.  
- **`demo_barrier.ipynb`**: Barrier option simulations, correction demo.  
- **`demo_ou.ipynb`**: OU simulation, calibration, diagnostics.  

### Documentation (`docs/`)
Exported reports in PDF format, generated from the notebooks and suitable for offline viewing or sharing:
- **`demo_bs.pdf`**
- **`demo_greeks.pdf`**
- **`demo_barrier.pdf`**
- **`demo_ou.pdf`**
- **`demo_iv_surface.pdf`**

### Tests (`tests/`)
- Sanity checks with `pytest`:  
  - MC vs analytic BS pricing.  
  - Greeks vs BS closed-form.  
  - Barrier option knock-out logic.  
  - OU parameter recovery via MLE.

---

## Why it’s useful
- **Bridges theory and code**: from stochastic calculus to practical pricing and calibration.  
- **Reusable modules**: clean Python implementations with unit tests.  
- **Market realism**: incorporates bid/ask spreads, barriers, and calibration.  
- **Learning by doing**: notebooks connect formulas, intuition, and code.

---

## Getting started

```bash
git clone https://github.com/your-username/option-mini-lab.git
cd option-mini-lab
pip install -r requirements.txt
pytest   # run tests
```

Then explore the `notebooks/` to see derivations, figures, and pricing demos.

---

## Roadmap
- Add control variates and Sobol quasi-MC for variance reduction.  
- Implement Crank–Nicolson PDE solver and cross-validate with MC.  
- Extend barrier module to down-and-out and knock-in variants.  
- Expand implied-volatility surface and add Heston/CIR models.  
- Benchmark NumPy vs Numba JIT implementations.

---

## License
MIT License.
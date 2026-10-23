# quantkit design notes

This document explains how quantkit is put together, why it is built that way, and how its correctness is checked. It is meant to be read alongside the code: each section names the module it describes.

## 1. Goals and non-goals

**Goals**

- Implement the standard models of an introductory quantitative finance course (option pricing, mean–variance and risk-parity portfolios, backtesting) in code that a reader can follow line by line.
- Verify every model against something it did not produce itself: a textbook value, a closed form, an independent implementation, or an identity.
- Fail loudly: validate inputs at the public boundary and raise typed errors with messages that name the problem.

**Non-goals**

- Speed at scale. The pricing API is scalar, the optimizers use a general-purpose solver, and the backtester is daily. These are deliberate trade-offs for readability.
- Market data. There is no downloader; users bring a CSV, and examples use clearly labelled synthetic data.
- Production trading features: order management, execution simulation, intraday data, corporate actions.

## 2. Architecture

```text
src/quantkit/
├── exceptions.py        QuantkitError → InvalidParameterError (ValueError) → ArbitrageBoundError, DataValidationError
│                        QuantkitError → ConvergenceError (RuntimeError)
├── _validation.py       shared argument checks and type aliases (FloatArray, SeedLike)
├── pricing/             option.py (OptionSpec) · black_scholes.py · implied_vol.py · binomial.py
│                        monte_carlo.py · asian.py
├── portfolio/           estimators.py · mean_variance.py · risk_parity.py · risk.py
├── backtest/            engine.py · strategies.py · metrics.py
├── data.py              synthetic GBM prices, CSV loader
└── cli.py, __main__.py  argparse front end (`quantkit ...`, `python -m quantkit`)
```

Dependencies point one way: `pricing`, `portfolio` and `backtest` depend only on `_validation` and `exceptions` (and `backtest.engine` on `backtest.metrics`); `data` depends on `_validation`; only `cli` depends on everything, and only `__main__` imports `cli`. This keeps each subpackage testable on its own.

**Conventions used everywhere**

| Quantity | Convention |
| --- | --- |
| Rates, yields, volatilities, returns | Decimals (`0.05` = 5%), annualized unless the name says otherwise |
| Time | Years for option maturities; trading days (252 per year) for price series |
| Price tables | Wide `DataFrame`: sorted `DatetimeIndex` named `date`, one column per ticker, no missing values |
| Weights | `pandas.Series` indexed by ticker; inputs are aligned by label, never by position |
| Greeks | Exact partial derivatives: vega per 1.00 of volatility, theta per year, rho per 1.00 of rate |
| Risk numbers | VaR, CVaR and maximum drawdown are positive loss fractions |

**Errors.** Every deliberate error derives from `QuantkitError`. Bad inputs also derive from `ValueError` and solver failures from `RuntimeError`, so callers can use either the library's or the standard library's hierarchy. `ArbitrageBoundError` carries the offending price and both bounds as attributes.

## 3. Option pricing (`quantkit.pricing`)

### `OptionSpec`

A frozen, keyword-only, slotted dataclass holding the contract (strike, maturity, call/put) and the market (spot, rate, volatility, dividend yield). Keyword-only fields make `OptionSpec(spot=42, strike=40, ...)` impossible to transpose by accident; `frozen=True` makes specs hashable and safe to share; `dataclasses.replace(option, vol=0.25)` derives bumped copies, which is how the finite-difference tests work. `__post_init__` validates every field and stores plain floats, so NumPy scalars do not leak into reprs. Exercise style is an argument of the pricer, not a field, because the same contract is priced as European by BSM and as either style by the tree.

**Trade-off:** scalar inputs keep formulas readable but make pricing a large option chain a Python loop. A vectorized interface is on the roadmap.

### Black–Scholes–Merton (`black_scholes.py`)

$$c = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad p = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

- `norm_cdf(x) = 0.5 * erfc(-x / sqrt(2))`. The textbook `0.5 * (1 + erf(x / sqrt(2)))` subtracts two nearly equal numbers for very negative `x` and returns 0 for deep out-of-the-money options; `erfc` keeps full relative precision. The test suite compares it with SciPy's `ndtr` down to `x = -37`.
- The Greeks are the analytic derivatives (Hull's formulas with a dividend yield), with `theta = -dV/dT` (calendar-time decay, per year).

### Implied volatility (`implied_vol.py`)

1. Compute the model-free bounds `max(PV(S) - PV(K), 0) < call < PV(S)` (and the put analogue). A price outside them admits a static arbitrage, and no volatility can produce it, so `ArbitrageBoundError` is raised before any solving.
2. Inside the bounds, `f(sigma) = BSM(sigma) - price` is continuous and strictly increasing (vega > 0), so it has exactly one root.
3. Solve with Brent's method on `[1e-6, 10]`. Brent keeps the root bracketed at every step, so it cannot diverge. Newton's method converges faster near the root but can overshoot wildly when vega is tiny (deep in or out of the money).
4. If the root lies outside the interval (the price sits numerically on a bound), raise `ConvergenceError` instead of returning a misleading endpoint.

### Cox–Ross–Rubinstein tree (`binomial.py`)

- `u = exp(sigma sqrt(dt))`, `d = 1/u`, `p = (exp((r - q) dt) - d) / (u - d)`. If `p` is not in (0, 1), the tree is not arbitrage-free (the drift over one step exceeds an up-move), so the function raises and suggests more steps.
- Node prices are computed as `S * exp(log_u * (2j - i))` rather than `S * u**j * d**(i-j)`, which avoids overflow and round-off for large trees.
- Backward induction is vectorized per level: `V_i = disc * (p * V_{i+1}[1:] + (1-p) * V_{i+1}[:-1])`. Total cost is O(N²) time and O(N) memory: a few milliseconds for 1,000 steps on a laptop.
- **American exercise** replaces each node value with `max(continuation, exercise value)`.
- **Convergence** to BSM is O(1/N) but not monotone: the error oscillates as nodes move relative to the strike. The tests therefore compare the *worst* error over windows of consecutive step counts, rather than assuming the error falls at every step.
- **American call = European call when q = 0 and r ≥ 0** (Merton 1973): the call is worth at least `S - K e^{-r(T-t)} ≥ S - K` alive, so early exercise never helps. The tree reproduces this to 1e-12; with `r < 0` or `q > 0` the result does not hold, and the tests include a high-dividend case where the American call is worth more.

### Monte Carlo (`monte_carlo.py`)

- GBM has the exact solution `S_T = S exp((r - q - sigma²/2) T + sigma sqrt(T) Z)`, so a European payoff needs one normal draw per path and has no discretization error, only sampling error.
- **Antithetic variates** pair each `Z` with `-Z`. A call or put payoff is monotone in `Z`, so the two halves of a pair are negatively correlated and their average has lower variance than two independent draws. The standard error is computed from the pair averages (which are i.i.d.), not from the 2n correlated payoffs; using the latter would understate the error.
- The 95% interval is `price ± 1.96 SE` (central limit theorem). A test runs 400 seeds and checks that the interval covers the BSM price in 92–98% of runs, a direct check that the standard error is right.
- Randomness comes from `numpy.random.default_rng(seed)`; nothing touches global random state.

### Asian options (`asian.py`)

The geometric average `G` of lognormal prices is lognormal. With `n` equally spaced fixings, `ln G` has mean `ln S + (r - q - sigma²/2) T (n+1)/(2n)` and variance `sigma² T (n+1)(2n+1)/(6n²)`, which gives a Black–Scholes-type formula (Kemna & Vorst 1990). With `n = 1` the formula reduces to BSM, which the tests check to 1e-12.

The arithmetic option has no closed form. The Monte Carlo estimator subtracts the geometric payoff's simulation error, whose mean is known: `Y_cv = Y - b (X - E[X])` with `b = Cov(Y, X) / Var(X)`. Because the two averages are almost perfectly correlated, the standard error falls by more than 10x in the tests. Estimating `b` from the same paths adds a bias of order 1/n_paths (Glasserman 2003, §4.1), which is negligible at these sample sizes; a pilot run would remove it at the cost of extra code.

## 4. Portfolio construction (`quantkit.portfolio`)

### Estimation (`estimators.py`)

- Simple returns aggregate across assets (a portfolio's return is `w'R`), which is what mean–variance needs; log returns aggregate across time.
- Annualization multiplies means and covariances by 252. This assumes returns are serially uncorrelated.
- **Ledoit–Wolf (2004)** shrinks the sample covariance `S` towards a scaled identity `m I` (`m` = average variance): `Σ̂ = δ m I + (1 - δ) S`. The intensity `δ = min(b̄², d²) / d²` weighs how noisy `S` is (`b̄²`, the average squared distance of the one-observation estimates `x_k x_k'` from `S`) against how far `S` is from the target (`d²`). The implementation uses the identity `Σ_k ||x_k x_k' - S||² = Σ_k ||x_k||⁴ - n ||S||²` to avoid building n matrices. It follows the paper's `1/n` normalization and matches scikit-learn's `ledoit_wolf` to 1e-10 in the tests. The shrunk matrix is always positive definite, even with more assets than observations, which is what matters to an optimizer that inverts it.
- *Alternative not taken:* the constant-correlation target ("Honey, I shrunk the sample covariance matrix") suits equities better but is less standard to validate against.

### Mean–variance (`mean_variance.py`)

**Closed forms (unbounded weights, shorting allowed).** Minimizing `w'Σw` subject to `1'w = 1` gives `Σw ∝ 1`, so `w_GMV = Σ⁻¹1 / (1'Σ⁻¹1)`. The tangency portfolio is `w ∝ Σ⁻¹(μ - r_f 1)`. For a target return `m`, with `A = 1'Σ⁻¹1`, `B = 1'Σ⁻¹μ`, `C = μ'Σ⁻¹μ`, `D = AC - B²` (Merton 1972):

$$w(m) = \frac{(C - mB)\,\Sigma^{-1}\mathbf{1} + (mA - B)\,\Sigma^{-1}\mu}{D}, \qquad \sigma^2(m) = \frac{Am^2 - 2Bm + C}{D}$$

The code uses `np.linalg.solve` instead of forming `Σ⁻¹`. The tangency formula needs `1'Σ⁻¹(μ - r_f) > 0`, i.e. the GMV portfolio must earn more than `r_f`; otherwise the normalization flips the sign and returns the *minimum*-Sharpe portfolio, so the function raises instead.

**Numerical optimizer (SLSQP).** Bounds, which are what real portfolios need, have no closed form, so `min_variance`, `max_sharpe` and `efficient_return` use `scipy.optimize.minimize(method="SLSQP")` with analytic gradients. Implementation details that matter:

- *The closed forms validate the optimizer.* Run with `bounds=None`, SLSQP must reproduce them; the tests require weights within 1e-6 and the objective (variance or Sharpe ratio) within about 1e-10 in relative terms.
- *Objective scaling.* SLSQP stops on an absolute change in the objective. Variances are around 0.01, so the variance objective is divided by the equal-weight portfolio's variance to make the tolerance unit-free.
- *Maximum Sharpe as a convex QP.* The Sharpe ratio is not concave in `w`. During development, maximizing it directly with SLSQP occasionally stopped early on random test problems ("Positive directional derivative for linesearch"). With `e = μ - r_f`, the substitution `κ = 1/(e'w)`, `y = κw` turns the problem into: minimize `y'Σy` subject to `e'y = 1`, `Σy_i = κ ≥ 0` and `lower·κ ≤ y_i ≤ upper·κ` (Cornuéjols & Tütüncü). For any portfolio with positive excess return the objective equals `1/Sharpe²`, so the minimizer is the tangency portfolio, and `w = y/κ` recovers it. All constraints stay linear, so the problem is a convex QP.
- *Attainable returns.* With the same box `[l, u]` for every weight, the highest attainable return is a fractional knapsack: start at `l` everywhere and spend the remaining budget on the best assets first. This gives exact feasibility checks for target returns without a linear-programming solver.
- *Top of the frontier.* At exactly the highest attainable return the feasible set collapses to a corner where more constraints are active than there are free variables, and SLSQP's linearized subproblem can become inconsistent. The default frontier stops one millionth of the return range below that corner.
- *Clean-up.* SLSQP satisfies bounds up to round-off (for example `-1e-17` for long-only), so final weights are clipped to the bounds.

### Risk parity (`risk_parity.py`)

Asset `i` contributes `RC_i = w_i (Σw)_i / (w'Σw)`; by Euler's theorem the contributions sum to one. Maillard, Roncalli & Teïletche (2010) show that the equal-contribution portfolio solves the strictly convex problem

$$\min_{y>0}\; \tfrac12\, y'\Sigma y - \sum_i b_i \ln y_i ,$$

whose first-order condition is `y_i (Σy)_i = b_i`; normalizing `w = y / Σy` gives the portfolio. With the other coordinates fixed, the objective is minimized in `y_i` by the positive root of `Σ_ii y_i² + c_i y_i - b_i = 0` with `c_i = Σ_{j≠i} Σ_ij y_j`, so the code uses cyclical coordinate descent with that exact update (Griveau-Billion, Richard & Roncalli 2013): no step sizes, no line search, guaranteed descent. It starts from `y_i = sqrt(b_i)/σ_i`, which is the exact answer for uncorrelated assets, and stops when every contribution is within `1e-10` of its budget.

### Risk metrics (`risk.py`)

- **Historical VaR** uses the `m = ceil(n (1 - level))` worst returns as the tail; VaR is the smallest loss in the tail (the empirical `1 - level` quantile) and CVaR is the tail mean. For 99% and 500 observations that is the 5th-worst return. In floating point `(1 - 0.99) * 500 = 5.000000000000004`, so `ceil` would give 6; a small slack constant fixes this, and a test pins it down.
- **Gaussian VaR/CVaR**: `z σ - μ` and `σ φ(z) / (1 - level) - μ`, using the standard library's `statistics.NormalDist`.

## 5. Backtesting (`quantkit.backtest`)

### The timing contract

```text
close of day t                                  day t+1
──────────────────────────────────────────────  ─────────────────────────────────
strategy sees prices[0..t] → target weights W[t]
trade from drifted weights D[t] to W[t]         hold W[t]; earn r[t+1] = P[t+1]/P[t] - 1
pay cost_bps x turnover[t]                      weights drift with prices
```

A strategy is any callable `prices -> target weights` with the same shape, where row `t` may use only rows `0..t`. A fully specified row means "trade to these weights at today's close"; an all-NaN row means "do not trade today" (positions drift); a partially specified row is rejected as ambiguous. Weights need not sum to one (`1 - Σw` is cash at 0%), and negative weights are shorts.

### Accounting

With `h[t]` the weights held during day `t` (`h[0] = 0`, all cash):

```text
gross[t]    = Σ_i h[t,i] r[t,i]
drifted[t]  = h[t] (1 + r[t]) / (1 + gross[t])            weights just before trading
turnover[t] = Σ_i |W[t,i] - drifted[t,i]|  on trade days, else 0
net[t]      = (1 + gross[t]) (1 - c · turnover[t]) - 1      c = cost_bps / 10,000
equity[t]   = initial_capital · Π_{s≤t} (1 + net[s])
```

- **Turnover is drift-aware.** Comparing today's target with yesterday's target (as many simple backtesters do) charges buy-and-hold for "trading" every day as its weights drift. Comparing with the drifted weights charges only real trades: buy-and-hold has turnover 1.0 on day 0 and zero afterwards. Turnover is two-sided (switching 100% from A to B is 2.0) because each leg pays the cost.
- **Costs are multiplicative.** Paying `c × turnover` of wealth scales the whole portfolio, so weights and turnover do not depend on the cost level, and wealth with costs equals wealth without costs times `Π(1 - c · turnover)` exactly. The test suite checks this identity.

### Vectorization

A day-by-day loop would be the most obvious implementation. The engine avoids it with one observation: between trades the share counts are fixed, so on any day the value of each position is its weight at the last trade (the "anchor") times its price growth since then, and cash is constant. Hence

```text
post_trade[t] = W[s] ⊙ (P[t] / P[s]) / (Σ_i W[s,i] P[t,i]/P[s,i] + 1 - Σ_i W[s,i])
```

for the last trade day `s ≤ t`, computed for all days with a forward fill. The tests run the vectorized engine and an independent reference loop on random target tables (with cash, shorts, leverage and no-trade days) and require agreement to 1e-10.

### Avoiding look-ahead

- The engine applies the weights decided at `t` to the return of `t+1`, and every quantity at `t` uses only data up to `t`.
- The built-in strategies use trailing windows only (`rolling`, `shift` with positive lags).
- **Calendar look-ahead:** rebalancing on the *last* trading day of each month looks innocent but is not causal, because on a given day you cannot know it is the last trading day without tomorrow's date. `rebalance_days` flags the *first* trading day of each period, which needs only today's and yesterday's dates.
- **Tests:** for every strategy and several cutoffs `t`, prices after `t` are multiplied by random factors and the backtest is re-run; the weights in effect on days `0..t+1`, and equity and turnover through day `t`, must be identical. A deliberately cheating strategy (buying whatever rises tomorrow) is a negative control: it fails the same check, and its equity explodes.
- `check_no_lookahead(strategy, prices)` exposes the strategy-level check for custom strategies.

### Metrics (`metrics.py`)

Sample standard deviations use `ddof=1`. The annual risk-free rate is converted to a per-period rate that compounds to it. Drawdowns are measured from a running peak that starts at the initial capital, so a loss on the first day counts. Maximum drawdown duration is the longest run of periods below a previous peak, until recovery or the end of the data. Hit rate ignores zero-return periods (for example, days spent in cash). Ratios with a zero denominator, or one that is zero up to round-off, return NaN rather than infinity.

## 6. How correctness is verified

| Kind of check | Examples |
| --- | --- |
| Textbook values | Hull's BSM call and put; Hull's Greeks example; Hull's American put on 5–500-step trees |
| Closed forms | Unbounded mean–variance portfolios; inverse-volatility risk parity for uncorrelated or two assets; geometric Asian with one fixing = BSM; Gaussian VaR |
| Independent implementations | scikit-learn's Ledoit–Wolf; a day-by-day reference backtest; a separate simulation of the geometric average; finite-difference Greeks |
| Identities and properties (Hypothesis) | Put–call parity; no-arbitrage bounds; IV round trips; American ≥ European; CVaR ≥ VaR |
| Statistical checks with fixed seeds | Monte Carlo within 3 standard errors; confidence-interval coverage; simulated moments and correlations |
| Optimality | No random feasible portfolio beats the minimum-variance or maximum-Sharpe result; first-order conditions of the closed forms |
| Negative controls | A peeking strategy must fail the look-ahead checks; prices violating no-arbitrage must raise |
| Documentation | The README's Python examples run as doctests |

Monte Carlo and simulation tests use fixed seeds, so each either always passes or always fails; they cannot fail by chance in CI. Hypothesis property tests draw new random inputs on every run (200 per test, or 5,000 with `pytest --hypothesis-profile=thorough`). Tolerances are tied to the method's error: for example, the binomial test allows an error below `0.05 · S / steps` because CRR converges at rate 1/steps, and the gamma finite difference allows a relative error of 1e-4 because a second difference loses roughly half the digits to round-off.

## 7. Tooling

- **uv** manages the environment; `uv.lock` pins every dependency for every supported Python. The minimum versions in `pyproject.toml` are the oldest versions the lockfile actually exercises (the Python 3.11 job resolves NumPy 2.4 and SciPy 1.17).
- **mypy --strict** covers `src`, `tests` and `examples`. SciPy and scikit-learn ship no type information, so a single override ignores their missing stubs, and their results are wrapped in `float()` or `np.asarray()`. Tests that deliberately pass wrong types use `typing.cast` instead of `type: ignore`.
- **ruff** handles linting (including pydocstyle with Google-style docstrings) and formatting.
- **pytest** runs unit tests, Hypothesis properties and the README doctests, with warnings turned into errors so deprecations surface immediately. Coverage must stay at or above 90% in CI.
- **CI** (GitHub Actions) runs lint, format check, type check, tests with coverage, and CLI smoke tests on Python 3.11, 3.12 and 3.13, with read-only permissions and cancellation of superseded runs.

## 8. Questions worth being able to answer

- *Why does the binomial error oscillate instead of falling smoothly?* The strike falls at different positions between terminal nodes as the step count changes; the payoff kink is not resolved consistently.
- *Why Brent rather than Newton for implied volatility?* Guaranteed convergence on a bracket; Newton divides by vega, which can be nearly zero.
- *Why is the Monte Carlo standard error computed from pair averages?* The two halves of an antithetic pair are dependent; only the pair averages are i.i.d.
- *Why shrink the covariance matrix?* With few observations per parameter the sample covariance has extreme eigenvalues; optimizers invert it and amplify the noise. Shrinkage trades a little bias for much less variance.
- *Why does the maximum-Sharpe portfolio often concentrate in one or two assets?* It leans on the noisiest input, the expected returns; small estimation errors in `μ` move the weights a lot (see the README examples).
- *How do you know the backtest has no look-ahead?* The contract is explicit, the engine shifts weights by one day, the schedule uses first trading days, and a perturbation test shows that future prices cannot change past weights, with a cheating strategy as a negative control.
- *Why is turnover compared with drifted weights?* So that buy-and-hold is not charged for price moves.
- *What would you change for production use?* Execution lag and slippage, per-asset constraints, a QP solver, vectorized pricing, and robust estimates of expected returns.

See the README's *Limitations and roadmap* section for what is out of scope.

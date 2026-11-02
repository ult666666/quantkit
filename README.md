# quantkit

Options pricing, portfolio optimization, backtesting and VaR model validation in Python. Every model is checked against closed-form results, textbook values or an independent method.

[![CI](https://github.com/ult666666/quantkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ult666666/quantkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

> **Note:** quantkit is educational and research software. It is not investment advice, and nothing in this repository is a recommendation to buy or sell anything.

## Why

Quantitative finance code fails quietly: a sign slip in a Greek, an off-by-one day in a backtest or an optimizer that stops early all still return plausible-looking numbers. quantkit is a small, readable library built around verification. Each model is tested against something it did not produce itself: a textbook value, a closed-form solution, an independent implementation, or an identity that must hold. The backtester has an explicit, tested no-look-ahead contract.

## Features

- **Option pricing** (`quantkit.pricing`)
  - Black–Scholes–Merton prices and analytic Greeks (delta, gamma, vega, theta, rho) with a continuous dividend yield, on a validated, frozen `OptionSpec` dataclass.
  - Implied volatility with Brent's method, raising a typed `ArbitrageBoundError` when a price violates the no-arbitrage bounds.
  - Cox–Ross–Rubinstein binomial tree for European and American (early-exercise) options.
  - Monte Carlo under geometric Brownian motion with antithetic variates, standard errors and 95% confidence intervals, reproducible from a seed.
  - Arithmetic-average Asian options by Monte Carlo, with the geometric-average closed form as a control variate.
- **Portfolio construction** (`quantkit.portfolio`)
  - Simple and log returns; annualized sample covariance and Ledoit–Wolf shrinkage.
  - Minimum-variance, maximum-Sharpe and target-return portfolios with weight bounds (SciPy SLSQP), closed-form solutions for the unbounded case, and efficient frontiers.
  - Equal-risk-contribution (risk parity) and custom risk-budget portfolios.
  - Volatility, historical and Gaussian VaR and CVaR (expected shortfall).
- **Backtesting** (`quantkit.backtest`)
  - Daily vectorized engine: weights decided at the close of day *t* are held from day *t*+1, with drift-aware turnover and proportional costs in basis points.
  - Strategies: buy-and-hold, periodically rebalanced equal weight, moving-average crossover, time-series momentum and inverse volatility.
  - Metrics: CAGR, volatility, Sharpe, Sortino, maximum drawdown and its duration, Calmar, hit rate and turnover, plus a look-ahead checker for custom strategies.
- **VaR model validation** (`quantkit.var`)
  - Rolling one-day Value at Risk (VaR) and expected shortfall (ES, the mean loss beyond VaR) forecasts by historical simulation, a Gaussian model, EWMA (RiskMetrics, decay 0.94) and filtered historical simulation, each from a trailing window that never contains the return it is scored against.
  - Exception backtests: Kupiec's proportion-of-failures test and Christoffersen's independence and conditional coverage tests, with p-values, plus the exact probability that each test rejects a correct model at a given sample length.
  - The Basel traffic light: green, yellow and red zones and the capital plus factors for 250 days of 99% VaR.
  - Acerbi and Székely's Z2 test of expected shortfall, with p-values from a null distribution simulated from the forecasts themselves.
- **Data** (`quantkit.data`): a seeded correlated-GBM price simulator and a validating CSV loader (long or wide format).
- **CLI**: `quantkit price | iv | simulate | backtest | frontier | var-backtest`.
- Fully typed (`mypy --strict`, ships `py.typed`) and linted with ruff; the CI workflow targets Python 3.11–3.13.

## Quick start

quantkit is not on PyPI. Install the library and the `quantkit` CLI from the GitHub release (Python 3.11+):

```bash
pip install https://github.com/ult666666/quantkit/releases/download/v0.1.0/quantkit-0.1.0-py3-none-any.whl
```

Each release also attaches the source distribution and SHA-256 checksums. To work on the code, install it from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ult666666/quantkit.git
cd quantkit
uv sync                 # creates .venv with quantkit and the development tools
uv run quantkit --help
```

Or, with pip inside a virtual environment: `pip install .` (or `pip install ".[plot]"` for charts).

Price Hull's textbook option four ways. The outputs shown here are checked as doctests by the test suite:

```python
>>> from quantkit.pricing import (
...     OptionSpec, binomial_price, bsm_greeks, bsm_price, implied_vol, monte_carlo_price,
... )
>>> call = OptionSpec(spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5, kind="call")
>>> round(bsm_price(call), 4)
4.7594
>>> g = bsm_greeks(call)
>>> print(f"delta={g.delta:.4f} gamma={g.gamma:.4f} vega={g.vega:.4f} theta={g.theta:.4f}")
delta=0.7791 gamma=0.0500 vega=8.8134 theta=-4.5591
>>> round(implied_vol(4.76, spot=42, strike=40, rate=0.10, maturity=0.5, kind="call"), 4)
0.2001
>>> mc = monte_carlo_price(call, n_paths=200_000, seed=7)
>>> print(f"{mc.price:.4f} +/- {mc.std_error:.4f}, 95% CI [{mc.ci_low:.4f}, {mc.ci_high:.4f}]")
4.7547 +/- 0.0055, 95% CI [4.7440, 4.7654]
>>> put = OptionSpec(spot=50, strike=50, rate=0.10, vol=0.40, maturity=5 / 12, kind="put")
>>> round(binomial_price(put, steps=500, exercise="american"), 3)  # Hull reports 4.283
4.283

```

Build portfolios from (here, synthetic) prices and backtest a strategy:

```python
>>> from quantkit.data import simulate_gbm_prices
>>> from quantkit.portfolio import (
...     annualized_mean_returns, compute_returns, ledoit_wolf_covariance, max_sharpe,
...     risk_contributions, risk_parity,
... )
>>> prices = simulate_gbm_prices(  # synthetic: 5 years of correlated GBM
...     tickers=["AAA", "BBB", "CCC"], n_days=1260, drift=[0.04, 0.06, 0.08],
...     vol=[0.10, 0.15, 0.25], correlation=0.3, seed=1,
... )
>>> returns = compute_returns(prices)
>>> cov = ledoit_wolf_covariance(returns)
>>> weights = risk_parity(cov)
>>> weights.round(3).to_dict()
{'AAA': 0.47, 'BBB': 0.331, 'CCC': 0.199}
>>> risk_contributions(weights, cov).round(4).to_dict()
{'AAA': 0.3333, 'BBB': 0.3333, 'CCC': 0.3333}
>>> max_sharpe(annualized_mean_returns(returns), cov, risk_free_rate=0.02).round(3).to_dict()
{'AAA': 0.529, 'BBB': 0.471, 'CCC': 0.0}
>>> from quantkit.backtest import MovingAverageCrossover, run_backtest
>>> result = run_backtest(prices, MovingAverageCrossover(fast=50, slow=200), cost_bps=10)
>>> s = result.summary()
>>> print(f"CAGR {s['cagr']:.2%}, Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.1%}")
CAGR 0.36%, Sharpe 0.08, max drawdown 14.9%

```

The maximum-Sharpe portfolio leaves out `CCC` because its *estimated* mean return over this sample is negative (−10.9% a year), even though it was simulated with a +8% drift: five years of data pin down expected returns very poorly. That estimation error, not the optimizer, is the main weakness of mean–variance optimization in practice.

Forecast the equal-weight portfolio's one-day 99% VaR and ES each day from the previous 250 days, then backtest the forecasts:

```python
>>> from quantkit.var import (
...     backtest_var, basel_traffic_light, es_backtest, null_rejection_rates, rolling_var_es,
... )
>>> portfolio = returns.mean(axis=1)  # daily-rebalanced equal weights
>>> forecast = rolling_var_es(portfolio, method="fhs", level=0.99, window=250)
>>> print(f"{len(forecast)} forecasts, {forecast.exceptions.sum()} exceptions; "
...       f"latest VaR {forecast.var.iloc[-1]:.2%}, ES {forecast.es.iloc[-1]:.2%}")
759 forecasts, 8 exceptions; latest VaR 1.50%, ES 1.71%
>>> result = backtest_var(forecast.exceptions, level=0.99)
>>> print(f"LR_uc p={result.unconditional.p_value:.3f}, LR_ind p={result.independence.p_value:.3f}, "
...       f"LR_cc p={result.conditional.p_value:.3f}")
LR_uc p=0.882, LR_ind p=0.680, LR_cc p=0.908
>>> basel_traffic_light(int(forecast.last(250).exceptions.sum()))
TrafficLight(zone='green', n_exceptions=3, plus_factor=0.0)
>>> z2 = es_backtest(forecast, n_simulations=999, seed=0)
>>> print(f"Z2 {z2.statistic:.2f}, simulated p-value {z2.p_value:.3f}")
Z2 -0.13, simulated p-value 0.164
>>> rates = null_rejection_rates(250, level=0.99)  # how often a correct model fails at 5%
>>> print(f"{rates.unconditional:.3f} {rates.independence:.3f} {rates.conditional:.3f}")
0.095 0.014 0.011

```

FHS (filtered historical simulation) needs two 250-day windows of history, so the 1,259 returns give 759 forecasts. The last line is a warning about the tests themselves: with 2.5 exceptions expected in 250 days, the nominal 5% Kupiec test rejects a correct model 9.5% of the time, and the two Christoffersen tests about 1%.

## Usage

### Command line

Price an option with all three methods and print the Greeks:

```console
$ uv run quantkit price --spot 42 --strike 40 --rate 0.10 --vol 0.20 --maturity 0.5
European call: S=42 K=40 T=0.5 r=0.1 q=0 sigma=0.2

method                                 price   std. error
Black-Scholes-Merton                  4.7594
CRR binomial (500 steps)              4.7593
Monte Carlo (100,000 paths)           4.7659       0.0078

Greeks (Black-Scholes-Merton)
delta     0.7791
gamma     0.0500
vega      8.8134   per 1.00 change in volatility
theta    -4.5591   per year
rho      13.9820   per 1.00 change in the rate
```

American options use the binomial tree:

```console
$ uv run quantkit price --spot 50 --strike 50 --rate 0.10 --vol 0.40 --maturity 0.4167 --kind put --american
American put: S=50 K=50 T=0.4167 r=0.1 q=0 sigma=0.4

method                                     price
CRR binomial, American (500 steps)        4.2832
CRR binomial, European (500 steps)        4.0736
Black-Scholes-Merton, European            4.0761
early-exercise premium                    0.2096
```

The premium compares two trees with the same number of steps, so the tree's own discretization error cancels out.

Implied volatility, and the error for a price that no volatility can produce:

```console
$ uv run quantkit iv --price 4.76 --spot 42 --strike 40 --rate 0.10 --maturity 0.5
implied volatility: 0.200066 (20.01%)
$ uv run quantkit iv --price 1.50 --spot 42 --strike 40 --rate 0.10 --maturity 0.5
quantkit: error: price 1.5 is outside the no-arbitrage bounds (3.95082, 42); no volatility reproduces it
```

Backtests and frontiers read a CSV of daily closes, either long (`date,ticker,close`) or wide (`date` plus one column per ticker). To try them without real data, generate a **synthetic** file (five correlated GBM assets `AAA`–`EEE`, seed 42):

```console
$ uv run quantkit simulate --output prices.csv
wrote 1260 days x 5 tickers of synthetic GBM prices to prices.csv
$ uv run quantkit backtest prices.csv --strategy ma-crossover --cost-bps 10
5 assets, 1260 days (2020-01-01 to 2024-10-29), costs 10 bps of traded notional

                           ma-crossover (50/200) buy-and-hold (equal weight)
total return                               16.7%                        6.1%
CAGR                                       3.14%                       1.18%
volatility                                 8.22%                      11.92%
Sharpe                                      0.42                        0.16
Sortino                                     0.61                        0.22
max drawdown                               14.2%                       27.3%
max drawdown length (days)                   446                        1101
Calmar                                      0.22                        0.04
hit rate                                   50.9%                       51.8%
annual turnover                            1.31x                       0.20x
```

The crossover "wins" on this one synthetic history, but GBM has no trends to follow; the backtest example below shows why a single backtest proves little. Other strategies: `--strategy buy-and-hold | equal-weight | momentum | inverse-vol` (see `quantkit backtest --help`).

```console
$ uv run quantkit frontier prices.csv --shrinkage --risk-free 0.02
1259 daily returns of 5 assets; expected returns = annualized historical means, Ledoit-Wolf covariance; weights in [0, 1]; risk-free rate 2.00%

             return volatility Sharpe   AAA   BBB    CCC   DDD  EEE
1              0.5%       7.6%  -0.20 75.7% 19.1%   5.2%  0.0% 0.0%
2              1.5%       7.9%  -0.06 60.7% 24.1%  14.5%  0.6% 0.0%
3              2.5%       8.6%   0.06 45.9% 28.0%  22.6%  3.4% 0.0%
4              3.5%       9.6%   0.15 31.1% 32.0%  30.7%  6.2% 0.0%
5              4.5%      10.9%   0.23 16.3% 35.9%  38.8%  9.0% 0.0%
6              5.5%      12.3%   0.28  1.5% 39.8%  46.8% 11.8% 0.0%
7              6.4%      14.2%   0.31  0.0% 18.3%  65.8% 16.0% 0.0%
8              7.4%      17.6%   0.31  0.0%  0.0% 100.0%  0.0% 0.0%
min variance   0.5%       7.6%  -0.20 75.7% 19.1%   5.2%  0.0% 0.0%
max Sharpe     7.2%      16.2%   0.32  0.0%  0.0%  81.7% 18.3% 0.0%
```

Add `--plot frontier.png` to also save a chart (needs the `plot` extra), and, for example, `--min-weight -0.2 --max-weight 0.6` to change the weight bounds.

Backtest rolling VaR and ES forecasts of the equal-weight portfolio (or one column, with `--ticker`):

```console
$ uv run quantkit var-backtest prices.csv --method ewma --level 0.99
equal-weight portfolio of 5 assets, rebalanced daily: 1259 daily returns, 2020-01-02 to 2024-10-29
99% one-day VaR and ES, 250-day window, EWMA decay 0.94
backtest: 1009 days, 2020-12-17 to 2024-10-29

                                         EWMA
exceptions                                 14
expected exceptions                      10.1
exception rate                          1.39%
Kupiec LR_uc p-value                    0.243
Christoffersen LR_ind p-value           0.185
conditional coverage LR_cc p-value      0.209
Basel zone, last 250 days           green (3)
Acerbi-Szekely Z2                       -0.42
Z2 simulated p-value                    0.102

LR p-values use chi-squared limits. Over 1009 days at 99%, a correct model fails the 5% tests
with probability 5.5% (LR_uc), 1.7% (LR_ind) and 2.7% (LR_cc).
Z2 p-values compare with 9,999 paths simulated from each method's own forecasts (seed 0).
```

Without `--method` it compares all four methods on the days they all cover. An exception is a day whose loss exceeded that day's VaR forecast; at 99% a correct model expects one in a hundred days, independent of each other. The Basel zone counts the exceptions of the last 250 days, as bank supervisors do. Z2 is negative when losses beyond VaR are larger than the ES forecasts promised.

### Examples and figures

The scripts in [`examples/`](examples) regenerate the figures below (`uv run python examples/<name>.py`). The walk-forward example writes CSVs instead of a figure.

**Binomial convergence** ([`binomial_convergence.py`](examples/binomial_convergence.py)). CRR tree prices for Hull's call approach the Black–Scholes–Merton value with an error that shrinks like 1/steps, oscillating as tree nodes move relative to the strike.

![CRR binomial price converging to Black-Scholes-Merton](docs/images/binomial_convergence.png)

**Efficient frontier** ([`efficient_frontier.py`](examples/efficient_frontier.py)), on **synthetic** data: the long-only frontier from SLSQP, the unconstrained frontier from its closed form (always to the left, since removing constraints can only lower risk), the minimum-variance and maximum-Sharpe portfolios and the capital market line. The means are estimated from five years of simulated prices, so `EEE`, simulated with an 11% drift, shows a negative estimated return.

![Efficient frontier on synthetic data](docs/images/efficient_frontier.png)

**Backtest** ([`backtest_trend.py`](examples/backtest_trend.py)), on **synthetic** data: a 50/200-day moving-average crossover against equal-weight buy-and-hold over ten years, with 10 bps costs. On this history (seed 42) the crossover ends ahead. The script then repeats the comparison on 100 independent synthetic histories:

```text
over 100 independent synthetic histories, CAGR(MA) - CAGR(buy-and-hold): mean -3.41%, standard error 0.29%; MA crossover came out ahead in 14 of 100
```

Seed 42 happens to be one of the 14. With no trends in the data, the crossover mostly sits in cash while the assets drift upward.

![Equity curves and drawdowns on synthetic data](docs/images/backtest_equity.png)

**VaR backtest** ([`var_backtest.py`](examples/var_backtest.py)), on **synthetic** data: GARCH(1,1) returns, whose volatility jumps after large moves, with fat-tailed Student-t shocks. Historical simulation weights a year of history equally, so its VaR (red) reacts late to the volatility burst and its exceptions cluster; filtered historical simulation (blue) rescales history by the current EWMA volatility. Over 1,250 days at 99%:

```text
                            historical    Gaussian       EWMA        FHS
exceptions                          20          24         24         16
expected                          12.5        12.5       12.5       12.5
Kupiec p                        0.0499      0.0037     0.0037     0.3403
independence p                  0.0389      0.0833     0.4815     0.5193
cond. coverage p                0.0172      0.0033     0.0115     0.5142
Basel zone (last 250 days)  yellow (5)  yellow (7)  green (2)  green (2)
Z2                               -0.79       -1.25      -1.14      -0.33
Z2 p                            0.0014      0.0002     0.0005     0.0484
```

All three exception tests reject historical simulation, and the independence test shows why: its exceptions bunch up in the volatile period. EWMA follows the volatility (independence p = 0.48) but keeps normal tails, so it has as many exceptions as the Gaussian model, 24 against 12.5 expected. FHS passes the three exception tests; Z2 still rejects it, narrowly, at 5%.

![VaR forecasts, exceptions and cumulative exception counts on synthetic data](docs/images/var_backtest.png)

**Size and power of the backtests** ([`var_size_power.py`](examples/var_size_power.py)), on **synthetic** data. Each row simulates 1,000 independent paths, forecasts with a 250-day window and runs the 5% tests on the last 250 or 1,000 days (`uv run python examples/var_size_power.py`, about 40 seconds):

```text
Share of 1,000 SYNTHETIC paths rejected by each 5% test (Monte Carlo standard error at most 1.6%)

                                                  LR_uc   LR_ind    LR_cc       Z2 exceptions
size: EWMA forecasts on returns from the EWMA model, 99%
  250 days, measured                              10.5%     1.8%     1.0%     5.1%      1.00%
  250 days, exact                                  9.5%     1.4%     1.1%     5.0%
  1,000 days, measured                             4.2%     2.2%     2.2%     4.4%      1.00%
  1,000 days, exact                                5.5%     1.8%     2.7%     5.0%
power: Gaussian forecasts, 99%
  normal returns, 250 days                         8.1%     1.3%     0.8%     5.3%      1.02%
  Student-t(4) returns, 250 days                  13.0%     1.4%     6.8%    42.2%      1.65%
  normal returns, 1,000 days                       2.2%     1.2%     0.9%     3.9%      1.05%
  Student-t(4) returns, 1,000 days                46.9%     1.9%    31.3%    87.3%      1.63%
power: Gaussian forecasts, 97.5%
  normal returns, 1,000 days                       1.2%     2.3%     2.1%     4.0%      2.57%
  Student-t(4) returns, 1,000 days                 3.4%     2.0%     3.2%    43.3%      2.70%
```

- **Size.** The paths are drawn from the EWMA model itself, so the forecasts are exactly right and every rejection is a false alarm. The measured rates agree with the exact ones from `null_rejection_rates` within the Monte Carlo error, and they are not 5%: with 2.5 expected exceptions in 250 days, zero exceptions (8.1% likely) already fails Kupiec's test, and the independence test rarely has a pair of exceptions to work with.
- **Power.** Student-t returns with 4 degrees of freedom have the same variance as the normal ones but fatter tails, so Gaussian VaR is too low. Exception counts catch this in only 13% of 250-day samples. Z2, which also weighs how large the tail losses are, catches it in 42%, and in 87% of 1,000-day samples. At 97.5% (the level of the Basel FRTB expected shortfall) the Gaussian VaR happens to have about the right number of exceptions, so the count-based tests see nothing (3.4%) while Z2 rejects 43% of samples.

**Walk-forward portfolio construction** ([`walk_forward_risk_parity.py`](examples/walk_forward_risk_parity.py)), on **synthetic** data. This example connects trailing-return estimation, risk-parity allocation and backtesting. Each monthly allocation uses only closes before the rebalance date. A monthly equal-weight benchmark shares the same cash warmup, dates and costs.

```bash
uv run python examples/walk_forward_risk_parity.py
```

The command exports prices, both target tables, estimation windows, net wealth and a common-period summary to `walk-forward-output/`. The [walkthrough](docs/WALK_FORWARD.md) explains the timing, CSV fields, reproducible inputs and limitations.

## How it works

```mermaid
flowchart LR
    CSV["CSV file or synthetic GBM<br/>quantkit.data"] --> PX["wide price table<br/>(dates x tickers)"]
    PX --> PF["quantkit.portfolio<br/>returns, covariance, optimizers, VaR"]
    PF --> VR["quantkit.var<br/>rolling VaR/ES forecasts, exception tests,<br/>Basel traffic light, ES backtest"]
    PX --> ST["strategy<br/>target weights per close"]
    ST --> EN["backtest engine<br/>weights held from t+1, costs"]
    EN --> MT["metrics<br/>CAGR, Sharpe, drawdown, ..."]
    OS["OptionSpec"] --> PR["quantkit.pricing<br/>BSM, IV, CRR tree, Monte Carlo, Asian"]
```

- **Pricing.** Every pricer takes the same frozen, validated `OptionSpec`, so one contract can be priced by independent methods and the results compared. The normal CDF is computed with `erfc`, which stays accurate deep in the tails. Implied volatility checks the model-free bounds first and then runs Brent's method, which cannot diverge the way Newton's method can when vega is tiny. The CRR tree rolls values back one vectorized level at a time; Monte Carlo samples the terminal price exactly, so its only error is sampling error.
- **Portfolio.** Optimizers use SLSQP so that weight bounds are easy to add, and the unbounded problems also have closed forms that serve as a reference in the tests. The maximum-Sharpe problem is solved through a change of variables that turns it into a convex quadratic program. Risk parity minimizes a convex log-barrier objective (Maillard, Roncalli & Teïletche 2010) by cyclical coordinate descent.
- **Backtest.** The timing contract is explicit:

  | At the close of day *t* | During day *t*+1 |
  | --- | --- |
  | The strategy sees prices for days 0..*t* and returns target weights (or NaN for "no trade"). The portfolio trades to those weights at that close and pays `cost_bps` on the traded notional. | The new weights earn day *t*+1's return. Between trades, weights drift with prices. |

  The engine is vectorized: between trades the share counts are constant, so each day's weights follow from the weights at the last trade and the price growth since then.

- **VaR model validation.** Every forecaster is a location-scale model `r[t] = mu[t] + sigma[t] e[t]` fitted to the window before day `t`, with normal or resampled innovations `e`; VaR, ES and the simulated null of the ES backtest all come from that one description. The likelihood-ratio tests use `0 log 0 = 0`, so they stay finite with no exceptions or no consecutive pairs, and a dynamic program over the exception sequence gives their exact size.

[docs/DESIGN.md](docs/DESIGN.md) explains the architecture, the main design decisions and their trade-offs in more detail.

## Correctness

| Component | Checked against |
| --- | --- |
| BSM prices | Hull's textbook values (call ≈ 4.76, put ≈ 0.81); put–call parity and no-arbitrage bounds over random inputs (Hypothesis) |
| BSM Greeks | Hull's Greek-letters example; central finite differences; call–put Greek identities |
| Implied volatility | Round trips through BSM over random inputs; typed errors for prices outside the bounds |
| Binomial tree | Convergence to BSM at rate 1/steps; Hull's American put values for 5, 30, 50, 100 and 500 steps; a one-step tree by hand; American put ≥ European put; American call = European call when q = 0 |
| Monte Carlo | Within 3 standard errors of BSM; 95% intervals contain the BSM price in 92–98% of 400 seeded runs; antithetic variates lower the standard error |
| Asian options | Geometric formula equals BSM with one fixing and matches an independent simulation; the control-variate estimate agrees with plain Monte Carlo at over 10x lower standard error |
| Ledoit–Wolf | scikit-learn's `ledoit_wolf`, including more assets than observations |
| Mean–variance | Closed-form minimum-variance, tangency and target-return portfolios (unbounded case); first-order conditions; no random feasible portfolio beats the optimizer; budget and bounds respected |
| Risk parity | Equal risk contributions; inverse-volatility closed forms (uncorrelated or two assets); volatility between minimum-variance and 1/N |
| VaR / CVaR | Hand-computed tails; Gaussian closed forms; historical converges to parametric on normal data |
| Rolling VaR / ES forecasts | Every window agrees with the single-sample `historical_var` and `parametric_var`; the RiskMetrics recursion; a Hull–White volatility-rescaling loop for FHS; **no look-ahead**: changing the return a forecast is scored against (and all later ones) leaves the forecast unchanged, and a peeking forecaster fails the same check; exception probabilities on i.i.d. data match closed forms (`m / (window + 1)` for historical simulation, a Student-t prediction interval for the Gaussian model) |
| Exception tests | Hand-computed Kupiec and Christoffersen statistics; SciPy's binomial log-likelihoods and `chi2` p-values; `LR_cc = LR_uc + LR_ind`; exact sizes against brute-force enumeration of all 2^T exception sequences; simulated sizes on returns drawn from the forecast model itself |
| Basel traffic light | The zones, plus factors and cumulative probabilities of the 1996 Basel table, and the 95% / 99.99% binomial rule behind them |
| ES backtest (Z2) | A hand calculation; mean zero when returns follow the forecasts; a null distribution that does not depend on the volatility path; power against fat tails |
| Backtester | An independent day-by-day reference loop; buy-and-hold equals the asset return; costs reduce wealth by exactly the computed amount; a hand-computed two-day example; **no look-ahead**: perturbing prices after day *t* leaves every weight in effect through day *t*+1 unchanged, and a deliberately cheating strategy fails the same check |
| Metrics | Hand-computed series (drawdown and its duration, CAGR, Sharpe, Sortino, Calmar, hit rate) |
| Data | Simulated means, volatilities and correlations match their targets; CSV round trips and specific error messages |

Monte Carlo and other statistical tests use fixed seeds, so they cannot fail by chance; the Hypothesis property tests draw fresh random inputs on every run (200 per test, or 5,000 with `pytest --hypothesis-profile=thorough`). The CI workflow runs ruff, `mypy --strict`, the tests with a 90% coverage floor, the README doctests, CLI smoke tests and the example scripts on Python 3.11, 3.12 and 3.13.

## Development

```bash
uv sync                                   # runtime + dev dependencies (pytest, hypothesis, mypy, ruff, ...)
uv run ruff check .                       # lint
uv run ruff format --check .              # formatting
uv run mypy                               # strict type checking of src, tests and examples
uv run pytest --cov=quantkit              # unit tests, Hypothesis properties and README doctests
uv run python examples/backtest_trend.py  # regenerate a figure in docs/images
```

## Limitations and roadmap

**Models.** Black–Scholes–Merton and GBM assume constant volatility and rates, lognormal prices and continuous dividends: no volatility smile, jumps or discrete dividends. Monte Carlo covers European and Asian payoffs only (no Longstaff–Schwartz for American options). The CRR tree converges slowly and with oscillation; smoother trees such as Leisen–Reimer are not implemented. Implied volatilities outside 0.0001%–1000% are reported as errors. The pricing API is scalar; large option chains would call for a vectorized interface.

**Portfolio.** Expected returns are plain historical means, which are very noisy. The same weight bounds apply to every asset, with no per-asset, sector or turnover constraints. SLSQP is a general-purpose solver that works well for tens of assets; a dedicated QP solver would scale better. Risk parity is long-only. VaR is one-period and in return units.

**VaR models and backtests.** Forecasts cover one day and one return series, in return units: there is no multi-day scaling, no P&L from positions and no revaluation of options. The EWMA decay is fixed rather than estimated (no GARCH fit), and FHS filters with that EWMA volatility. The likelihood-ratio p-values are asymptotic; `null_rejection_rates` reports how far the true size is from the nominal one, but the tests do not yet offer exact or Monte Carlo p-values. The independence test only looks at consecutive days. The traffic light implements the 1996 table for 250 days of 99% VaR, not the 97.5% zones or the P&L attribution test of the Basel FRTB standards. Z2's null distribution is simulated from each method's own forecasts, so it tests the forecast distribution's tail as a whole and needs positive ES forecasts.

**Backtesting.** Daily close-to-close only: trades fill at the same close that produced the signal, and the only friction is a proportional cost (no slippage, market impact, borrowing or financing costs). Idle cash earns nothing. Prices must be complete (no missing values and no assets entering mid-sample) and should already be adjusted for splits and dividends. The no-look-ahead guarantee covers the engine and the built-in strategies; custom strategies can be checked with `check_no_lookahead`.

**Data and verification.** There is deliberately no data downloader: bring your own CSV with ISO dates. CI runs lint, strict type checks and the full suite with README doctests on Python 3.11, 3.12 and 3.13.

**Roadmap ideas:** per-asset and linear constraints, a vectorized BSM interface, Longstaff–Schwartz for American options, Black–Litterman expected returns, an engine option to fill trades after the signal close, Monte Carlo p-values and duration-based tests (Christoffersen & Pelletier 2004) for VaR exceptions, and GARCH-filtered historical simulation. The walk-forward example lags its estimation window by one close, but its trades still fill at the signal close.

## References

- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson (chapters on the Black–Scholes–Merton model, the Greek letters, binomial trees and numerical procedures).
- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183.
- Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229–263.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Kemna, A. G. Z. & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113–129.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Merton, R. C. (1972). An analytic derivation of the efficient portfolio frontier. *Journal of Financial and Quantitative Analysis*, 7(4), 1851–1872.
- Cornuéjols, G. & Tütüncü, R. (2007). *Optimization Methods in Finance*. Cambridge University Press.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Maillard, S., Roncalli, T. & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60–70.
- Griveau-Billion, T., Richard, J.-C. & Roncalli, T. (2013). A fast algorithm for computing high-dimensional risk parity portfolios. arXiv:1311.4057.
- Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th edition.
- Hull, J. & White, A. (1998). Incorporating volatility updating into the historical simulation method for value-at-risk. *Journal of Risk*, 1(1), 5–19.
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Christoffersen, P. & Pelletier, D. (2004). Backtesting value-at-risk: A duration-based approach. *Journal of Financial Econometrics*, 2(1), 84–108.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements*. [bis.org/publ/bcbs22.htm](https://www.bis.org/publ/bcbs22.htm)
- Acerbi, C. & Székely, B. (2014). Backtesting expected shortfall. *Risk*, December 2014.
- Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746–762.
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd edition. McGraw-Hill (chapter on backtesting VaR).
- Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press (Monte Carlo tests).

## License

[MIT](LICENSE) © 2026 Charles Dai

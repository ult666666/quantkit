# quantkit

A Python library for option pricing, portfolio construction, backtesting and VaR model
validation. Each model is tested against something it did not produce itself: a
textbook value, a closed form, an independent implementation or an identity that must
hold.

[![CI](https://github.com/ult666666/quantkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ult666666/quantkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

## Install

quantkit is not on PyPI. Install the library and the `quantkit` command from GitHub
(Python 3.11 or newer):

```bash
pip install "https://github.com/ult666666/quantkit/releases/download/v0.2.0/quantkit-0.2.0-py3-none-any.whl"
```

To work on the code, clone the repository and run `uv sync` with [uv](https://docs.astral.sh/uv/),
or `pip install ".[plot]"` in a virtual environment for the charts.

## Example

Hull's textbook call: its price, Greeks and implied volatility, a Monte Carlo check,
then an American put on a binomial tree. CI runs this block, and the two below, as
doctests:

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

## API sketch

| Module | Main entry points |
| --- | --- |
| `quantkit.pricing` | `OptionSpec`; `bsm_price`, `bsm_greeks`, `implied_vol`, `binomial_price` (European or American), `monte_carlo_price`, `asian_monte_carlo_price` |
| `quantkit.portfolio` | `compute_returns`, `ledoit_wolf_covariance`; `min_variance`, `max_sharpe`, `efficient_return`, `efficient_frontier`, `risk_parity`; historical and parametric VaR and CVaR |
| `quantkit.backtest` | `run_backtest` with `BuyAndHold`, `EqualWeight`, `MovingAverageCrossover`, `TimeSeriesMomentum` or `InverseVolatility`; performance metrics; `check_no_lookahead` for your own strategies |
| `quantkit.var` | `rolling_var_es` (historical, Gaussian, EWMA, filtered historical simulation); `backtest_var` (Kupiec and Christoffersen tests); `basel_traffic_light`; `es_backtest` (Acerbi and Székely's Z2) |
| `quantkit.data` | `simulate_gbm_prices` (seeded, correlated) and `load_prices_csv` (long or wide format) |
| `quantkit` command | `price`, `iv`, `simulate`, `backtest`, `frontier` and `var-backtest`; see [docs/CLI.md](docs/CLI.md) |

The package is typed (`mypy --strict`, with `py.typed` shipped). Public functions
validate their inputs and raise typed errors whose messages name the problem.

Portfolios and a backtest, on five years of synthetic prices:

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

The maximum-Sharpe portfolio leaves out `CCC`. It was simulated with a +8% drift, yet
its estimated mean return over these five years is −10.9% a year. Five years of data
pin down expected returns poorly, and that estimation error, more than the optimizer,
is the weak point of mean–variance optimization.

Rolling VaR and expected shortfall (ES, the mean loss beyond VaR) forecasts for the
equal-weight portfolio, each made from a trailing window only, then backtested:

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

FHS (filtered historical simulation) needs two 250-day windows of history before its
first forecast. That is why the 1,259 returns give 759 forecasts. The last line is
about the tests themselves: with so few exceptions expected, a correct model fails the
nominal 5% Kupiec test 9.5% of the time.

## How it works

Every pricer takes the same frozen, validated `OptionSpec`, so one contract can be
priced by independent methods and the results compared. Implied volatility checks the
no-arbitrage bounds first, then runs Brent's method, which cannot diverge the way
Newton's method can when vega is tiny. The optimizers use SciPy's SLSQP so that weight
bounds are easy to add, and the unbounded problems also have closed forms that the
tests use as a reference.

The backtester's timing contract is explicit:

| At the close of day *t* | During day *t*+1 |
| --- | --- |
| The strategy sees prices for days 0..*t* and returns target weights (or NaN for "no trade"). The portfolio trades to those weights at that close and pays `cost_bps` on the traded notional. | The new weights earn day *t*+1's return. Between trades, weights drift with prices. |

A perturbation test changes every price after day *t* and requires the weights in
effect through day *t*+1 to stay the same; a strategy that peeks fails it. The VaR
forecasts pass the same kind of test. [docs/DESIGN.md](docs/DESIGN.md) covers the
architecture, the trade-offs and the full list of checks per component.

More:

- [docs/CLI.md](docs/CLI.md): the command line, with sample output.
- [docs/EXAMPLES.md](docs/EXAMPLES.md): figures and studies, including the size and power
  of the VaR backtests on 1,000 simulated paths.
- [docs/WALK_FORWARD.md](docs/WALK_FORWARD.md): a walk-forward risk-parity example that
  exports its inputs and results as CSV files.

## Development

```bash
uv sync                                   # runtime + dev dependencies (pytest, hypothesis, mypy, ruff, ...)
uv run ruff check .                       # lint
uv run ruff format --check .              # formatting
uv run mypy                               # strict type checking of src, tests and examples
uv run pytest --cov=quantkit              # unit tests, Hypothesis properties and README doctests
uv run python examples/backtest_trend.py  # regenerate a figure in docs/images
```

CI runs these on Python 3.11, 3.12 and 3.13, plus the README doctests against an
installed wheel, the CLI and the example scripts.

## Limits

- This is educational software, not investment advice, and there is no data
  downloader: bring a CSV of daily closes.
- Black–Scholes–Merton and GBM assume constant volatility and rates, so there is no
  smile, no jumps and no discrete dividends.
- Expected returns are plain historical means, and one set of weight bounds applies to
  every asset.
- Backtests are daily close to close, fill at the signal's close and charge only a
  proportional cost.
- VaR forecasts cover one day of one return series, and the likelihood-ratio p-values
  are asymptotic.

The full list, with the roadmap, is in [docs/DESIGN.md](docs/DESIGN.md#10-limitations-and-roadmap).

Built with AI coding assistants. The tests and CI checks described above verify its
behaviour.

## License

[MIT](LICENSE) © 2026 Charles Dai

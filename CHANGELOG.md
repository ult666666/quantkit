# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-10-23

First release.

### Added

- `quantkit.pricing`: Black–Scholes–Merton prices and analytic Greeks with a
  continuous dividend yield; implied volatility via Brent's method with typed
  no-arbitrage errors; Cox–Ross–Rubinstein trees for European and American
  options; Monte Carlo under GBM with antithetic variates and confidence
  intervals; geometric-average Asian options in closed form and
  arithmetic-average Asian options by Monte Carlo with a control variate.
- `quantkit.portfolio`: simple and log returns; sample and Ledoit–Wolf
  covariance; minimum-variance, maximum-Sharpe and target-return portfolios with
  weight bounds (SLSQP) and closed forms for the unbounded case; efficient
  frontiers; risk parity and risk budgeting; volatility, historical and Gaussian
  VaR and CVaR.
- `quantkit.backtest`: a daily vectorized backtester with an explicit
  no-look-ahead contract, drift-aware turnover and proportional costs; five
  example strategies; performance metrics; a look-ahead checker.
- `quantkit.data`: a seeded correlated-GBM price simulator and a validating CSV
  loader for long and wide layouts.
- `quantkit` command-line interface with `price`, `iv`, `simulate`, `backtest`
  and `frontier` commands.
- Example scripts that generate the README figures, design notes in
  `docs/DESIGN.md`, and a GitHub Actions workflow for Python 3.11–3.13.

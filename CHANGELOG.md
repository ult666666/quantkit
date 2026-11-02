# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `quantkit.var`, VaR model validation:
  - `rolling_var_es`: rolling one-day VaR and expected shortfall forecasts by
    historical simulation, a Gaussian model, EWMA (RiskMetrics, decay 0.94) and
    filtered historical simulation. Each forecast uses only the window before
    the day it forecasts; `RiskForecast.simulate` draws from the forecast
    distributions.
  - `backtest_var`, `kupiec_test`, `christoffersen_test` and
    `conditional_coverage_test`: exception counts and likelihood-ratio tests
    with asymptotic p-values, finite with no exceptions, only exceptions or no
    consecutive pairs. `null_rejection_rates` computes the exact probability
    that each test rejects a correct model at a given sample length.
  - `basel_traffic_light`: the Basel Committee's 1996 zones and capital plus
    factors for 250 days of 99% VaR.
  - `acerbi_szekely_z2` and `es_backtest`: Acerbi and Székely's Z2 test of
    expected shortfall with a simulated null distribution.
- `quantkit var-backtest` command, and the `examples/var_backtest.py` (figure)
  and `examples/var_size_power.py` (size and power study) scripts, which CI
  now runs.

- `examples/walk_forward_risk_parity.py`: a walk-forward example that rebuilds a
  risk-parity portfolio each month from trailing returns, compares it with an
  equal-weight benchmark and exports the results as CSVs. The walkthrough is in
  `docs/WALK_FORWARD.md`.
- A release workflow that builds the wheel and sdist from a version tag and
  attaches them, with SHA-256 checksums, to the GitHub release.

### Changed

- The tail size `ceil(n * (1 - level))` of historical VaR moved to a shared
  helper, so single-sample and rolling historical VaR always use the same rule.
  Results are unchanged.

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

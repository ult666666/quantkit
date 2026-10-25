# Walk-forward risk-parity example

Add a runnable educational example joining existing portfolio estimation and backtesting APIs. Generate deterministic fictional AAA/BBB/CCC prices, estimate a trailing covariance at each monthly rebalance, construct risk-parity targets, and compare a monthly equal-weight benchmark on identical dates, initial capital, warmup and proportional costs. No statistical or optimization primitive changes.

At the close of date t, the estimator uses exactly `lookback` simple returns from `lookback + 1` closes ending at t−1. The engine trades at close t and holds those targets during t+1. The extra one-close lag makes the information boundary explicit. Both strategies remain in zero-interest cash until the first monthly rebalance with sufficient history, then drift between rebalances. Long-only targets sum to one.

Export fictional prices, both daily target tables (blank rows mean no trade), daily wealth, evaluation-period summary and estimation-window dates as CSVs. Summaries start on the shared first trade date, including its entry cost; full wealth includes the shared cash warmup. Defaults: 1,260 business days, 252-return window, seed 42, 10 bps costs, initial wealth 1. No plots or extra dependencies.

Approaches considered: a new strategy class would expand the library API; a notebook adds tooling; a standalone example is selected to keep composition and its assumptions inspectable. Tests cover future/current-close shocks, feasible allocations, shared cash warmup, fixed-target cost monotonicity, auditable estimation windows and deterministic actual CLI exports. Document GBM assumptions, synthetic data, same-close fills, covariance-estimation risk and the absence of any performance or investment claim.

# Walk-forward portfolio construction

This runnable example connects quantkit's existing covariance estimator,
risk-parity optimizer and backtest engine. It builds a new allocation each
month using a trailing history, then evaluates that allocation on subsequent
returns. All prices describe fictional assets; no market data is downloaded.

After `uv sync`, run from the repository root:

```bash
uv run python examples/walk_forward_risk_parity.py
```

The default run writes six CSVs to `walk-forward-output/`. To choose a location
or run a shorter example:

```bash
uv run python examples/walk_forward_risk_parity.py --days 400 --lookback 60 --seed 7 --cost-bps 10 --output-dir /tmp/quantkit-walk-forward
```

The CLI prints the simulation inputs and written paths. It exits 1 on an
invalid window, insufficient history, invalid costs or an I/O error. No plotting
extra, database, credentials or network connection is needed after installation.

## The scenario

Three fictional assets, `AAA`, `BBB` and `CCC`, have different simulated
volatilities. The workflow asks how to turn an estimated covariance matrix into
a sequence of portfolio allocations, while keeping the estimation and
subsequent evaluation periods separate.

| Input | Default |
| --- | --- |
| Prices | 1,260 business-day closes from 2020-01-01, seed 42 |
| Annual GBM drift | 4%, 6%, 8% for AAA, BBB, CCC |
| Annual GBM volatility | 10%, 18%, 28% |
| Pairwise Brownian correlation | 0.3 |
| Estimation | 252 prior simple returns, Ledoit–Wolf covariance |
| Risk parity | Long-only weights with equal estimated variance contributions |
| Benchmark | Equal capital weights, rebalanced on the same dates |
| Rebalancing | First business-day close of each month with sufficient history |
| Starting wealth | 1 in each portfolio |
| Trading costs | 10 basis points of traded notional, including initial entry |
| Idle cash | Zero interest |

These are simulation settings, not asset forecasts or recommended allocations.
Equal estimated risk contribution does not mean equal capital weights or equal
realized risk contributions in the following month.

## The information boundary

For a rebalance at price-table position **t**:

1. Read `lookback + 1` closes ending at **t−1**. They produce exactly
   `lookback` simple returns; neither close t nor later data enters the
   covariance estimate.
2. Estimate the covariance with `ledoit_wolf_covariance` and construct targets
   with `risk_parity`. Set equal-weight targets on that same date.
3. The existing `backtest_weights` engine trades at close **t**, charges costs
   there, and holds those weights during **t+1**. Between trades, share counts
   stay fixed and portfolio weights drift.

The one-close estimation lag is deliberate: the target can be prepared before
the rebalance close. The fill still uses that close's price under the engine's
existing convention. There is no attempt to model an actual closing auction.

Both portfolios start in cash and wait for the **same** first eligible monthly
rebalance. They do not invest immediately when the rolling window fills partway
through a month. In the default run, the first trade is **2021-01-01**; its 252
returns come from closes **2020-01-14 through 2020-12-31**. Exposure begins on the
following synthetic business day. Each portfolio pays the same 0.1% initial
entry cost, reducing initial wealth from 1 to 0.999 at the first trade.

## Inspect the CSVs

| File | Contents |
| --- | --- |
| `prices.csv` | The exact fictional daily closes used by both portfolios |
| `risk_parity_targets.csv` | Date-indexed target weight per asset; blank rows mean no trade |
| `equal_weight_targets.csv` | Benchmark targets with the identical rebalance calendar |
| `estimation_windows.csv` | Each rebalance date, first/last historical close, and return count |
| `wealth.csv` | Both net wealth curves on the full date index, including cash warmup |
| `summary.csv` | Existing performance metrics for the shared evaluation period |

Targets are decisions made at a close, **not** weights held during that same
day. Blank rows keep existing shares unchanged; during the initial warmup they
leave the portfolio in cash. A zero target would instead instruct the engine
to sell a position. The target tables include all dates so this distinction is
inspectable.

The summary starts at the first trade close and ends at the final close. It
includes initial entry costs and excludes the earlier all-cash warmup from
annualization. Both strategies use exactly that period, 252 periods per year
and a zero risk-free rate. The full wealth exports retain the warmup; their
final values minus 1 match the summary's total returns. Ratio metrics can be
blank when their denominators are zero, following the library's NaN convention.

Inspecting these metrics demonstrates accounting and timing. A difference
between the curves on one synthetic history is not evidence of investment
performance, an alpha claim or a recommendation to trade.

## Assumptions and limits

The price simulator uses correlated geometric Brownian motion with constant
drift and volatility. Its business dates omit weekends but include exchange
holidays. It has no fat tails, volatility clustering, changing asset universes
or missing prices. The estimated covariance can still differ from subsequent
realized covariance; shrinkage and risk parity do not remove estimation risk.

The engine uses daily close-to-close returns, same-close fills and proportional
costs. It does not model slippage, market impact, auction fills, taxes or cash
interest. The example makes no change to those documented
[engine assumptions](DESIGN.md). Results describe only the specified fictional
sample and conventions.

The example preserves the existing public APIs. Its tests perturb current and
future prices to check that prior targets stay unchanged, check the shared
cash warmup and feasible weights, and verify that adding costs cannot improve
wealth when targets are held fixed. Actual subprocess CLI runs verify the
export schema and deterministic files:

```bash
uv run pytest tests/test_walk_forward_example.py -q
```

Regeneration is deterministic with a fixed seed and the dependencies in
`uv.lock`. To reproduce the default exports in two directories:

```bash
uv run python examples/walk_forward_risk_parity.py --output-dir /tmp/quantkit-first
uv run python examples/walk_forward_risk_parity.py --output-dir /tmp/quantkit-second
diff -r /tmp/quantkit-first /tmp/quantkit-second
```

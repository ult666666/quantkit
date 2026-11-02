# Command line

The `quantkit` command prices options, writes synthetic prices, and runs backtests,
efficient frontiers and VaR backtests on a CSV of daily closes. The outputs below come
from these commands run in a source checkout, where `uv run` puts the command on the
path; after `pip install`, drop the `uv run` prefix. Every command has `--help`.

## Options

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

## Backtests and frontiers

Backtests and frontiers read a CSV of daily closes, either long (`date,ticker,close`) or wide (`date` plus one column per ticker). To try them without real data, generate a synthetic file (five correlated GBM assets `AAA`–`EEE`, seed 42):

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

## VaR backtests

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

"""A daily, vectorized portfolio backtester with an explicit no-look-ahead contract.

Timing contract
---------------
Prices form a wide table with trading days ``t = 0, 1, ..., T-1`` as rows.

1. A *strategy* maps the price table to a table of target weights with the same
   shape. Row ``t`` may use only prices from rows ``0..t``: the information
   available at the close of day ``t``. A fully specified row means "rebalance
   to these weights at today's close"; an all-NaN row means "do not trade
   today". Partially specified rows are rejected.
2. Trades execute at the close of day ``t`` at that close's prices. The new
   weights are held during day ``t + 1`` and earn its return
   ``r[t+1] = P[t+1] / P[t] - 1``. A decision made with day ``t``'s prices can
   therefore never earn day ``t``'s return: there is no look-ahead.
3. Between trades the share counts stay fixed, so weights drift with prices.
   Weights need not sum to one: ``1 - sum(w)`` is cash earning zero, and negative
   weights are short positions.
4. Each trade costs ``cost_bps`` basis points of the traded notional, deducted
   from wealth at the close of the trading day.

Accounting, with weights as fractions of current wealth::

    held[t]      weights held during day t (set at the close of day t-1; 0 on day 0)
    gross[t]   = sum_i held[t, i] * r[t, i]                  return before costs
    drifted[t] = held[t] * (1 + r[t]) / (1 + gross[t])       weights just before trading
    turnover[t]= sum_i |target[t, i] - drifted[t, i]|         on trading days, else 0
    net[t]     = (1 + gross[t]) * (1 - cost * turnover[t]) - 1
    equity[t]  = initial_capital * prod_{s <= t} (1 + net[s])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import require_finite, require_positive, validate_prices
from quantkit.backtest.metrics import TRADING_DAYS, performance_summary
from quantkit.exceptions import InvalidParameterError

Strategy = Callable[[pd.DataFrame], pd.DataFrame]
"""Maps a wide price table to target weights; see the module docstring for the contract."""


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a backtest. Every series shares the price table's DatetimeIndex.

    Attributes:
        equity: Portfolio value at each close, after that day's trading costs.
        returns: Net daily returns. Day 0 has no market exposure yet, so its
            return is just the cost of the initial trades.
        weights: Weights held during each day, i.e. those set at the previous
            close (all zero on day 0).
        turnover: Traded notional at each close as a fraction of wealth (the sum
            of absolute weight changes; buying 100% from cash is 1.0).
        costs: Trading costs at each close as a fraction of wealth.
    """

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    def summary(
        self, *, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
    ) -> pd.Series:
        """Headline performance metrics; see :func:`performance_summary`."""
        return performance_summary(
            self.returns,
            turnover=self.turnover,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    cost_bps: float = 0.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Backtest ``strategy`` on daily ``prices``.

    Args:
        prices: Wide table of closing prices (DatetimeIndex x tickers).
        strategy: Callable returning target weights; see the module docstring.
        cost_bps: Proportional trading cost in basis points of traded notional.
        initial_capital: Wealth before the first trade.

    Returns:
        Equity curve, daily returns, weights, turnover and costs.
    """
    prices = validate_prices(prices, require_datetime_index=True)
    return backtest_weights(
        prices, strategy(prices), cost_bps=cost_bps, initial_capital=initial_capital
    )


def backtest_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Backtest a precomputed table of target weights (all-NaN rows = no trade).

    The computation is vectorized over days. Between two trades the share counts
    are fixed, so the weight of an asset is its weight at the last trade scaled by
    its price growth since then, divided by the growth of the whole portfolio
    (cash does not grow). That gives every day's post-trade weights without a
    Python loop over days.

    Raises:
        InvalidParameterError: If the weights do not match the prices, contain
            partially specified rows or infinite values, the cost is negative,
            or the portfolio loses all of its value (possible with leverage or
            short positions).
    """
    prices = validate_prices(prices, require_datetime_index=True)
    targets = _validate_targets(target_weights, prices)
    if require_finite("cost_bps", cost_bps) < 0:
        raise InvalidParameterError(f"cost_bps must be non-negative, got {cost_bps}")
    cost_rate = cost_bps / 10_000.0
    initial_capital = require_positive("initial_capital", initial_capital)

    trades = targets.notna().all(axis=1)
    # Weights and prices as of the most recent trade (zero weights = all cash
    # before the first trade; the growth there is irrelevant, so set it to 1).
    anchor_weights = targets.ffill().fillna(0.0)
    anchor_prices = prices.loc[trades].reindex(prices.index).ffill()
    growth = (prices / anchor_prices).fillna(1.0)
    positions = anchor_weights * growth
    cash = 1.0 - anchor_weights.sum(axis=1)
    post_trade = positions.div(positions.sum(axis=1) + cash, axis=0)

    held = post_trade.shift(1).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    gross = (held * asset_returns).sum(axis=1)
    wiped_out = gross <= -1.0
    if wiped_out.any():
        raise InvalidParameterError(
            f"the portfolio lost all of its value on {_first_day(gross.index, wiped_out)}; "
            "check the leverage and short positions in the target weights"
        )
    drifted = (held * (1.0 + asset_returns)).div(1.0 + gross, axis=0)
    turnover = (post_trade - drifted).abs().sum(axis=1).where(trades, 0.0)
    costs = cost_rate * turnover
    if (costs >= 1.0).any():
        raise InvalidParameterError("trading costs exceed the portfolio value")
    net = (1.0 + gross) * (1.0 - costs) - 1.0

    return BacktestResult(
        equity=(initial_capital * (1.0 + net).cumprod()).rename("equity"),
        returns=net.rename("returns"),
        weights=held,
        turnover=turnover.rename("turnover"),
        costs=costs.rename("costs"),
    )


def _first_day(index: pd.Index, mask: pd.Series | npt.NDArray[np.bool_]) -> str:
    """The first date of ``index`` where ``mask`` is True, formatted for messages."""
    return str(pd.Timestamp(index[int(np.argmax(np.asarray(mask)))]).date())


def _validate_targets(target_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(target_weights, pd.DataFrame):
        raise InvalidParameterError(
            f"strategy must return a DataFrame, got {type(target_weights).__name__}"
        )
    if not (
        target_weights.index.equals(prices.index) and target_weights.columns.equals(prices.columns)
    ):
        raise InvalidParameterError(
            "target weights must have exactly the same dates and tickers as the prices"
        )
    try:
        targets = target_weights.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("target weights must be numeric") from exc
    values = targets.to_numpy()
    if np.isinf(values).any():
        raise InvalidParameterError("target weights must not be infinite")
    n_missing = np.isnan(values).sum(axis=1)
    partial = (n_missing > 0) & (n_missing < values.shape[1])
    if partial.any():
        raise InvalidParameterError(
            f"target weights on {_first_day(targets.index, partial)} are partially "
            "specified; give every ticker a weight, or NaN for all of them to skip trading"
        )
    return targets


def check_no_lookahead(
    strategy: Strategy, prices: pd.DataFrame, *, cutoffs: int = 5, seed: int = 0
) -> None:
    """Raise if ``strategy``'s weight for some day depends on later prices.

    For several cutoff days ``t`` the prices after ``t`` are scrambled (multiplied
    by random factors) and the strategy is re-run. A causal strategy produces
    identical target weights for days ``0..t``.

    Args:
        strategy: The strategy to check.
        prices: A realistic price table to run it on.
        cutoffs: Number of cutoff days to try.
        seed: Seed for the cutoffs and the scrambling factors.

    Raises:
        InvalidParameterError: If a look-ahead is detected.
    """
    prices = validate_prices(prices, require_datetime_index=True)
    rng = np.random.default_rng(seed)
    baseline = strategy(prices).to_numpy()
    n_cutoffs = min(cutoffs, len(prices) - 1)
    for cutoff in sorted(int(c) for c in rng.choice(len(prices) - 1, n_cutoffs, replace=False)):
        later = prices.iloc[cutoff + 1 :]
        scrambled = prices.copy()
        scrambled.iloc[cutoff + 1 :] = later.to_numpy() * rng.uniform(0.5, 1.5, later.shape)
        rerun = strategy(scrambled).to_numpy()
        if not np.array_equal(baseline[: cutoff + 1], rerun[: cutoff + 1], equal_nan=True):
            day = pd.Timestamp(prices.index[cutoff]).date()
            raise InvalidParameterError(
                f"look-ahead detected: target weights up to {day} changed when only "
                "later prices were modified"
            )

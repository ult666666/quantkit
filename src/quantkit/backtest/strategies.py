"""Example strategies for :func:`quantkit.backtest.run_backtest`.

Each strategy is a small frozen dataclass whose ``__call__`` maps a wide price
table to target weights, following the engine's contract: row ``t`` uses only
prices up to day ``t``, and an all-NaN row means "do not trade today". Signals
use trailing windows (``rolling``, ``shift`` with positive lags), which only ever
look backwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import require_finite, require_int
from quantkit.exceptions import InvalidParameterError

Frequency = Literal["D", "W", "M", "Q", "Y"]
"""Rebalancing calendar: daily, weekly, monthly, quarterly or yearly."""

_FREQUENCIES = ("D", "W", "M", "Q", "Y")


def rebalance_days(index: pd.DatetimeIndex, frequency: Frequency) -> npt.NDArray[np.bool_]:
    """Flag the first trading day of each calendar period (day 0 is always flagged).

    Deciding whether day ``t`` starts a new period needs only the dates of days
    ``t - 1`` and ``t``, so the schedule never peeks ahead. Rebalancing on the
    *last* trading day of each month would: on a given day you cannot know it is
    the month's last trading day without seeing tomorrow's date.
    """
    if frequency not in _FREQUENCIES:
        raise InvalidParameterError(f"frequency must be one of {_FREQUENCIES}, got {frequency!r}")
    periods = index.to_period(frequency)
    flags = np.ones(len(index), dtype=bool)
    flags[1:] = periods[1:] != periods[:-1]
    return flags


def _trade_only_on(targets: pd.DataFrame, trade_days: npt.NDArray[np.bool_]) -> pd.DataFrame:
    """Blank out (set to NaN) the rows on which the strategy should not trade."""
    result = targets.copy()
    result.loc[~trade_days] = np.nan
    return result


def _dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """The price table's dates, which calendar-based rebalancing needs."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise InvalidParameterError("prices must be indexed by a pandas DatetimeIndex")
    return prices.index


@dataclass(frozen=True)
class BuyAndHold:
    """Buy at the first close, then never trade again; weights drift with prices.

    Attributes:
        weights: Initial weight per ticker (tickers not listed get zero). Defaults
            to equal weights.
    """

    weights: Mapping[str, float] | None = None

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights: the initial allocation on day 0, no trades afterwards."""
        if self.weights is None:
            initial = pd.Series(1.0 / prices.shape[1], index=prices.columns)
        else:
            unknown = set(self.weights) - set(prices.columns)
            if unknown:
                raise InvalidParameterError(f"weights mention unknown tickers: {sorted(unknown)}")
            initial = pd.Series(
                {
                    ticker: require_finite(f"weight of {ticker}", w)
                    for ticker, w in self.weights.items()
                }
            ).reindex(prices.columns, fill_value=0.0)
        targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        targets.iloc[0] = initial.to_numpy()
        return targets


@dataclass(frozen=True)
class EqualWeight:
    """Hold ``1/N`` of wealth in each asset, rebalancing once per period.

    Attributes:
        rebalance: Rebalance on the first trading day of each ``"D"``ay,
            ``"W"``eek, ``"M"``onth, ``"Q"``uarter or ``"Y"``ear.
    """

    rebalance: Frequency = "M"

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights: ``1/N`` on rebalance days, no trades otherwise."""
        targets = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
        return _trade_only_on(targets, rebalance_days(_dates(prices), self.rebalance))


@dataclass(frozen=True)
class MovingAverageCrossover:
    """Trend following: hold an asset while its fast moving average is above its slow one.

    Each asset owns a slot of ``1/N`` of wealth, invested while the ``fast``-day
    simple moving average of its price is above the ``slow``-day one and held in
    cash otherwise (long/flat, no shorting). The portfolio trades only on days
    when at least one signal flips, resetting every position to its slot size;
    in between, positions drift.

    Attributes:
        fast: Window of the fast moving average, in trading days.
        slow: Window of the slow moving average; must exceed ``fast``.
    """

    fast: int = 50
    slow: int = 200

    def __post_init__(self) -> None:
        """Validate the windows."""
        require_int("fast", self.fast, minimum=1)
        require_int("slow", self.slow, minimum=1)
        if self.fast >= self.slow:
            raise InvalidParameterError(
                f"fast window ({self.fast}) must be shorter than slow window ({self.slow})"
            )

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights: ``1/N`` per asset in an uptrend, re-set when a signal flips."""
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()
        # While the averages are still warming up they are NaN; NaN > NaN is False,
        # so the slot stays in cash until the slow average exists.
        in_trend = (fast_ma > slow_ma).astype(float)
        targets = in_trend / prices.shape[1]
        flipped = targets.ne(targets.shift(1)).any(axis=1).to_numpy()
        return _trade_only_on(targets, flipped)


@dataclass(frozen=True)
class TimeSeriesMomentum:
    """Time-series momentum: follow each asset's own trailing return.

    A simplified version of Moskowitz, Ooi & Pedersen (2012): on each rebalance
    day, every asset with a positive return over the past ``lookback`` days gets
    weight ``+1/N``, and every asset with a negative one ``-1/N`` (or zero when
    ``long_only``). The paper also scales positions by volatility; this version
    does not.

    Attributes:
        lookback: Trailing window in trading days (252 = about 12 months).
        rebalance: Rebalance calendar, as in :class:`EqualWeight`.
        long_only: Replace short positions with cash.
    """

    lookback: int = 252
    rebalance: Frequency = "M"
    long_only: bool = False

    def __post_init__(self) -> None:
        """Validate the lookback window."""
        require_int("lookback", self.lookback, minimum=1)

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights: ``+/- 1/N`` by the sign of the trailing return."""
        trailing = prices / prices.shift(self.lookback) - 1.0
        # gt/lt are False for the NaN warm-up rows, which therefore stay in cash.
        signal = trailing.gt(0.0).astype(float)
        if not self.long_only:
            signal = signal - trailing.lt(0.0).astype(float)
        targets = signal / prices.shape[1]
        return _trade_only_on(targets, rebalance_days(_dates(prices), self.rebalance))


@dataclass(frozen=True)
class InverseVolatility:
    """Fully invested, with weights proportional to 1 / (trailing volatility).

    Volatility is the standard deviation of the last ``window`` daily returns.
    Assets that have been calmer get more capital, so each contributes roughly
    similar risk when correlations are similar (a simple cousin of risk parity).

    Attributes:
        window: Trailing window for volatility, in trading days.
        rebalance: Rebalance calendar, as in :class:`EqualWeight`.
    """

    window: int = 63
    rebalance: Frequency = "M"

    def __post_init__(self) -> None:
        """Validate the volatility window."""
        require_int("window", self.window, minimum=2)

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights proportional to inverse volatility on rebalance days."""
        inverse_vol = 1.0 / prices.pct_change().rolling(self.window).std()
        # Rows are NaN until a full window of returns exists, so no trade happens
        # (the portfolio stays in cash) during the warm-up.
        targets = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)
        return _trade_only_on(targets, rebalance_days(_dates(prices), self.rebalance))

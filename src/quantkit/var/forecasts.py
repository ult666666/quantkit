"""Rolling one-day Value at Risk and expected shortfall forecasts.

A forecast for day ``t`` is made at the close of day ``t - 1``. It uses the
returns of days ``t - h, ..., t - 1`` and nothing else: never the return
``r[t]`` that it is scored against, and nothing older than its history ``h``.

Every method describes the next return as a location-scale model::

    r[t] = mu[t] + sigma[t] * e[t]

The methods differ in how ``mu[t]`` and ``sigma[t]`` are estimated from the
window ``r[t - window], ..., r[t - 1]`` and in where the innovation ``e[t]``
comes from:

==========  ===========  ====================  ===============================  ==========
method      mu[t]        sigma[t]              e[t]                             history h
==========  ===========  ====================  ===============================  ==========
historical  0            1                     resampled window returns         window
gaussian    window mean  window std (ddof=1)   standard normal                  window
ewma        0            EWMA volatility       standard normal                  window
fhs         0            EWMA volatility       resampled standardized returns   2 * window
==========  ===========  ====================  ===============================  ==========

* **EWMA** (RiskMetrics, J.P. Morgan/Reuters 1996) weights the squared return
  ``k`` days back by ``decay**k``, normalized over the window so the weights sum
  to one: ``sigma[t]**2 = sum_k w_k r[t-1-k]**2``. The mean is taken as zero.
* **Filtered historical simulation** (Barone-Adesi, Giannopoulos & Vosper 1999;
  the volatility weighting of Hull & White 1998) divides each window return
  ``r[s]`` by the EWMA volatility forecast for its own day ``s``, then rescales
  the standardized returns by today's volatility. Each of those volatilities
  needs its own window, so FHS needs two windows of history.

VaR and ES at confidence level ``c`` are positive loss fractions, as in
:mod:`quantkit.portfolio.risk`:

* standard normal innovations: ``VaR = z_c sigma - mu`` and
  ``ES = sigma phi(z_c) / (1 - c) - mu``;
* empirical innovations: the tail is the ``m = ceil(window * (1 - c))`` smallest
  innovations, ``VaR = -(mu + sigma e_(m))`` and ``ES = -(mu + sigma mean(tail))``.
  For historical simulation these are exactly :func:`~quantkit.portfolio.historical_var`
  and :func:`~quantkit.portfolio.historical_cvar` of the window.

The same representation defines :meth:`RiskForecast.simulate`, which draws
returns from each day's forecast distribution; the expected shortfall backtest
uses it to simulate its null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quantkit._validation import (
    FloatArray,
    SeedLike,
    as_1d_array,
    require_int,
    require_level,
    tail_count,
)
from quantkit.exceptions import InvalidParameterError
from quantkit.var.backtests import var_exceptions

Method = Literal["historical", "gaussian", "ewma", "fhs"]
"""A rolling forecasting method; see the module docstring."""

METHODS: tuple[Method, ...] = ("historical", "gaussian", "ewma", "fhs")

RISKMETRICS_DECAY = 0.94
"""Decay factor that RiskMetrics uses for daily volatility forecasts."""

_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class RiskForecast:
    """Rolling one-day VaR and ES forecasts and the returns they are scored against.

    Every series is indexed by the day being forecast: ``var.iloc[k]`` was made at
    the previous close, before ``returns.iloc[k]`` was known.

    Attributes:
        method: The forecasting method.
        level: Confidence level, e.g. 0.99.
        window: Number of past returns in each estimation window.
        decay: EWMA decay factor (used by ``ewma`` and ``fhs`` only).
        returns: Realized returns on the forecast days.
        var: VaR forecasts, as positive loss fractions.
        es: Expected shortfall forecasts; ``es >= var``.
        location: ``mu[t]`` of the location-scale model.
        scale: ``sigma[t]`` of the location-scale model.
        innovations: For ``historical`` and ``fhs``, each day's pool of innovations
            (one row per forecast day, ``window`` columns, oldest first). None when
            the innovations are standard normal.
    """

    method: Method
    level: float
    window: int
    decay: float
    returns: pd.Series
    var: pd.Series
    es: pd.Series
    location: pd.Series
    scale: pd.Series
    innovations: FloatArray | None

    def __len__(self) -> int:
        """Number of forecast days."""
        return len(self.var)

    @property
    def exceptions(self) -> pd.Series:
        """True on the days whose loss exceeded the VaR forecast (``return < -VaR``)."""
        return var_exceptions(self.returns, self.var)

    def last(self, n_days: int) -> RiskForecast:
        """The same forecasts restricted to the most recent ``n_days`` days."""
        n_days = require_int("n_days", n_days, minimum=1)
        if n_days > len(self):
            raise InvalidParameterError(
                f"n_days must be at most the number of forecasts ({len(self)}), got {n_days}"
            )
        keep = slice(len(self) - n_days, None)
        return replace(
            self,
            returns=self.returns.iloc[keep],
            var=self.var.iloc[keep],
            es=self.es.iloc[keep],
            location=self.location.iloc[keep],
            scale=self.scale.iloc[keep],
            innovations=None if self.innovations is None else self.innovations[keep],
        )

    def simulate(self, n_paths: int, *, seed: SeedLike = None) -> FloatArray:
        """Draw returns from each day's forecast distribution.

        Column ``k`` holds independent draws of ``location[k] + scale[k] * e``, with
        ``e`` standard normal or drawn uniformly from that day's innovation pool.
        If the forecasts are correct, the realized returns are one such path.

        Args:
            n_paths: Number of simulated paths (rows).
            seed: Seed or :class:`numpy.random.Generator` for reproducibility.

        Returns:
            An array of shape ``(n_paths, len(self))``.
        """
        n_paths = require_int("n_paths", n_paths, minimum=1)
        rng = np.random.default_rng(seed)
        n_days = len(self)
        if self.innovations is None:
            draws = rng.standard_normal((n_paths, n_days))
        else:
            picks = rng.integers(0, self.window, size=(n_paths, n_days))
            draws = self.innovations[np.arange(n_days), picks]
        simulated: FloatArray = self.location.to_numpy() + self.scale.to_numpy() * draws
        return simulated


def rolling_var_es(
    returns: pd.Series | npt.ArrayLike,
    *,
    method: Method = "historical",
    level: float = 0.99,
    window: int = 250,
    decay: float = RISKMETRICS_DECAY,
) -> RiskForecast:
    """Forecast one-day VaR and ES for every day that has enough history.

    Args:
        returns: Daily returns in time order, e.g. a portfolio's simple returns.
            A Series keeps its index (a DatetimeIndex must be sorted and unique);
            other sequences get a RangeIndex.
        method: ``"historical"``, ``"gaussian"``, ``"ewma"`` or ``"fhs"``.
        level: Confidence level strictly between 0 and 1, e.g. 0.99.
        window: Number of past returns used by each forecast (at least 2). FHS
            uses two windows: see the module docstring.
        decay: EWMA decay factor strictly between 0 and 1 (``ewma`` and ``fhs``).

    Returns:
        Forecasts for the days ``h, ..., n - 1``, where ``h`` is the method's
        history, aligned with the realized returns of those days.

    Raises:
        InvalidParameterError: For an unknown method, invalid parameters, too few
            returns, or an FHS window whose EWMA volatility is zero.
    """
    if method not in METHODS:
        raise InvalidParameterError(f"method must be one of {METHODS}, got {method!r}")
    level = require_level("level", level)
    window = require_int("window", window, minimum=2)
    decay = require_level("decay", decay)
    series = _return_series(returns)
    r = series.to_numpy()
    history = 2 * window if method == "fhs" else window
    if len(r) <= history:
        raise InvalidParameterError(
            f"the {method} method with window {window} needs more than {history} returns, "
            f"got {len(r)}"
        )

    # Row k holds r[k], ..., r[k + window - 1]: the window for day t = k + window.
    windows = sliding_window_view(r, window)[: len(r) - window]
    n_days = len(r) - history
    location = np.zeros(n_days)
    innovations: FloatArray | None = None
    if method == "historical":
        scale = np.ones(n_days)
        innovations = windows
    elif method == "gaussian":
        location = windows.mean(axis=1)
        scale = windows.std(axis=1, ddof=1)
    else:
        volatility = _ewma_volatility(windows, decay)  # forecasts for days window..n-1
        if method == "ewma":
            scale = volatility
        else:
            if (volatility <= 0.0).any():
                day = series.index[window + int(np.argmax(volatility <= 0.0))]
                raise InvalidParameterError(
                    f"the EWMA volatility forecast for {day} is zero, so FHS cannot "
                    "standardize that return; every window needs a non-zero return"
                )
            standardized = r[window:] / volatility
            # Row j holds the standardized returns of days window + j, ...,
            # 2 * window + j - 1: the pool for day t = 2 * window + j.
            innovations = sliding_window_view(standardized, window)[:n_days]
            scale = volatility[window:]

    var, es = _var_and_es(location, scale, innovations, level)
    index = series.index[history:]
    return RiskForecast(
        method=method,
        level=level,
        window=window,
        decay=decay,
        returns=series.iloc[history:],
        var=pd.Series(var, index=index, name="var"),
        es=pd.Series(es, index=index, name="es"),
        location=pd.Series(location, index=index, name="location"),
        scale=pd.Series(scale, index=index, name="scale"),
        innovations=innovations,
    )


def ewma_weights(window: int, decay: float = RISKMETRICS_DECAY) -> FloatArray:
    """EWMA weights for a window of squared returns, oldest first, summing to one.

    The return ``k`` days before the forecast day gets weight proportional to
    ``decay**k``. RiskMetrics uses ``(1 - decay) * decay**k`` over an unlimited
    history; normalizing over a finite window differs from that by the omitted
    weight ``decay**window`` (about 2e-7 for 250 days at 0.94).
    """
    window = require_int("window", window, minimum=1)
    decay = require_level("decay", decay)
    weights: FloatArray = np.power(decay, np.arange(window - 1, -1, -1, dtype=np.float64))
    normalized: FloatArray = weights / weights.sum()
    return normalized


def _ewma_volatility(windows: FloatArray, decay: float) -> FloatArray:
    volatility: FloatArray = np.sqrt(np.square(windows) @ ewma_weights(windows.shape[1], decay))
    return volatility


def _var_and_es(
    location: FloatArray, scale: FloatArray, innovations: FloatArray | None, level: float
) -> tuple[FloatArray, FloatArray]:
    """VaR and ES of ``location + scale * e`` for normal or empirical innovations."""
    if innovations is None:
        z = _STANDARD_NORMAL.inv_cdf(level)
        var = z * scale - location
        es = scale * (_STANDARD_NORMAL.pdf(z) / (1.0 - level)) - location
        return var, es
    m = tail_count(innovations.shape[1], level)
    # After partitioning, column m - 1 holds the m-th smallest innovation and the
    # columns before it the m - 1 smaller ones (in no particular order).
    tail = np.partition(innovations, m - 1, axis=1)[:, :m]
    var = -(location + scale * tail[:, -1])
    es = -(location + scale * tail.mean(axis=1))
    return var, es


def _return_series(returns: pd.Series | npt.ArrayLike) -> pd.Series:
    """A float copy of ``returns`` as a Series, after checking it is a usable series."""
    values = as_1d_array("returns", returns, min_length=2).copy()
    if not isinstance(returns, pd.Series):
        return pd.Series(values, name="returns")
    index = returns.index
    if index.has_duplicates:
        raise InvalidParameterError("returns must not have duplicate index labels")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise InvalidParameterError("returns must be sorted by date")
    return pd.Series(values, index=index, name="returns")

"""Performance metrics for a series of periodic (e.g. daily) portfolio returns.

All functions take simple returns, one per period, and annualize with
``periods_per_year`` (252 trading days by default). Ratios that would divide by
zero, such as a Sharpe ratio of a constant return series, are returned as NaN.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import FloatArray, as_1d_array, require_finite, require_int
from quantkit.exceptions import InvalidParameterError

TRADING_DAYS = 252


def _returns(returns: npt.ArrayLike, min_length: int = 1) -> FloatArray:
    r = as_1d_array("returns", returns, min_length=min_length)
    if (r < -1.0).any():
        raise InvalidParameterError("returns must be >= -1 (a loss cannot exceed 100%)")
    return r


def _no_dispersion(dispersion: float, values: FloatArray) -> bool:
    """Whether a standard deviation is zero up to floating-point round-off.

    A constant series has zero standard deviation in exact arithmetic, but
    round-off can leave something like 1e-19, which would make a ratio explode.
    """
    return dispersion <= 1e-12 * float(np.max(np.abs(values)))


def _per_period_rate(annual_rate: float, periods_per_year: int) -> float:
    """Per-period rate that compounds to ``annual_rate`` over one year."""
    if not require_finite("risk_free_rate", annual_rate) > -1.0:
        raise InvalidParameterError(f"risk_free_rate must exceed -1, got {annual_rate}")
    return math.pow(1.0 + annual_rate, 1.0 / periods_per_year) - 1.0


def total_return(returns: npt.ArrayLike) -> float:
    """Compounded return over the whole series: ``prod(1 + r) - 1``."""
    return float(np.prod(1.0 + _returns(returns))) - 1.0


def cagr(returns: npt.ArrayLike, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate: ``prod(1 + r) ** (periods_per_year / n) - 1``."""
    r = _returns(returns)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    growth = float(np.prod(1.0 + r))
    return math.pow(growth, periods_per_year / len(r)) - 1.0


def annualized_volatility(
    returns: npt.ArrayLike, *, periods_per_year: int = TRADING_DAYS
) -> float:
    """Sample standard deviation of returns scaled by ``sqrt(periods_per_year)``."""
    r = _returns(returns, min_length=2)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    return float(r.std(ddof=1)) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: npt.ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio: ``mean(r - rf) / std(r - rf) * sqrt(periods_per_year)``.

    ``risk_free_rate`` is annual and is converted to the per-period rate that
    compounds to it. Scaling by ``sqrt(periods_per_year)`` assumes returns are
    serially uncorrelated.
    """
    r = _returns(returns, min_length=2)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    excess = r - _per_period_rate(risk_free_rate, periods_per_year)
    std = float(excess.std(ddof=1))
    if _no_dispersion(std, excess):
        return math.nan
    return float(excess.mean()) / std * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: npt.ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sortino ratio: mean excess return over downside deviation.

    The downside deviation is ``sqrt(mean(min(r - rf, 0)^2))`` over *all*
    periods, so only returns below the risk-free rate count as risk.
    """
    r = _returns(returns, min_length=2)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    excess = r - _per_period_rate(risk_free_rate, periods_per_year)
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    if _no_dispersion(downside, excess):
        return math.nan
    return float(excess.mean()) / downside * math.sqrt(periods_per_year)


def _drawdowns(r: FloatArray) -> FloatArray:
    wealth = np.cumprod(1.0 + r)
    # The running peak starts at the initial wealth of 1, so a loss in the very
    # first period already counts as a drawdown.
    peak = np.maximum(np.maximum.accumulate(wealth), 1.0)
    result: FloatArray = wealth / peak - 1.0
    return result


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown after each period: wealth relative to its running peak, minus 1 (<= 0)."""
    if not isinstance(returns, pd.Series):
        raise InvalidParameterError("returns must be a pandas Series")
    return pd.Series(_drawdowns(_returns(returns)), index=returns.index, name="drawdown")


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Largest peak-to-trough decline in wealth, as a positive fraction (0.25 = -25%)."""
    return -float(np.min(_drawdowns(_returns(returns))))


def max_drawdown_duration(returns: npt.ArrayLike) -> int:
    """Longest run of consecutive periods spent below a previous wealth peak.

    The count runs from the first period under water until wealth recovers to
    its old peak (or until the end of the series if it never does).
    """
    longest = current = 0
    for underwater in _drawdowns(_returns(returns)) < 0.0:
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(returns: npt.ArrayLike, *, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by maximum drawdown (NaN if there was no drawdown)."""
    drawdown = max_drawdown(returns)
    if drawdown == 0.0:
        return math.nan
    return cagr(returns, periods_per_year=periods_per_year) / drawdown


def hit_rate(returns: npt.ArrayLike) -> float:
    """Fraction of periods with a positive return, among periods with a non-zero one.

    Zero-return periods (for example, days spent entirely in cash) are excluded so
    that inactivity does not dilute the statistic. NaN if every return is zero.
    """
    r = _returns(returns)
    active = r[r != 0.0]
    if active.size == 0:
        return math.nan
    return float(np.mean(active > 0.0))


def annualized_turnover(turnover: npt.ArrayLike, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Average traded notional per year, as a multiple of portfolio value.

    ``turnover`` holds the per-period sum of absolute weight changes, so 2.0 per
    year means trading the whole portfolio out and back in once a year.
    """
    t = as_1d_array("turnover", turnover)
    if (t < 0.0).any():
        raise InvalidParameterError("turnover must be non-negative")
    return float(t.mean()) * require_int("periods_per_year", periods_per_year, minimum=1)


def performance_summary(
    returns: npt.ArrayLike,
    *,
    turnover: npt.ArrayLike | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """All headline metrics in one labelled Series (handy for side-by-side tables)."""
    metrics: dict[str, float] = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year=periods_per_year),
        "volatility": annualized_volatility(returns, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "sortino": sortino_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_periods": float(max_drawdown_duration(returns)),
        "calmar": calmar_ratio(returns, periods_per_year=periods_per_year),
        "hit_rate": hit_rate(returns),
    }
    if turnover is not None:
        metrics["annual_turnover"] = annualized_turnover(
            turnover, periods_per_year=periods_per_year
        )
    return pd.Series(metrics, name="value")

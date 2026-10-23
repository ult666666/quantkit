"""Portfolio risk measures: volatility, Value at Risk and expected shortfall.

Value at Risk (VaR) and conditional VaR (CVaR, also called expected shortfall)
are reported as positive numbers in return units: a one-period 95% VaR of 0.021
means a loss of 2.1% or more is expected on 5% of periods. CVaR is the average
loss on those worst periods, so CVaR >= VaR.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd

from quantkit._validation import FloatArray, as_1d_array, require_level, validate_covariance
from quantkit.exceptions import InvalidParameterError

_STANDARD_NORMAL = NormalDist()

# (1 - 0.99) * 500 evaluates to 5.000000000000004 in binary floating point; this
# slack stops math.ceil from turning that into 6.
_FLOAT_SLACK = 1e-9


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Portfolio volatility ``sqrt(w' Sigma w)``, in the units of ``cov``."""
    sigma = validate_covariance(cov)
    if not isinstance(weights, pd.Series) or set(weights.index) != set(cov.index):
        raise InvalidParameterError("weights must be a Series covering exactly the assets in cov")
    w = weights.reindex(cov.index).to_numpy(dtype=np.float64)
    return math.sqrt(max(float(w @ sigma @ w), 0.0))


def _tail(returns: npt.ArrayLike, level: float) -> FloatArray:
    """The ``ceil(n * (1 - level))`` worst returns, worst first."""
    r = np.sort(as_1d_array("returns", returns, min_length=2))
    level = require_level("level", level)
    n_tail = max(1, math.ceil(len(r) * (1.0 - level) - _FLOAT_SLACK))
    return r[:n_tail]


def historical_var(returns: npt.ArrayLike, *, level: float = 0.95) -> float:
    """Historical-simulation Value at Risk.

    The worst ``m = ceil(n * (1 - level))`` of ``n`` observed returns form the
    tail, and VaR is the smallest loss among them: the ``m``-th worst return,
    negated. This is the empirical ``(1 - level)`` quantile, e.g. the 5th worst
    of 500 daily returns for 99% VaR.

    Args:
        returns: Periodic returns (a 1-D array, list or Series).
        level: Confidence level strictly between 0 and 1, e.g. 0.95 or 0.99.

    Returns:
        VaR as a positive loss fraction (negative if even the tail is a gain).
    """
    return -float(_tail(returns, level)[-1])


def historical_cvar(returns: npt.ArrayLike, *, level: float = 0.95) -> float:
    """Historical conditional VaR (expected shortfall): the mean loss in the tail.

    Uses the same ``ceil(n * (1 - level))`` worst returns as
    :func:`historical_var` and averages them.
    """
    return -float(_tail(returns, level).mean())


def _mean_and_vol(returns: npt.ArrayLike, level: float) -> tuple[float, float, float]:
    r = as_1d_array("returns", returns, min_length=2)
    z = _STANDARD_NORMAL.inv_cdf(require_level("level", level))
    return float(r.mean()), float(r.std(ddof=1)), z


def parametric_var(returns: npt.ArrayLike, *, level: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR: ``z_level * sigma - mu``.

    Assumes returns are normal with the sample mean ``mu`` and standard deviation
    ``sigma``; ``z_level`` is the standard normal quantile (1.645 at 95%). Real
    returns have fatter tails, so this usually understates extreme losses.
    """
    mu, sigma, z = _mean_and_vol(returns, level)
    return z * sigma - mu


def parametric_cvar(returns: npt.ArrayLike, *, level: float = 0.95) -> float:
    """Gaussian expected shortfall: ``sigma * phi(z_level) / (1 - level) - mu``.

    For a normal distribution the mean of the worst ``1 - level`` fraction of
    outcomes has this closed form, where ``phi`` is the standard normal density.
    """
    mu, sigma, z = _mean_and_vol(returns, level)
    return sigma * _STANDARD_NORMAL.pdf(z) / (1.0 - level) - mu

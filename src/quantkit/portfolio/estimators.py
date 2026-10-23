"""Estimate returns, expected returns and covariance matrices from prices."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantkit._validation import require_int, validate_prices, validate_returns
from quantkit.exceptions import InvalidParameterError

ReturnMethod = Literal["simple", "log"]


def compute_returns(prices: pd.DataFrame, *, method: ReturnMethod = "simple") -> pd.DataFrame:
    """Periodic returns from a wide price table.

    * ``"simple"``: ``R_t = P_t / P_(t-1) - 1``. Simple returns aggregate across
      assets: a portfolio's return is the weighted sum ``sum_i w_i R_i``, so they
      are the right input for mean-variance optimization.
    * ``"log"``: ``r_t = ln(P_t / P_(t-1))``. Log returns aggregate across time:
      the log return over many periods is the sum of the one-period log returns.

    Args:
        prices: Wide price table (rows = dates, columns = assets).
        method: ``"simple"`` or ``"log"``.

    Returns:
        Returns with the first date dropped (it has no previous price).
    """
    prices = validate_prices(prices)
    growth = (prices / prices.shift(1)).iloc[1:]
    if method == "simple":
        return growth - 1.0
    if method == "log":
        return pd.DataFrame(np.log(growth.to_numpy()), index=growth.index, columns=growth.columns)
    raise InvalidParameterError(f"method must be 'simple' or 'log', got {method!r}")


def annualized_mean_returns(returns: pd.DataFrame, *, periods_per_year: int = 252) -> pd.Series:
    """Annualized arithmetic mean return per asset: ``mean(R) * periods_per_year``.

    This is the usual plug-in estimate of expected returns for mean-variance
    optimization. It is very noisy: with daily data its standard error is about
    ``sigma / sqrt(years)``, i.e. several percentage points per year even with a
    decade of history.
    """
    returns = validate_returns(returns)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    return returns.mean() * periods_per_year


def sample_covariance(returns: pd.DataFrame, *, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized sample covariance matrix (unbiased ``1 / (n - 1)`` normalization).

    Annualizing by ``periods_per_year`` assumes returns are serially
    uncorrelated, so that variance grows linearly with the horizon.
    """
    returns = validate_returns(returns)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    return returns.cov() * periods_per_year


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
    """Optimal shrinkage intensity of Ledoit & Wolf (2004) towards a scaled identity.

    Let ``X`` be the ``n x p`` matrix of demeaned returns, ``S = X'X / n`` the
    sample covariance and ``m = trace(S) / p`` its average variance. Ledoit and
    Wolf shrink ``S`` towards the target ``F = m I`` and estimate the intensity
    ``delta`` that minimizes the expected squared Frobenius distance to the true
    covariance. With ``||A||^2 = trace(A A') / p``::

        d^2     = ||S - m I||^2                          (dispersion of S around F)
        b_bar^2 = (1 / n^2) sum_k ||x_k x_k' - S||^2     (estimation noise in S)
        delta   = min(b_bar^2, d^2) / d^2

    where ``x_k`` is the k-th (demeaned) observation. Noisy estimates (``b_bar^2``
    large relative to ``d^2``) are shrunk harder; the ``min`` caps ``delta`` at 1.

    Args:
        returns: Periodic returns, one column per asset.

    Returns:
        The shrinkage intensity in ``[0, 1]``.
    """
    returns = validate_returns(returns)
    x = returns.to_numpy()
    x = x - x.mean(axis=0)
    n, p = x.shape
    sample = x.T @ x / n
    mu = np.trace(sample) / p
    d2 = float(np.sum((sample - mu * np.eye(p)) ** 2)) / p
    if d2 == 0.0:  # S is already a multiple of the identity: nothing to shrink.
        return 0.0
    # sum_k ||x_k x_k' - S||_F^2 simplifies to sum_k ||x_k||^4 - n ||S||_F^2,
    # which avoids building n separate p x p matrices.
    row_norms_sq = np.sum(x**2, axis=1)
    b2_bar = float(np.sum(row_norms_sq**2) - n * np.sum(sample**2)) / (n**2 * p)
    return min(b2_bar, d2) / d2


def ledoit_wolf_covariance(returns: pd.DataFrame, *, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage estimate of the covariance matrix.

    Returns ``delta m I + (1 - delta) S`` (see :func:`ledoit_wolf_shrinkage`),
    scaled by ``periods_per_year``. Unlike the sample covariance, the result is
    always well conditioned and invertible, even with more assets than
    observations, which makes optimizers built on its inverse far more stable.

    Note:
        Following the paper, ``S`` uses the ``1 / n`` normalization, so with zero
        shrinkage this differs from :func:`sample_covariance` by ``n / (n - 1)``.
    """
    returns = validate_returns(returns)
    periods_per_year = require_int("periods_per_year", periods_per_year, minimum=1)
    x = returns.to_numpy()
    x = x - x.mean(axis=0)
    n, p = x.shape
    sample = x.T @ x / n
    target = np.trace(sample) / p * np.eye(p)
    delta = ledoit_wolf_shrinkage(returns)
    shrunk = delta * target + (1.0 - delta) * sample
    return pd.DataFrame(shrunk * periods_per_year, index=returns.columns, columns=returns.columns)

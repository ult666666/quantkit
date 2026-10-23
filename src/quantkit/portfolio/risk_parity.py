"""Risk parity (equal risk contribution) and risk budgeting portfolios."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantkit._validation import (
    FloatArray,
    require_int,
    require_positive,
    require_positive_definite,
    validate_covariance,
)
from quantkit.exceptions import ConvergenceError, InvalidParameterError


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance contributed by each asset.

    Asset ``i`` contributes ``RC_i = w_i (Sigma w)_i / (w' Sigma w)``. Because
    portfolio volatility is homogeneous of degree one in the weights, Euler's
    theorem splits it exactly into these contributions, so they sum to one.

    Args:
        weights: Portfolio weights indexed by asset.
        cov: Covariance matrix labelled by asset.

    Returns:
        Fractional risk contributions indexed like ``cov``.
    """
    sigma = validate_covariance(cov)
    if not isinstance(weights, pd.Series) or set(weights.index) != set(cov.index):
        raise InvalidParameterError("weights must be a Series covering exactly the assets in cov")
    w = weights.reindex(cov.index).to_numpy(dtype=np.float64)
    variance = float(w @ sigma @ w)
    if not variance > 0:
        raise InvalidParameterError("portfolio variance must be positive")
    return pd.Series(w * (sigma @ w) / variance, index=cov.index, name="risk_contribution")


def risk_parity(
    cov: pd.DataFrame,
    *,
    budgets: pd.Series | None = None,
    tol: float = 1e-10,
    max_sweeps: int = 10_000,
) -> pd.Series:
    """Long-only portfolio whose risk contributions match the given budgets.

    With equal budgets (the default) this is the equal risk contribution (ERC)
    or "risk parity" portfolio of Maillard, Roncalli & Teiletche (2010). They show
    it solves the convex problem::

        minimize  f(y) = 0.5 y' Sigma y - sum_i b_i ln(y_i)   over y > 0

    Setting the gradient to zero gives ``y_i (Sigma y)_i = b_i``: each asset's
    risk contribution equals its budget. Rescaling ``w = y / sum(y)`` keeps the
    contributions' proportions, so ``w`` is the risk-budgeting portfolio. ``f`` is
    strictly convex, so the solution is unique.

    ``f`` is minimized by cyclical coordinate descent (Griveau-Billion, Richard &
    Roncalli 2013). Holding the other coordinates fixed, ``f`` is minimized in
    ``y_i`` by the positive root of the quadratic::

        Sigma_ii y_i^2 + c_i y_i - b_i = 0,   c_i = sum_(j != i) Sigma_ij y_j

    Sweeps over all assets repeat until every contribution is within ``tol`` of
    its budget.

    Args:
        cov: Positive definite covariance matrix labelled by asset.
        budgets: Target risk contributions indexed like ``cov``; positive and
            summing to one. Default: equal budgets ``1/n``.
        tol: Maximum allowed gap between a contribution and its budget.
        max_sweeps: Iteration limit.

    Returns:
        Long-only weights indexed like ``cov``, summing to one.

    Raises:
        InvalidParameterError: For an invalid covariance matrix or budgets.
        ConvergenceError: If the contributions do not reach ``tol``.
    """
    sigma = validate_covariance(cov)
    require_positive_definite(sigma)
    tol = require_positive("tol", tol)
    max_sweeps = require_int("max_sweeps", max_sweeps, minimum=1)
    b = _check_budgets(budgets, cov.index)

    # For uncorrelated assets the solution is exactly y_i = sqrt(b_i) / sigma_i,
    # which makes a good starting point in general.
    y = np.sqrt(b / np.diag(sigma))
    for _ in range(max_sweeps):
        for i in range(len(y)):
            c = float(sigma[i] @ y) - sigma[i, i] * y[i]
            y[i] = (-c + math.sqrt(c * c + 4.0 * sigma[i, i] * b[i])) / (2.0 * sigma[i, i])
        weights = y / y.sum()
        contributions = weights * (sigma @ weights) / float(weights @ sigma @ weights)
        if np.max(np.abs(contributions - b)) < tol:
            return pd.Series(weights, index=cov.index, name="weight")
    raise ConvergenceError(f"risk parity did not converge within {max_sweeps} sweeps")


def _check_budgets(budgets: pd.Series | None, labels: pd.Index) -> FloatArray:
    if budgets is None:
        return np.full(len(labels), 1.0 / len(labels))
    if not isinstance(budgets, pd.Series) or set(budgets.index) != set(labels):
        raise InvalidParameterError("budgets must be a Series covering exactly the assets in cov")
    b = budgets.reindex(labels).to_numpy(dtype=np.float64)
    if not np.isfinite(b).all() or (b <= 0).any():
        raise InvalidParameterError("risk budgets must be positive and finite")
    if not math.isclose(b.sum(), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise InvalidParameterError(f"risk budgets must sum to one, got {b.sum():.6g}")
    return b / b.sum()  # remove round-off so contributions can match to ``tol``

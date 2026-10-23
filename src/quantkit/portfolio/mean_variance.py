"""Mean-variance portfolio optimization (Markowitz 1952).

A fully invested portfolio with weights ``w`` (``sum(w) = 1``) has expected
return ``w'mu`` and variance ``w' Sigma w``. This module finds the global
minimum-variance (GMV) portfolio, the maximum-Sharpe (tangency) portfolio, and the
minimum-variance portfolio for each target return, which together trace the
efficient frontier.

Two solvers are provided on purpose:

* **Closed forms** for the problem without weight bounds (short sales allowed),
  derived from the Lagrangian first-order conditions (Merton 1972).
* **A numerical optimizer** (SciPy's SLSQP) that also handles weight bounds
  such as long-only.

The closed forms are an independent reference: run without bounds, the optimizer
must reproduce them, and the test suite checks that it does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantkit._validation import (
    FloatArray,
    require_finite,
    require_int,
    require_positive_definite,
    validate_covariance,
)
from quantkit.exceptions import ConvergenceError, InvalidParameterError

Bounds = tuple[float, float]
"""Lower and upper limit applied to every asset's weight."""

LONG_ONLY: Bounds = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class PortfolioPerformance:
    """Annualized expected return, volatility and Sharpe ratio of a portfolio."""

    expected_return: float
    volatility: float
    sharpe: float


# --------------------------------------------------------------------------- inputs


def _covariance(cov: pd.DataFrame) -> FloatArray:
    sigma = validate_covariance(cov)
    require_positive_definite(sigma)
    return sigma


def _aligned(name: str, values: pd.Series, labels: pd.Index) -> FloatArray:
    """Return ``values`` ordered like ``labels``, which must name the same assets."""
    if not isinstance(values, pd.Series):
        raise InvalidParameterError(f"{name} must be a pandas Series indexed by asset")
    if len(values) != len(labels) or set(values.index) != set(labels):
        raise InvalidParameterError(f"{name} and cov must cover exactly the same assets")
    array = values.reindex(labels).to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise InvalidParameterError(f"{name} must not contain NaN or infinite values")
    return array


def _check_bounds(bounds: Bounds | None, n_assets: int) -> Bounds | None:
    if bounds is None:
        return None
    low = require_finite("lower bound", bounds[0])
    high = require_finite("upper bound", bounds[1])
    if low > high:
        raise InvalidParameterError(f"lower bound {low} exceeds upper bound {high}")
    # Equal weights are feasible exactly when n * low <= 1 <= n * high.
    if not n_assets * low <= 1.0 <= n_assets * high:
        raise InvalidParameterError(
            f"bounds {bounds} make a fully invested portfolio of {n_assets} assets infeasible"
        )
    return low, high


def _as_series(weights: FloatArray, labels: pd.Index) -> pd.Series:
    return pd.Series(weights, index=labels, name="weight")


# ---------------------------------------------------------------------- closed forms


def min_variance_closed_form(cov: pd.DataFrame) -> pd.Series:
    """Global minimum-variance weights with no bounds: ``Sigma^-1 1 / (1' Sigma^-1 1)``.

    Minimizing ``w' Sigma w`` subject to ``1'w = 1`` gives the first-order
    condition ``2 Sigma w = lambda 1``, so ``w`` is proportional to
    ``Sigma^-1 1``; dividing by the sum enforces full investment.

    Args:
        cov: Positive definite covariance matrix labelled by asset.

    Returns:
        Weights indexed like ``cov``. Negative weights are short positions.
    """
    sigma = _covariance(cov)
    # Solving Sigma x = 1 is cheaper and more accurate than forming Sigma^-1.
    x = np.linalg.solve(sigma, np.ones(len(sigma)))
    return _as_series(x / x.sum(), cov.index)


def max_sharpe_closed_form(
    expected_returns: pd.Series, cov: pd.DataFrame, *, risk_free_rate: float = 0.0
) -> pd.Series:
    """Tangency (maximum-Sharpe) weights with no bounds.

    The Sharpe ratio ``(w'mu - rf) / sqrt(w' Sigma w)`` is maximized by
    ``w`` proportional to ``Sigma^-1 (mu - rf 1)``, normalized to sum to one.

    The normalizing sum ``1' Sigma^-1 (mu - rf 1)`` is positive exactly when the
    GMV portfolio earns more than ``rf``. If it is not, dividing by it flips the
    sign and the formula would return the *minimum*-Sharpe portfolio, so this
    raises instead.

    Raises:
        InvalidParameterError: If the GMV expected return does not exceed ``rf``.
    """
    sigma = _covariance(cov)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    risk_free_rate = require_finite("risk_free_rate", risk_free_rate)
    x = np.linalg.solve(sigma, mu - risk_free_rate)
    if x.sum() <= 0.0:
        raise InvalidParameterError(
            "no tangency portfolio: the minimum-variance portfolio does not earn more "
            "than the risk-free rate"
        )
    return _as_series(x / x.sum(), cov.index)


def efficient_return_closed_form(
    expected_returns: pd.Series, cov: pd.DataFrame, target_return: float
) -> pd.Series:
    """Minimum-variance weights for a target return, with no bounds (Merton 1972).

    With ``A = 1' Sigma^-1 1``, ``B = 1' Sigma^-1 mu``, ``C = mu' Sigma^-1 mu`` and
    ``D = AC - B^2``, the solution for target return ``m`` is::

        w = [(C - m B) Sigma^-1 1 + (m A - B) Sigma^-1 mu] / D

    and its variance is ``(A m^2 - 2 B m + C) / D``, a parabola in ``m``.

    Raises:
        InvalidParameterError: If all expected returns are equal (``D = 0``), in
            which case every portfolio has the same expected return.
    """
    sigma = _covariance(cov)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    m = require_finite("target_return", target_return)
    inv_ones = np.linalg.solve(sigma, np.ones(len(mu)))
    inv_mu = np.linalg.solve(sigma, mu)
    a, b, c = inv_ones.sum(), inv_mu.sum(), mu @ inv_mu
    d = a * c - b * b
    if d <= 1e-14 * a * c:
        raise InvalidParameterError("the efficient frontier is degenerate: all assets share mu")
    return _as_series(((c - m * b) * inv_ones + (m * a - b) * inv_mu) / d, cov.index)


# ------------------------------------------------------------------- numerical (SLSQP)


def _linear(kind: Literal["eq", "ineq"], matrix: FloatArray, rhs: float = 0.0) -> dict[str, Any]:
    """SLSQP constraint ``matrix @ x == rhs`` (``"eq"``) or ``matrix @ x >= rhs``."""
    return {"type": kind, "fun": lambda x: matrix @ x - rhs, "jac": lambda x: matrix}


def _slsqp(
    objective: Callable[[FloatArray], float],
    gradient: Callable[[FloatArray], FloatArray],
    x0: FloatArray,
    *,
    bounds: list[tuple[float | None, float | None]] | None,
    constraints: list[dict[str, Any]],
) -> FloatArray:
    """Run SciPy's SLSQP and turn a failure into a :class:`ConvergenceError`."""
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise ConvergenceError(f"SLSQP did not converge: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _min_variance_weights(
    sigma: FloatArray,
    bounds: Bounds | None,
    extra_constraints: list[dict[str, Any]] | None = None,
) -> FloatArray:
    """Minimize ``w' Sigma w`` over fully invested weights within ``bounds``."""
    n_assets = len(sigma)
    equal = np.full(n_assets, 1.0 / n_assets)  # feasible whenever the bounds are
    # SLSQP stops on an *absolute* change in the objective, so measure variance
    # relative to the equal-weight portfolio to make the tolerance unit-free.
    scale = 1.0 / float(equal @ sigma @ equal)
    weights = _slsqp(
        lambda w: scale * float(w @ sigma @ w),
        lambda w: 2.0 * scale * (sigma @ w),
        equal,
        bounds=None if bounds is None else [bounds] * n_assets,
        constraints=[_linear("eq", np.ones(n_assets), 1.0), *(extra_constraints or [])],
    )
    if bounds is not None:
        # SLSQP honours bounds only up to round-off (e.g. -1e-17 for long-only).
        weights = np.clip(weights, *bounds)
    return weights


def min_variance(cov: pd.DataFrame, *, bounds: Bounds | None = LONG_ONLY) -> pd.Series:
    """Global minimum-variance portfolio, optionally with weight bounds.

    Args:
        cov: Positive definite covariance matrix labelled by asset.
        bounds: ``(lower, upper)`` limit for every weight; the default is
            long-only, ``None`` allows unlimited long and short positions.

    Returns:
        Weights indexed like ``cov``, summing to one.

    Raises:
        InvalidParameterError: For invalid inputs or infeasible bounds.
        ConvergenceError: If the optimizer fails.
    """
    sigma = _covariance(cov)
    checked = _check_bounds(bounds, len(sigma))
    return _as_series(_min_variance_weights(sigma, checked), cov.index)


def _highest_return_weights(mu: FloatArray, bounds: Bounds) -> FloatArray:
    """Fully invested weights within ``bounds`` that maximize ``w'mu``.

    With the same box for every asset this is a fractional knapsack problem:
    start every weight at its lower bound, then spend the remaining budget on the
    assets with the highest expected return first, each up to its upper bound.
    """
    low, high = bounds
    weights = np.full(len(mu), low)
    budget = 1.0 - weights.sum()
    for i in np.argsort(mu)[::-1]:
        extra = min(high - low, budget)
        weights[i] += extra
        budget -= extra
    return weights


def max_sharpe(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    bounds: Bounds | None = LONG_ONLY,
) -> pd.Series:
    """Maximum-Sharpe (tangency) portfolio, optionally with weight bounds.

    The Sharpe ratio is not a concave function of the weights, so maximizing it
    directly is fragile. Instead this solves an equivalent convex quadratic
    program (Cornuejols & Tutuncu, *Optimization Methods in Finance*). With excess
    returns ``e = mu - rf`` and the change of variables ``kappa = 1 / (e'w)``,
    ``y = kappa * w``::

        minimize   y' Sigma y
        subject to e'y = 1,   sum(y) = kappa >= 0,   lower * kappa <= y_i <= upper * kappa

    For any portfolio with ``e'w > 0`` the objective equals
    ``w' Sigma w / (e'w)^2 = 1 / Sharpe(w)^2``, so minimizing it maximizes the
    Sharpe ratio, and ``w = y / kappa`` recovers the weights. Multiplying the
    bounds by ``kappa`` ("homogenizing" them) keeps every constraint linear.

    Args:
        expected_returns: Annualized expected returns, indexed like ``cov``.
        cov: Annualized, positive definite covariance matrix.
        risk_free_rate: Annualized risk-free rate.
        bounds: ``(lower, upper)`` limit for every weight (default long-only);
            ``None`` for no limits.

    Returns:
        Weights indexed like ``cov``, summing to one.

    Raises:
        InvalidParameterError: For invalid inputs or infeasible bounds, or if no
            asset (with ``bounds=None``) or no feasible portfolio (with bounds)
            has an expected return above the risk-free rate.
        ConvergenceError: If the optimizer fails or no tangency portfolio exists.
    """
    sigma = _covariance(cov)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    excess = mu - require_finite("risk_free_rate", risk_free_rate)
    n_assets = len(mu)
    checked = _check_bounds(bounds, n_assets)

    # Start from equal weights, or from the highest-return portfolio if equal
    # weights do not beat the risk-free rate; either maps to a feasible (y, kappa).
    start = np.full(n_assets, 1.0 / n_assets)
    if excess @ start <= 0.0:
        if checked is None:
            start = np.eye(n_assets)[np.argmax(excess)]
        else:
            start = _highest_return_weights(excess, checked)
    if excess @ start <= 0.0:
        raise InvalidParameterError(
            "no feasible portfolio has an expected return above the risk-free rate"
        )
    z0 = np.append(start, 1.0) / (excess @ start)  # z = (y_1, ..., y_n, kappa)

    def y_of(z: FloatArray) -> FloatArray:
        return z[:n_assets]

    constraints = [
        _linear("eq", np.append(excess, 0.0), 1.0),  # e'y = 1
        _linear("eq", np.append(np.ones(n_assets), -1.0)),  # sum(y) - kappa = 0
    ]
    if checked is not None:
        low, high = checked
        identity = np.eye(n_assets)
        column = np.ones((n_assets, 1))
        constraints.append(_linear("ineq", np.hstack([identity, -low * column])))  # y >= low k
        constraints.append(_linear("ineq", np.hstack([-identity, high * column])))  # y <= high k

    z = _slsqp(
        lambda z: float(y_of(z) @ sigma @ y_of(z)),
        lambda z: np.append(2.0 * (sigma @ y_of(z)), 0.0),
        z0,
        bounds=[(None, None)] * n_assets + [(0.0, None)],
        constraints=constraints,
    )
    kappa = z[-1]
    if kappa <= 1e-12:
        raise ConvergenceError("no tangency portfolio exists for these inputs")
    weights = y_of(z) / kappa
    if checked is not None:
        weights = np.clip(weights, *checked)  # remove round-off bound violations
    return _as_series(weights, cov.index)


def _return_range(mu: FloatArray, bounds: Bounds | None) -> tuple[float, float]:
    """Lowest and highest expected return of a fully invested portfolio within bounds."""
    if bounds is None:
        return -np.inf, np.inf
    lowest = float(_highest_return_weights(-mu, bounds) @ mu)
    highest = float(_highest_return_weights(mu, bounds) @ mu)
    return lowest, highest


def efficient_return(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    target_return: float,
    *,
    bounds: Bounds | None = LONG_ONLY,
) -> pd.Series:
    """Minimum-variance portfolio with a given expected return.

    Args:
        expected_returns: Annualized expected returns, indexed like ``cov``.
        cov: Annualized, positive definite covariance matrix.
        target_return: Required expected return ``w'mu``.
        bounds: ``(lower, upper)`` limit for every weight (default long-only);
            ``None`` for no limits.

    Returns:
        Weights indexed like ``cov``, summing to one.

    Raises:
        InvalidParameterError: If the target cannot be reached within the bounds.
        ConvergenceError: If the optimizer fails.
    """
    sigma = _covariance(cov)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    target = require_finite("target_return", target_return)
    checked = _check_bounds(bounds, len(sigma))
    lowest, highest = _return_range(mu, checked)
    slack = 1e-12 * max(1.0, abs(target))
    if not lowest - slack <= target <= highest + slack:
        raise InvalidParameterError(
            f"target return {target:.4g} is outside the attainable range "
            f"[{lowest:.4g}, {highest:.4g}] for these bounds"
        )
    weights = _min_variance_weights(sigma, checked, [_linear("eq", mu, target)])
    return _as_series(weights, cov.index)


def portfolio_performance(
    weights: pd.Series,
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> PortfolioPerformance:
    """Expected return, volatility and Sharpe ratio of a portfolio."""
    sigma = validate_covariance(cov)
    w = _aligned("weights", weights, cov.index)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    expected = float(w @ mu)
    volatility = float(np.sqrt(w @ sigma @ w))
    sharpe = (expected - risk_free_rate) / volatility if volatility > 0 else float("nan")
    return PortfolioPerformance(expected_return=expected, volatility=volatility, sharpe=sharpe)


def efficient_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    *,
    n_points: int = 20,
    bounds: Bounds | None = LONG_ONLY,
    risk_free_rate: float = 0.0,
    max_return: float | None = None,
) -> pd.DataFrame:
    """Sample the efficient frontier at evenly spaced target returns.

    Targets run from the return of the minimum-variance portfolio (the left tip
    of the frontier) up to ``max_return``. By default that is the highest return
    attainable within ``bounds`` (or the highest asset return when
    ``bounds=None``). At exactly the highest attainable return the feasible set
    shrinks to a single corner, where SLSQP's linearized subproblem can become
    numerically inconsistent, so the default top target stops a hair (one
    millionth of the return range) below it.

    Args:
        expected_returns: Annualized expected returns, indexed like ``cov``.
        cov: Annualized, positive definite covariance matrix.
        n_points: Number of frontier portfolios.
        bounds: ``(lower, upper)`` limit for every weight (default long-only);
            ``None`` for no limits.
        risk_free_rate: Used for the ``sharpe`` column.
        max_return: Highest target return; see above for the default.

    Returns:
        One row per point with columns ``expected_return``, ``volatility`` and
        ``sharpe``, followed by one weight column per asset.

    Raises:
        InvalidParameterError: For invalid inputs, or if ``max_return`` does not
            exceed the minimum-variance return (with bounds, the default can hit
            this when the minimum-variance portfolio already has the highest
            attainable return, so the frontier is a single point).
    """
    n_points = require_int("n_points", n_points, minimum=2)
    sigma = _covariance(cov)
    mu = _aligned("expected_returns", expected_returns, cov.index)
    checked = _check_bounds(bounds, len(sigma))
    gmv_return = float(_min_variance_weights(sigma, checked) @ mu)
    if max_return is None:
        top = float(np.max(mu)) if checked is None else _return_range(mu, checked)[1]
        max_return = top - 1e-6 * (top - gmv_return)
    if max_return <= gmv_return:
        raise InvalidParameterError(
            f"max_return {max_return:.6g} must exceed the expected return of the "
            f"minimum-variance portfolio, {gmv_return:.6g}"
        )
    rows: list[dict[str, float]] = []
    for target in np.linspace(gmv_return, max_return, n_points):
        weights = efficient_return(expected_returns, cov, float(target), bounds=checked)
        perf = portfolio_performance(weights, expected_returns, cov, risk_free_rate=risk_free_rate)
        rows.append(
            {
                "expected_return": perf.expected_return,
                "volatility": perf.volatility,
                "sharpe": perf.sharpe,
                **{str(asset): float(w) for asset, w in weights.items()},
            }
        )
    return pd.DataFrame(rows)

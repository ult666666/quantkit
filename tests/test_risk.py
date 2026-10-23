"""VaR and CVaR against hand calculations and Gaussian closed forms."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantkit.exceptions import InvalidParameterError
from quantkit.portfolio import (
    historical_cvar,
    historical_var,
    parametric_cvar,
    parametric_var,
    portfolio_volatility,
)

TEN_RETURNS = [0.03, -0.05, 0.01, -0.02, 0.04, -0.04, 0.0, 0.02, -0.01, -0.03]


def test_historical_var_and_cvar_by_hand() -> None:
    # 80%: the tail is ceil(10 * 0.2) = 2 worst returns, -5% and -4%.
    assert historical_var(TEN_RETURNS, level=0.80) == pytest.approx(0.04)
    assert historical_cvar(TEN_RETURNS, level=0.80) == pytest.approx(0.045)
    # 95%: ceil(0.5) = 1, so only the single worst return.
    assert historical_var(TEN_RETURNS, level=0.95) == pytest.approx(0.05)
    assert historical_cvar(TEN_RETURNS, level=0.95) == pytest.approx(0.05)


def test_tail_size_is_exact_despite_floating_point() -> None:
    # (1 - 0.99) * 500 is 5.000000000000004 in floating point; the tail must
    # still be the 5 worst of 500 returns.
    returns = np.random.default_rng(0).permutation(np.arange(500) / 1000.0 - 0.25)
    assert historical_var(returns, level=0.99) == pytest.approx(0.25 - 0.004)
    assert historical_cvar(returns, level=0.99) == pytest.approx(0.25 - 0.002)


def test_parametric_formulas_with_known_mean_and_volatility() -> None:
    returns = np.array([0.02, -0.02, 0.02, -0.02, 0.01, -0.01])  # mean 0
    sigma = float(np.std(returns, ddof=1))
    assert parametric_var(returns, level=0.95) == pytest.approx(1.6448536269514722 * sigma)
    # phi(z_0.95) / 0.05 = 2.0627128075074...
    assert parametric_cvar(returns, level=0.95) == pytest.approx(2.062712807507416 * sigma)
    shifted = returns + 0.001
    assert parametric_var(shifted) == pytest.approx(parametric_var(returns) - 0.001)


def test_historical_converges_to_parametric_for_normal_returns() -> None:
    returns = np.random.default_rng(42).normal(0.0005, 0.01, 400_000)
    for level in (0.95, 0.99):
        assert historical_var(returns, level=level) == pytest.approx(
            parametric_var(returns, level=level), rel=0.01
        )
        assert historical_cvar(returns, level=level) == pytest.approx(
            parametric_cvar(returns, level=level), rel=0.02
        )


@given(
    returns=st.lists(st.floats(-0.5, 0.5), min_size=2, max_size=300),
    level=st.floats(0.5, 0.999),
)
def test_cvar_is_at_least_var(returns: list[float], level: float) -> None:
    assert historical_cvar(returns, level=level) >= historical_var(returns, level=level) - 1e-12
    if np.std(returns) > 0:
        assert parametric_cvar(returns, level=level) >= parametric_var(returns, level=level)


def test_accepts_series_and_arrays_alike() -> None:
    series = pd.Series(TEN_RETURNS)
    assert historical_var(series) == historical_var(np.array(TEN_RETURNS))


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1, math.nan])
def test_rejects_invalid_levels(level: float) -> None:
    with pytest.raises(InvalidParameterError, match="level"):
        historical_var(TEN_RETURNS, level=level)
    with pytest.raises(InvalidParameterError, match="level"):
        parametric_var(TEN_RETURNS, level=level)


def test_rejects_invalid_returns() -> None:
    with pytest.raises(InvalidParameterError, match="at least 2"):
        historical_var([0.01])
    with pytest.raises(InvalidParameterError, match="NaN"):
        parametric_cvar([0.01, math.nan, 0.02])
    with pytest.raises(InvalidParameterError, match="one-dimensional"):
        historical_cvar(np.zeros((3, 2)))
    with pytest.raises(InvalidParameterError, match="numbers"):
        historical_var(["a", "b"])


def test_portfolio_volatility_by_hand() -> None:
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["X", "Y"], columns=["X", "Y"])
    weights = pd.Series({"Y": 0.25, "X": 0.75})  # order does not matter
    expected = math.sqrt(0.75**2 * 0.04 + 0.25**2 * 0.09 + 2 * 0.75 * 0.25 * 0.006)
    assert portfolio_volatility(weights, cov) == pytest.approx(expected)
    with pytest.raises(InvalidParameterError, match="covering"):
        portfolio_volatility(pd.Series({"X": 1.0}), cov)

"""Performance metrics against hand-computed series."""

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import (
    annualized_turnover,
    annualized_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    hit_rate,
    max_drawdown,
    max_drawdown_duration,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from quantkit.exceptions import InvalidParameterError

# Wealth path 1 -> 1.1 -> 0.88 -> 0.924 -> 1.2012: a 20% drawdown from the 1.1
# peak that lasts two periods before a new high.
RETURNS = [0.10, -0.20, 0.05, 0.30]


def test_compounding_metrics_by_hand() -> None:
    assert total_return(RETURNS) == pytest.approx(0.2012)
    # Four quarterly returns span exactly one year.
    assert cagr(RETURNS, periods_per_year=4) == pytest.approx(0.2012)
    # Over two years (two periods per year) the same growth is sqrt(1.2012) - 1.
    assert cagr(RETURNS, periods_per_year=2) == pytest.approx(math.sqrt(1.2012) - 1)


def test_drawdown_metrics_by_hand() -> None:
    assert max_drawdown(RETURNS) == pytest.approx(0.2)
    assert max_drawdown_duration(RETURNS) == 2
    series = drawdown_series(pd.Series(RETURNS))
    np.testing.assert_allclose(series, [0.0, -0.2, 0.924 / 1.1 - 1, 0.0], atol=1e-12)
    assert calmar_ratio(RETURNS, periods_per_year=4) == pytest.approx(0.2012 / 0.2)


def test_drawdown_counts_from_the_initial_capital() -> None:
    # Wealth 0.9 then 0.945: never above the starting value of 1.
    assert max_drawdown([-0.10, 0.05]) == pytest.approx(0.10)
    assert max_drawdown_duration([-0.10, 0.05]) == 2


def test_unrecovered_drawdown_lasts_until_the_end() -> None:
    assert max_drawdown_duration([0.1, -0.1, 0.01, 0.01, 0.01]) == 4


def test_volatility_sharpe_and_sortino_by_hand() -> None:
    r = [0.01, 0.02, -0.01, 0.0]
    # mean 0.005; deviations (0.005, 0.015, -0.015, -0.005) give a sample
    # variance of 5e-4 / 3.
    std = math.sqrt(5e-4 / 3)
    assert annualized_volatility(r) == pytest.approx(std * math.sqrt(252))
    assert sharpe_ratio(r) == pytest.approx(0.005 / std * math.sqrt(252))
    # Downside deviation: sqrt(mean([0, 0, 0.01^2, 0])) = 0.005.
    assert sortino_ratio(r) == pytest.approx(0.005 / 0.005 * math.sqrt(252))


def test_risk_free_rate_is_compounded_down_to_each_period() -> None:
    r = np.full(252, 0.0002) + np.tile([0.001, -0.001], 126)
    rf_daily = 1.03 ** (1 / 252) - 1
    expected = (r - rf_daily).mean() / (r - rf_daily).std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r, risk_free_rate=0.03) == pytest.approx(expected)


def test_hit_rate_ignores_flat_periods() -> None:
    assert hit_rate([0.01, 0.0, -0.02, 0.03, 0.0]) == pytest.approx(2 / 3)
    assert math.isnan(hit_rate([0.0, 0.0]))


def test_degenerate_ratios_are_nan() -> None:
    flat = [0.001] * 10
    assert math.isnan(sharpe_ratio(flat))  # zero volatility
    assert math.isnan(sortino_ratio(flat))  # no downside
    assert math.isnan(calmar_ratio(flat))  # no drawdown


def test_annualized_turnover() -> None:
    assert annualized_turnover([1.0, 0.0, 0.0, 0.0], periods_per_year=4) == pytest.approx(1.0)
    with pytest.raises(InvalidParameterError, match="non-negative"):
        annualized_turnover([-0.1, 0.2])


def test_summary_collects_the_individual_metrics() -> None:
    summary = performance_summary(RETURNS, turnover=[1.0, 0.0, 0.0, 0.0], periods_per_year=4)
    assert summary["cagr"] == pytest.approx(cagr(RETURNS, periods_per_year=4))
    assert summary["max_drawdown"] == pytest.approx(0.2)
    assert summary["max_drawdown_periods"] == 2
    assert summary["annual_turnover"] == pytest.approx(1.0)
    assert "annual_turnover" not in performance_summary(RETURNS).index


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidParameterError, match="-1"):
        total_return([0.1, -1.5])
    with pytest.raises(InvalidParameterError, match="at least 2"):
        sharpe_ratio([0.01])
    with pytest.raises(InvalidParameterError, match="risk_free_rate"):
        sharpe_ratio(RETURNS, risk_free_rate=-1.0)
    with pytest.raises(InvalidParameterError, match="Series"):
        drawdown_series(cast("pd.Series", np.array(RETURNS)))  # wrong type on purpose

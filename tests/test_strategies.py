"""Example strategies: schedules and signals on hand-built price paths."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import (
    BuyAndHold,
    EqualWeight,
    Frequency,
    InverseVolatility,
    MovingAverageCrossover,
    TimeSeriesMomentum,
    rebalance_days,
    run_backtest,
)
from quantkit.exceptions import InvalidParameterError


def frame(columns: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range(start, periods=n))


def test_rebalance_days_flag_the_first_trading_day_of_each_month() -> None:
    index = pd.bdate_range("2024-01-29", "2024-03-05")
    flagged = index[rebalance_days(index, "M")]
    # Day 0, then Feb 1 (Thursday) and Mar 1 (Friday).
    assert list(flagged) == [pd.Timestamp(d) for d in ("2024-01-29", "2024-02-01", "2024-03-01")]
    assert rebalance_days(index, "D").all()
    weekly = index[rebalance_days(index, "W")]
    assert (weekly[1:].day_of_week == 0).all()  # Mondays after the first day


def test_rebalance_days_reject_unknown_frequencies() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    with pytest.raises(InvalidParameterError, match="frequency"):
        rebalance_days(index, cast("Frequency", "H"))


def test_buy_and_hold_trades_once() -> None:
    prices = frame({"X": [1.0, 2.0, 3.0], "Y": [1.0, 1.0, 1.0]})
    targets = BuyAndHold({"X": 0.7})(prices)
    np.testing.assert_allclose(targets.iloc[0], [0.7, 0.0])
    assert targets.iloc[1:].isna().all().all()
    with pytest.raises(InvalidParameterError, match="unknown tickers"):
        BuyAndHold({"Z": 1.0})(prices)


def test_equal_weight_drifts_between_rebalances_and_resets_on_them() -> None:
    days = pd.bdate_range("2024-01-25", "2024-02-08")
    x = 100.0 * 1.01 ** np.arange(len(days))  # X rises 1% a day, Y is flat
    prices = pd.DataFrame({"X": x, "Y": 100.0}, index=days)
    weights = run_backtest(prices, EqualWeight(rebalance="M")).weights
    first_feb = int(days.searchsorted(pd.Timestamp("2024-02-01")))
    # Before the February rebalance X's weight has drifted above one half...
    assert weights["X"].iloc[first_feb] > 0.5
    # ...and the day after it the weights are back to exactly 50/50.
    np.testing.assert_allclose(weights.iloc[first_feb + 1], [0.5, 0.5])


def test_moving_average_crossover_follows_the_trend_and_trades_only_on_flips() -> None:
    up_then_down = [10.0, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10]
    prices = frame({"X": up_then_down, "Y": [5.0] * 11})
    targets = MovingAverageCrossover(fast=2, slow=3)(prices)
    # SMA2 > SMA3 from day 2 (first day both exist) while rising; the downturn
    # pulls SMA2 below SMA3 on day 7. Y is flat, so never in a trend.
    traded = targets.dropna()
    assert list(traded.index) == [prices.index[i] for i in (0, 2, 7)]
    np.testing.assert_allclose(traded.to_numpy(), [[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(("fast", "slow"), [(5, 5), (10, 5), (0, 5)])
def test_moving_average_crossover_rejects_bad_windows(fast: int, slow: int) -> None:
    with pytest.raises(InvalidParameterError):
        MovingAverageCrossover(fast=fast, slow=slow)


@pytest.mark.parametrize(("long_only", "expected_down"), [(False, -0.5), (True, 0.0)])
def test_time_series_momentum_follows_each_assets_trailing_return(
    long_only: bool, expected_down: float
) -> None:
    n = 30
    prices = frame({"UP": list(np.linspace(10, 20, n)), "DOWN": list(np.linspace(20, 10, n))})
    targets = TimeSeriesMomentum(lookback=5, rebalance="D", long_only=long_only)(prices)
    assert (targets.iloc[:5] == 0.0).all().all()  # no trailing return yet: cash
    np.testing.assert_allclose(targets["UP"].iloc[5:], 0.5)
    np.testing.assert_allclose(targets["DOWN"].iloc[5:], expected_down)


def test_inverse_volatility_weights_are_inversely_proportional_to_volatility() -> None:
    # Returns alternate +a, -a, so each asset's return volatility is proportional to a.
    n = 40
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    prices = pd.DataFrame(
        {
            "CALM": 100.0 * np.cumprod(1.0 + 0.01 * signs),
            "WILD": 100.0 * np.cumprod(1.0 + 0.03 * signs),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    targets = InverseVolatility(window=10, rebalance="D")(prices).dropna()
    assert targets.index[0] == prices.index[10]  # first full 10-return window
    np.testing.assert_allclose(targets["CALM"], 0.75)  # (1/0.01) / (1/0.01 + 1/0.03)
    np.testing.assert_allclose(targets["WILD"], 0.25)


def test_calendar_strategies_need_a_datetime_index() -> None:
    prices = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
    with pytest.raises(InvalidParameterError, match="DatetimeIndex"):
        EqualWeight()(prices)


def test_window_parameters_are_validated() -> None:
    with pytest.raises(InvalidParameterError, match="lookback"):
        TimeSeriesMomentum(lookback=0)
    with pytest.raises(InvalidParameterError, match="window"):
        InverseVolatility(window=1)

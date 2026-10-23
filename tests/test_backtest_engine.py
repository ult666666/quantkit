"""Backtest engine: exact accounting, costs, and the no-look-ahead contract."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import (
    BuyAndHold,
    EqualWeight,
    InverseVolatility,
    MovingAverageCrossover,
    Strategy,
    TimeSeriesMomentum,
    backtest_weights,
    check_no_lookahead,
    run_backtest,
)
from quantkit.data import simulate_gbm_prices
from quantkit.exceptions import DataValidationError, InvalidParameterError

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return simulate_gbm_prices(
        tickers=TICKERS,
        n_days=600,
        drift=[0.05, 0.08, 0.02],
        vol=[0.15, 0.30, 0.10],
        correlation=0.3,
        seed=1,
    )


def reference_backtest(
    prices: pd.DataFrame, targets: pd.DataFrame, cost_bps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plain day-by-day loop implementing the documented contract.

    It shares no code with the vectorized engine, so agreement between the two
    is an independent check of the engine's accounting.
    """
    p = prices.to_numpy()
    w_target = targets.to_numpy()
    n_days, n_assets = p.shape
    held = np.zeros(n_assets)  # weights carried into the current day
    equity, net_returns, held_log, turnover_log = [], [], [], []
    wealth = 1.0
    for t in range(n_days):
        asset_returns = p[t] / p[t - 1] - 1.0 if t > 0 else np.zeros(n_assets)
        gross = float(held @ asset_returns)
        drifted = held * (1.0 + asset_returns) / (1.0 + gross)
        if np.isnan(w_target[t]).all():
            post, turnover = drifted, 0.0
        else:
            post = w_target[t]
            turnover = float(np.abs(post - drifted).sum())
        net = (1.0 + gross) * (1.0 - cost_bps / 10_000 * turnover) - 1.0
        wealth *= 1.0 + net
        held_log.append(held)
        equity.append(wealth)
        net_returns.append(net)
        turnover_log.append(turnover)
        held = post
    return np.array(equity), np.array(net_returns), np.array(held_log), np.array(turnover_log)


def random_targets(prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random target weights, including cash, shorts, leverage and no-trade days."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.3, 0.8, size=prices.shape)
    values[rng.random(len(prices)) < 0.7] = np.nan  # trade on ~30% of days
    return pd.DataFrame(values, index=prices.index, columns=prices.columns)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("cost_bps", [0.0, 25.0])
def test_vectorized_engine_matches_reference_loop(
    prices: pd.DataFrame, seed: int, cost_bps: float
) -> None:
    targets = random_targets(prices, seed)
    result = backtest_weights(prices, targets, cost_bps=cost_bps)
    equity, net, held, turnover = reference_backtest(prices, targets, cost_bps)
    np.testing.assert_allclose(result.equity, equity, rtol=1e-10)
    np.testing.assert_allclose(result.returns, net, atol=1e-12)
    np.testing.assert_allclose(result.weights, held, atol=1e-12)
    np.testing.assert_allclose(result.turnover, turnover, atol=1e-12)


def test_buy_and_hold_equals_the_asset_return(prices: pd.DataFrame) -> None:
    single = prices[["BBB"]]
    result = run_backtest(single, BuyAndHold())
    np.testing.assert_allclose(result.equity, single["BBB"] / single["BBB"].iloc[0], rtol=1e-12)


def test_weighted_buy_and_hold_tracks_the_initial_share_counts(prices: pd.DataFrame) -> None:
    weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    result = run_backtest(prices, BuyAndHold(weights), initial_capital=1_000.0)
    growth = prices / prices.iloc[0]
    expected = 1_000.0 * (growth * pd.Series(weights)).sum(axis=1)
    np.testing.assert_allclose(result.equity, expected, rtol=1e-12)
    # No trading after day 0, so the only turnover is the initial purchase.
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert (result.turnover.iloc[1:] == 0.0).all()


def test_costs_reduce_wealth_by_exactly_the_computed_amount(prices: pd.DataFrame) -> None:
    strategy = EqualWeight(rebalance="W")
    free = run_backtest(prices, strategy, cost_bps=0.0)
    costly = run_backtest(prices, strategy, cost_bps=20.0)
    # Costs scale wealth down proportionally and do not change any weight, so
    # turnover is identical and wealth shrinks by prod(1 - c * turnover).
    pd.testing.assert_series_equal(free.turnover, costly.turnover)
    expected = free.equity * (1.0 - 0.002 * free.turnover).cumprod()
    np.testing.assert_allclose(costly.equity, expected, rtol=1e-12)
    np.testing.assert_allclose(costly.costs, 0.002 * free.turnover, rtol=1e-12)


def test_single_asset_cost_by_hand(prices: pd.DataFrame) -> None:
    single = prices[["AAA"]]
    result = run_backtest(single, BuyAndHold(), cost_bps=10.0)
    # One purchase of 100% of wealth at 10 bps.
    expected = (1.0 - 0.001) * single["AAA"].iloc[-1] / single["AAA"].iloc[0]
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert result.returns.iloc[0] == pytest.approx(-0.001)


def test_two_day_example_by_hand() -> None:
    prices = pd.DataFrame(
        {"X": [100.0, 110.0, 99.0], "Y": [50.0, 50.0, 55.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    targets = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)  # daily 50/50
    result = backtest_weights(prices, targets, cost_bps=10.0)
    # Day 0: buy 50/50 from cash, turnover 1, cost 0.1%.
    # Day 1: gross = 0.5 * 10% = 5%; drifted weights (0.55/1.05, 0.5/1.05);
    #        rebalancing back trades 2 * 0.025/1.05 = 0.05/1.05 of wealth.
    # Day 2: gross = 0.5 * (-10%) + 0.5 * 10% = 0; drifted (0.45, 0.55), trade 0.1.
    turnover_1 = 0.05 / 1.05
    expected_equity = [
        0.999,
        0.999 * 1.05 * (1 - 0.001 * turnover_1),
        0.999 * 1.05 * (1 - 0.001 * turnover_1) * (1 - 0.001 * 0.1),
    ]
    np.testing.assert_allclose(result.equity, expected_equity, rtol=1e-12)
    np.testing.assert_allclose(result.turnover, [1.0, turnover_1, 0.1], rtol=1e-12)
    np.testing.assert_allclose(result.weights.iloc[0], [0.0, 0.0])
    np.testing.assert_allclose(result.weights.iloc[1:], 0.5)


# ------------------------------------------------------------------ no look-ahead

STRATEGIES: list[Strategy] = [
    BuyAndHold(),
    EqualWeight(rebalance="M"),
    MovingAverageCrossover(fast=10, slow=40),
    TimeSeriesMomentum(lookback=60, rebalance="W"),
    TimeSeriesMomentum(lookback=60, long_only=True),
    InverseVolatility(window=20, rebalance="W"),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("cutoff", [45, 150, 300, 598])
def test_perturbing_later_prices_does_not_change_earlier_weights(
    prices: pd.DataFrame, strategy: Strategy, cutoff: int
) -> None:
    # Rescale every later price by an independent random factor. (+/-10% keeps the
    # long-short strategies from being wiped out by the artificial volatility.)
    rng = np.random.default_rng(cutoff)
    perturbed = prices.copy()
    later = perturbed.iloc[cutoff + 1 :]
    perturbed.iloc[cutoff + 1 :] = later.to_numpy() * rng.uniform(0.9, 1.1, later.shape)

    original = run_backtest(prices, strategy, cost_bps=10.0)
    shocked = run_backtest(perturbed, strategy, cost_bps=10.0)

    # Weights in effect on days 0..t+1 were all decided by the close of day t.
    pd.testing.assert_frame_equal(
        original.weights.iloc[: cutoff + 2], shocked.weights.iloc[: cutoff + 2]
    )
    # Wealth, turnover and costs through day t cannot depend on later prices.
    pd.testing.assert_series_equal(
        original.equity.iloc[: cutoff + 1], shocked.equity.iloc[: cutoff + 1]
    )
    pd.testing.assert_series_equal(
        original.turnover.iloc[: cutoff + 1], shocked.turnover.iloc[: cutoff + 1]
    )


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: type(s).__name__)
def test_built_in_strategies_pass_the_lookahead_checker(
    prices: pd.DataFrame, strategy: Strategy
) -> None:
    check_no_lookahead(strategy, prices, cutoffs=10)


def peeking_strategy(prices: pd.DataFrame) -> pd.DataFrame:
    """Deliberately broken: buys whatever will rise tomorrow (uses shift(-1))."""
    rises_tomorrow = prices.shift(-1) > prices
    return rises_tomorrow.astype(float) / prices.shape[1]


def test_lookahead_checks_catch_a_peeking_strategy(prices: pd.DataFrame) -> None:
    # Negative control: the same checks must fail for a strategy that cheats.
    with pytest.raises(InvalidParameterError, match="look-ahead"):
        check_no_lookahead(peeking_strategy, prices)
    # Reverse every asset's move from day 300 to day 301 by rescaling all later
    # prices; a strategy that peeks at day 301 must then change its day-300 weights.
    went_up = (prices.iloc[301] > prices.iloc[300]).to_numpy()
    perturbed = prices.copy()
    perturbed.iloc[301:] = prices.iloc[301:].to_numpy() * np.where(went_up, 0.5, 2.0)
    original = run_backtest(prices, peeking_strategy)
    shocked = run_backtest(perturbed, peeking_strategy)
    assert (original.weights.iloc[301] != shocked.weights.iloc[301]).all()
    # And cheating pays: knowing tomorrow's moves multiplies wealth many times over.
    assert original.equity.iloc[-1] > 10.0


def test_weights_take_effect_the_day_after_the_decision(prices: pd.DataFrame) -> None:
    result = run_backtest(prices, EqualWeight(rebalance="D"))
    assert (result.weights.iloc[0] == 0.0).all()
    np.testing.assert_allclose(result.weights.iloc[1:], 1.0 / 3.0)
    # Day 1 earns exactly the equal-weighted return of day 1.
    day1 = (prices.iloc[1] / prices.iloc[0] - 1.0).mean()
    assert result.returns.iloc[1] == pytest.approx(day1)


def test_summary_reports_turnover(prices: pd.DataFrame) -> None:
    summary = run_backtest(prices, EqualWeight(), cost_bps=5.0).summary()
    assert {"cagr", "sharpe", "max_drawdown", "annual_turnover"} <= set(summary.index)
    assert summary["annual_turnover"] > 0


# ------------------------------------------------------------------- validation


def test_rejects_mismatched_target_tables(prices: pd.DataFrame) -> None:
    with pytest.raises(InvalidParameterError, match="same dates and tickers"):
        backtest_weights(prices, pd.DataFrame(0.1, index=prices.index, columns=["AAA", "BBB"]))
    with pytest.raises(InvalidParameterError, match="DataFrame"):
        run_backtest(prices, returns_a_series)


def returns_a_series(prices: pd.DataFrame) -> pd.DataFrame:
    return cast("pd.DataFrame", prices.iloc[:, 0])  # wrong type on purpose


def test_rejects_partial_rows_and_infinite_weights(prices: pd.DataFrame) -> None:
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[5, 0] = 0.5
    with pytest.raises(InvalidParameterError, match="partially specified"):
        backtest_weights(prices, targets)
    targets.iloc[5] = [np.inf, 0.0, 0.0]
    with pytest.raises(InvalidParameterError, match="infinite"):
        backtest_weights(prices, targets)
    with pytest.raises(InvalidParameterError, match="numeric"):
        backtest_weights(prices, targets.astype(object).fillna("x"))


def test_rejects_invalid_costs_and_capital(prices: pd.DataFrame) -> None:
    with pytest.raises(InvalidParameterError, match="cost_bps"):
        run_backtest(prices, BuyAndHold(), cost_bps=-1.0)
    with pytest.raises(InvalidParameterError, match="initial_capital"):
        run_backtest(prices, BuyAndHold(), initial_capital=0.0)
    with pytest.raises(InvalidParameterError, match="costs exceed"):
        run_backtest(prices, BuyAndHold(), cost_bps=10_000.0)


def test_rejects_prices_without_sorted_dates(prices: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        run_backtest(prices.reset_index(drop=True), BuyAndHold())
    with pytest.raises(DataValidationError, match="sorted"):
        run_backtest(prices.iloc[::-1], BuyAndHold())


def test_detects_a_wiped_out_leveraged_portfolio() -> None:
    prices = pd.DataFrame(
        {"X": [100.0, 100.0, 40.0]}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    targets = pd.DataFrame({"X": [2.0, np.nan, np.nan]}, index=prices.index)  # 2x leverage
    with pytest.raises(InvalidParameterError, match="lost all of its value on 2024-01-03"):
        backtest_weights(prices, targets)

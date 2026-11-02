"""Rolling VaR/ES forecasts: single-window formulas, independent loops, no look-ahead."""

import math
from collections.abc import Callable
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import t as student_t

from quantkit.exceptions import InvalidParameterError
from quantkit.portfolio import historical_cvar, historical_var, parametric_cvar, parametric_var
from quantkit.var import METHODS, Method, RiskForecast, ewma_weights, rolling_var_es

WINDOW = 60
Forecaster = Callable[[pd.Series], pd.DataFrame]


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    """SYNTHETIC daily returns: Student-t(5) shocks with slowly varying volatility."""
    rng = np.random.default_rng(7)
    n = 400
    volatility = 0.01 * np.exp(0.5 * np.sin(np.arange(n) / 40.0))
    shocks = rng.standard_t(5, n) * math.sqrt(3.0 / 5.0)
    return pd.Series(volatility * shocks, index=pd.bdate_range("2021-01-01", periods=n))


def windows_before(r: np.ndarray, first: int, window: int) -> list[np.ndarray]:
    """For each day t >= first, the returns r[t - window], ..., r[t - 1]."""
    return [r[t - window : t] for t in range(first, len(r))]


@pytest.mark.parametrize("level", [0.95, 0.975, 0.99])
def test_historical_matches_the_single_window_functions(returns: pd.Series, level: float) -> None:
    forecast = rolling_var_es(returns, method="historical", level=level, window=WINDOW)
    r = returns.to_numpy()
    windows = windows_before(r, WINDOW, WINDOW)
    # np.partition (rolling) and np.sort (single window) must pick the same returns.
    expected_var = [historical_var(w, level=level) for w in windows]
    expected_es = [historical_cvar(w, level=level) for w in windows]
    np.testing.assert_array_equal(forecast.var, expected_var)
    np.testing.assert_allclose(forecast.es, expected_es, rtol=1e-14)
    assert forecast.var.index.equals(returns.index[WINDOW:])
    pd.testing.assert_series_equal(forecast.returns, returns.iloc[WINDOW:], check_names=False)


@pytest.mark.parametrize("level", [0.95, 0.99])
def test_gaussian_matches_the_single_window_functions(returns: pd.Series, level: float) -> None:
    forecast = rolling_var_es(returns, method="gaussian", level=level, window=WINDOW)
    windows = windows_before(returns.to_numpy(), WINDOW, WINDOW)
    np.testing.assert_allclose(
        forecast.var, [parametric_var(w, level=level) for w in windows], rtol=1e-12
    )
    np.testing.assert_allclose(
        forecast.es, [parametric_cvar(w, level=level) for w in windows], rtol=1e-12
    )


def test_ewma_by_hand() -> None:
    # Weights for lags 2, 1, 0 are 0.25, 0.5, 1 before normalizing by their sum 1.75.
    returns = [0.01, -0.02, 0.03, 0.0]
    forecast = rolling_var_es(returns, method="ewma", level=0.99, window=3, decay=0.5)
    sigma = math.sqrt((0.25 * 0.01**2 + 0.5 * 0.02**2 + 1.0 * 0.03**2) / 1.75)
    assert forecast.scale.iloc[0] == pytest.approx(sigma, rel=1e-14)
    z = NormalDist().inv_cdf(0.99)
    assert forecast.var.iloc[0] == pytest.approx(z * sigma, rel=1e-14)
    assert forecast.es.iloc[0] == pytest.approx(sigma * NormalDist().pdf(z) / 0.01, rel=1e-14)
    np.testing.assert_allclose(ewma_weights(3, 0.5), [0.25 / 1.75, 0.5 / 1.75, 1.0 / 1.75])


def test_ewma_matches_the_riskmetrics_recursion(returns: pd.Series) -> None:
    """The RiskMetrics update s2 <- decay * s2 + (1 - decay) * r^2, run over each window."""
    decay = 0.94
    forecast = rolling_var_es(returns, method="ewma", window=WINDOW, decay=decay)
    expected = []
    for window in windows_before(returns.to_numpy(), WINDOW, WINDOW):
        s2 = 0.0
        for r in window:
            s2 = decay * s2 + (1.0 - decay) * r * r
        # Started from zero, the recursion's weights sum to 1 - decay**WINDOW.
        expected.append(math.sqrt(s2 / (1.0 - decay**WINDOW)))
    np.testing.assert_allclose(forecast.scale, expected, rtol=1e-12)
    assert (forecast.location == 0.0).all()


def test_fhs_matches_a_volatility_weighted_loop(returns: pd.Series) -> None:
    """Hull and White's rescaling: r[s] * sigma[t] / sigma[s], then historical VaR."""
    window, level = 40, 0.975
    forecast = rolling_var_es(returns, method="fhs", level=level, window=window)
    r = returns.to_numpy()
    weights = ewma_weights(window)

    def sigma(day: int) -> float:
        return math.sqrt(float(np.square(r[day - window : day]) @ weights))

    expected_var, expected_es = [], []
    for t in range(2 * window, len(r)):
        rescaled = [r[s] * sigma(t) / sigma(s) for s in range(t - window, t)]
        expected_var.append(historical_var(rescaled, level=level))
        expected_es.append(historical_cvar(rescaled, level=level))
    np.testing.assert_allclose(forecast.var, expected_var, rtol=1e-12)
    np.testing.assert_allclose(forecast.es, expected_es, rtol=1e-12)
    assert forecast.var.index[0] == returns.index[2 * window]


@given(factor=st.floats(0.05, 20.0), method=st.sampled_from(METHODS))
def test_forecasts_scale_with_the_returns(factor: float, method: Method) -> None:
    returns = np.random.default_rng(3).normal(0.0005, 0.01, 200)
    base = rolling_var_es(returns, method=method, window=50)
    scaled = rolling_var_es(factor * returns, method=method, window=50)
    np.testing.assert_allclose(scaled.var, factor * base.var, rtol=1e-9)
    np.testing.assert_allclose(scaled.es, factor * base.es, rtol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_es_is_at_least_var(returns: pd.Series, method: Method) -> None:
    forecast = rolling_var_es(returns, method=method, level=0.975, window=WINDOW)
    assert (forecast.es >= forecast.var - 1e-15).all()
    assert (forecast.var > 0).all()


# --------------------------------------------------------------------- look-ahead


def as_forecaster(method: Method, window: int = WINDOW) -> Forecaster:
    def forecaster(r: pd.Series) -> pd.DataFrame:
        forecast = rolling_var_es(r, method=method, window=window)
        return pd.DataFrame({"var": forecast.var, "es": forecast.es})

    return forecaster


def peeking_forecaster(r: pd.Series) -> pd.DataFrame:
    """A deliberately broken forecaster whose window ends on the day it forecasts."""
    tomorrow = pd.Series(r.to_numpy()[1:], index=r.index[:-1])
    forecast = rolling_var_es(tomorrow, method="historical", window=WINDOW)
    return pd.DataFrame({"var": forecast.var, "es": forecast.es})


def lookahead_days(
    forecaster: Forecaster, returns: pd.Series, *, cutoffs: int = 6, seed: int = 0
) -> list[pd.Timestamp]:
    """Cutoff days whose forecast changed when only that day's and later returns changed.

    Like ``check_no_lookahead`` for strategies: the returns from each cutoff day
    on are scrambled, and every forecast up to and including the cutoff day
    (whose forecast is scored against the scrambled return) must stay identical.
    The cutoff day itself becomes a 50% loss, so a forecaster that peeks at it
    changes even when its VaR is a single order statistic, such as the worst
    return in the window.
    """
    rng = np.random.default_rng(seed)
    baseline = forecaster(returns)
    first = returns.index.get_loc(baseline.index[0])
    assert isinstance(first, int)
    violations = []
    for cutoff in sorted(rng.choice(np.arange(first, len(returns)), cutoffs, replace=False)):
        scrambled = returns.copy()
        later = scrambled.iloc[cutoff:]
        scrambled.iloc[cutoff:] = later.to_numpy() * rng.uniform(-3.0, 3.0, len(later))
        scrambled.iloc[cutoff] = -0.5
        rerun = forecaster(scrambled)
        through_cutoff = baseline.index <= returns.index[cutoff]
        if not np.array_equal(
            baseline.to_numpy()[through_cutoff], rerun.to_numpy()[through_cutoff]
        ):
            violations.append(returns.index[cutoff])
    return violations


@pytest.mark.parametrize("method", METHODS)
def test_no_forecast_uses_the_return_it_is_scored_against(
    returns: pd.Series, method: Method
) -> None:
    assert lookahead_days(as_forecaster(method), returns) == []


def test_lookahead_check_catches_a_peeking_forecaster(returns: pd.Series) -> None:
    # Negative control: the check must fail for a forecaster that sees r[t].
    assert len(lookahead_days(peeking_forecaster, returns)) == 6


@pytest.mark.parametrize("method", METHODS)
def test_forecasts_ignore_returns_older_than_their_history(
    returns: pd.Series, method: Method
) -> None:
    history = 2 * WINDOW if method == "fhs" else WINDOW
    baseline = as_forecaster(method)(returns)
    for day in (history + 5, len(returns) - 1):
        scrambled = returns.copy()
        scrambled.iloc[: day - history] = np.random.default_rng(day).normal(0, 0.05, day - history)
        rerun = as_forecaster(method)(scrambled)
        label = returns.index[day]
        np.testing.assert_array_equal(baseline.loc[label], rerun.loc[label])


# ------------------------------------------------- statistical checks on i.i.d. data


def exception_rate_and_error(
    method: Method, window: int, paths: np.ndarray
) -> tuple[float, float]:
    """Exception rate over independent paths, with a standard error from their spread."""
    rates = np.array(
        [rolling_var_es(path, method=method, window=window).exceptions.mean() for path in paths]
    )
    return float(rates.mean()), float(rates.std(ddof=1) / math.sqrt(len(rates)))


def test_historical_exception_probability_is_m_over_window_plus_one() -> None:
    # With i.i.d. continuous returns, r[t] is equally likely to take each of the
    # window + 1 ranks among r[t - window..t], so it falls below the m-th smallest
    # window return with probability m / (window + 1), whatever the distribution.
    # At 99% with 250 days, m = 3: the exception probability is 1.195%, not 1%.
    paths = np.random.default_rng(11).standard_t(3, (200, 1250))
    rate, error = exception_rate_and_error("historical", 250, paths)
    assert abs(rate - 3 / 251) < 4 * error
    assert abs(rate - 0.01) > 4 * error


def test_gaussian_exception_probability_includes_estimation_error() -> None:
    # For i.i.d. normal returns, (r[t] - mean) / (std * sqrt(1 + 1 / w)) has a
    # Student-t distribution with w - 1 degrees of freedom (the prediction
    # interval), so the exception probability is t_{w-1}(-z / sqrt(1 + 1 / w)).
    window, z = 30, NormalDist().inv_cdf(0.99)
    exact = float(student_t.cdf(-z / math.sqrt(1 + 1 / window), window - 1))  # 1.479%
    paths = np.random.default_rng(12).normal(0.0, 0.01, (400, 1030))
    rate, error = exception_rate_and_error("gaussian", window, paths)
    assert abs(rate - exact) < 4 * error
    assert abs(rate - 0.01) > 4 * error


@pytest.mark.parametrize("method", METHODS)
def test_simulated_returns_follow_each_days_forecast(returns: pd.Series, method: Method) -> None:
    level = 0.95
    forecast = rolling_var_es(returns, method=method, level=level, window=WINDOW).last(5)
    draws = forecast.simulate(100_000, seed=1)
    below = draws <= -forecast.var.to_numpy()
    if forecast.innovations is None:
        # Normal innovations: P(r <= -VaR) = 1 - level and E[r | r <= -VaR] = -ES.
        np.testing.assert_allclose(below.mean(axis=0), 1 - level, atol=0.0035)
        tail_mean = (draws * below).sum(axis=0) / below.sum(axis=0)
        np.testing.assert_allclose(tail_mean, -forecast.es, rtol=0.02)
    else:
        # Empirical innovations: every draw is location + scale * a pool value,
        # and the m = 3 smallest of the 60 pool values carry probability 3 / 60.
        pools = forecast.location.to_numpy() + forecast.scale.to_numpy() * forecast.innovations.T
        assert all(np.isin(draws[:, k], pools[:, k]).all() for k in range(len(forecast)))
        np.testing.assert_allclose(below.mean(axis=0), 3 / 60, atol=0.0035)
        tail_mean = (draws * below).sum(axis=0) / below.sum(axis=0)
        np.testing.assert_allclose(tail_mean, -forecast.es, rtol=0.01)


def test_simulation_is_reproducible(returns: pd.Series) -> None:
    forecast = rolling_var_es(returns, method="fhs", window=WINDOW)
    np.testing.assert_array_equal(forecast.simulate(3, seed=5), forecast.simulate(3, seed=5))
    assert forecast.simulate(3, seed=5).shape == (3, len(forecast))


# --------------------------------------------------------------------- the result


def test_last_keeps_the_most_recent_days(returns: pd.Series) -> None:
    forecast = rolling_var_es(returns, method="historical", window=WINDOW)
    tail = forecast.last(10)
    assert isinstance(tail, RiskForecast)
    assert len(tail) == 10
    pd.testing.assert_series_equal(tail.var, forecast.var.iloc[-10:])
    assert tail.innovations is not None
    assert forecast.innovations is not None
    np.testing.assert_array_equal(tail.innovations, forecast.innovations[-10:])
    pd.testing.assert_series_equal(tail.exceptions, forecast.exceptions.iloc[-10:])
    with pytest.raises(InvalidParameterError, match="at most"):
        forecast.last(len(forecast) + 1)


def test_exceptions_use_a_strict_inequality() -> None:
    # The fourth return equals the VaR forecast (the worst of the window): not an exception.
    forecast = rolling_var_es([-0.02, 0.01, 0.005, -0.02, -0.03], method="historical", window=3)
    assert list(forecast.var) == [0.02, 0.02]
    assert list(forecast.exceptions) == [False, True]


def test_arrays_get_a_range_index() -> None:
    forecast = rolling_var_es(np.linspace(-0.02, 0.02, 30), method="gaussian", window=10)
    assert list(forecast.var.index) == list(range(10, 30))


# --------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "garch"}, "method must be one of"),
        ({"level": 1.0}, "level"),
        ({"window": 1}, "window must be >= 2"),
        ({"decay": 1.0}, "decay"),
        ({"window": 400}, "needs more than 400 returns"),
        ({"method": "fhs", "window": 200}, "needs more than 400 returns"),
    ],
)
def test_rejects_invalid_arguments(
    returns: pd.Series, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        rolling_var_es(returns, **kwargs)  # type: ignore[arg-type]


def test_rejects_unusable_return_series(returns: pd.Series) -> None:
    with pytest.raises(InvalidParameterError, match="NaN"):
        rolling_var_es([0.01, math.nan, 0.02], window=2)
    with pytest.raises(InvalidParameterError, match="sorted by date"):
        rolling_var_es(returns.iloc[::-1], window=WINDOW)
    duplicated = pd.Series(returns.to_numpy()[:100], index=[0, 1] * 50)
    with pytest.raises(InvalidParameterError, match="duplicate"):
        rolling_var_es(duplicated, window=WINDOW)


def test_fhs_rejects_a_window_without_movement() -> None:
    flat = np.r_[np.full(30, 0.0), np.random.default_rng(0).normal(0, 0.01, 60)]
    with pytest.raises(InvalidParameterError, match=r"volatility forecast .* is zero"):
        rolling_var_es(flat, method="fhs", window=20)
    with pytest.raises(InvalidParameterError, match="decay"):
        ewma_weights(10, 0.0)

"""Acerbi-Szekely Z2: hand calculations, its zero mean under the null, invariances."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quantkit.exceptions import InvalidParameterError
from quantkit.var import acerbi_szekely_z2, es_backtest, rolling_var_es


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    """SYNTHETIC normal daily returns with a volatility that doubles halfway."""
    rng = np.random.default_rng(21)
    volatility = np.r_[np.full(300, 0.01), np.full(300, 0.02)]
    return pd.Series(volatility * rng.standard_normal(600))


def z2_of_paths(paths: np.ndarray, var: pd.Series, es: pd.Series, level: float) -> np.ndarray:
    """Z2 of every row, written out from the paper's formula for many paths at once."""
    exceptions = paths < -var.to_numpy()
    tail = np.where(exceptions, paths / es.to_numpy(), 0.0).sum(axis=1)
    result: np.ndarray = tail / (paths.shape[1] * (1.0 - level)) + 1.0
    return result


def test_z2_by_hand() -> None:
    # Exceptions on days 1 and 3 (returns below -VaR = -0.02); alpha = 1 - 0.75.
    # Z2 = (-0.03 / 0.03 - 0.05 / 0.04) / (4 * 0.25) + 1 = -2.25 + 1 = -1.25.
    z2 = acerbi_szekely_z2(
        [-0.03, 0.01, -0.05, 0.02], [0.02, 0.02, 0.02, 0.02], [0.03, 0.03, 0.04, 0.04], level=0.75
    )
    assert z2 == pytest.approx(-1.25, rel=1e-14)
    # A loss exactly at VaR is not an exception, and without exceptions Z2 = 1.
    assert acerbi_szekely_z2([-0.02, 0.01], [0.02, 0.02], [0.03, 0.03], level=0.9) == 1.0


def test_z2_is_zero_on_average_when_the_forecasts_are_right(returns: pd.Series) -> None:
    # If returns follow the forecast distributions, E[r 1{r < -VaR}] = -alpha ES each
    # day, so E[Z2] = 0 (Acerbi and Szekely 2014). Check on paths drawn from them.
    for method in ("gaussian", "ewma"):
        forecast = rolling_var_es(returns, method=method, level=0.975, window=100)
        paths = forecast.simulate(20_000, seed=2)
        z2 = z2_of_paths(paths, forecast.var, forecast.es, 0.975)
        assert z2[0] == pytest.approx(
            acerbi_szekely_z2(paths[0], forecast.var, forecast.es, level=0.975), rel=1e-12
        )
        assert abs(z2.mean()) < 4 * z2.std(ddof=1) / math.sqrt(z2.size)


def test_null_distribution_does_not_depend_on_the_volatility_path(returns: pd.Series) -> None:
    # With zero-mean normal forecasts, r / sigma is standard normal and Z2 depends
    # only on it, so two volatility paths share one null distribution. With the
    # same seed the simulated statistics are identical up to round-off.
    level, n_days = 0.99, 250
    calm = rolling_var_es(returns.iloc[:350], method="ewma", level=level, window=100)
    wild = rolling_var_es(3.0 * returns.iloc[-350:], method="ewma", level=level, window=100)
    first = es_backtest(calm.last(n_days), n_simulations=4_000, seed=9)
    second = es_backtest(wild.last(n_days), n_simulations=4_000, seed=9)
    assert first.critical_value == pytest.approx(second.critical_value, abs=1e-12)
    # The same quantile from standard normal draws with constant VaR = z and ES = phi(z) / alpha.
    z = NormalDist().inv_cdf(level)
    draws = np.random.default_rng(9).standard_normal((4_000, n_days))
    direct = [
        acerbi_szekely_z2(
            path, np.full(n_days, z), np.full(n_days, NormalDist().pdf(z) / 0.01), level=level
        )
        for path in draws
    ]
    assert first.critical_value == pytest.approx(float(np.quantile(direct, 0.05)), abs=1e-12)


def test_es_backtest_reports_a_monte_carlo_p_value(returns: pd.Series) -> None:
    forecast = rolling_var_es(returns, method="historical", level=0.975, window=250)
    result = es_backtest(forecast, n_simulations=999, seed=4)
    assert result.statistic == pytest.approx(
        acerbi_szekely_z2(forecast.returns, forecast.var, forecast.es, level=0.975)
    )
    assert 1 / 1000 <= result.p_value <= 1.0
    assert result.n_simulations == 999
    assert result.significance == 0.05
    # The volatility doubled on day 300, so a 250-day window badly underestimates ES.
    assert result.statistic < result.critical_value
    assert result.p_value < 0.05
    assert es_backtest(forecast, n_simulations=999, seed=4) == result


def test_es_backtest_batches_long_simulations(returns: pd.Series) -> None:
    # 2,000,000 path-days per batch: 30,000 paths of 350 days need several batches,
    # and must consume the random stream exactly as one large draw would.
    forecast = rolling_var_es(returns, method="ewma", window=250)
    batched = es_backtest(forecast, n_simulations=30_000, seed=1)
    direct = z2_of_paths(forecast.simulate(30_000, seed=1), forecast.var, forecast.es, 0.99)
    assert batched.critical_value == pytest.approx(float(np.quantile(direct, 0.05)), abs=1e-12)


def test_rejects_invalid_inputs(returns: pd.Series) -> None:
    with pytest.raises(InvalidParameterError, match="same length"):
        acerbi_szekely_z2([0.01, 0.02], [0.02], [0.03, 0.03], level=0.975)
    with pytest.raises(InvalidParameterError, match="positive"):
        acerbi_szekely_z2([0.01, 0.02], [0.02, 0.02], [0.03, 0.0], level=0.975)
    forecast = rolling_var_es(returns, method="gaussian", window=100)
    with pytest.raises(InvalidParameterError, match="n_simulations"):
        es_backtest(forecast, n_simulations=0)
    with pytest.raises(InvalidParameterError, match="significance"):
        es_backtest(forecast, significance=1.0)

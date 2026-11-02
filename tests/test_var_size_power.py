"""Size and power of the backtests in simulation, with fixed seeds.

Size: returns are drawn from the forecast model itself, so every rejection is a
false alarm. Power: Gaussian VaR is applied to fat-tailed Student-t returns.
"""

import math
from statistics import NormalDist

import numpy as np
import pytest

from quantkit.var import backtest_var, es_backtest, null_rejection_rates, rolling_var_es

WINDOW, DECAY, LEVEL = 250, 0.94, 0.99


def ewma_model_paths(n_paths: int, n_test: int, seed: int) -> tuple[np.ndarray, ...]:
    """SYNTHETIC returns drawn from the EWMA model: r[t] = sigma[t] z[t].

    ``sigma[t]**2`` is the normalized EWMA of the previous WINDOW squared returns,
    computed here without quantkit. Returns the paths, sigma and the shocks z for
    the last ``n_test`` days.
    """
    rng = np.random.default_rng(seed)
    weights = DECAY ** np.arange(WINDOW - 1, -1, -1.0)
    weights /= weights.sum()
    paths = np.empty((n_paths, WINDOW + n_test))
    paths[:, :WINDOW] = 0.01 * rng.standard_normal((n_paths, WINDOW))
    shocks = rng.standard_normal((n_paths, n_test))
    sigma = np.empty((n_paths, n_test))
    for k in range(n_test):
        t = WINDOW + k
        sigma[:, k] = np.sqrt(np.square(paths[:, t - WINDOW : t]) @ weights)
        paths[:, t] = sigma[:, k] * shocks[:, k]
    return paths, sigma, shocks


def standard_error(rate: float, n: int) -> float:
    return math.sqrt(rate * (1 - rate) / n)


@pytest.fixture(scope="module")
def model_paths() -> tuple[np.ndarray, ...]:
    return ewma_model_paths(n_paths=1_000, n_test=250, seed=2026)


def test_ewma_forecasts_recover_the_model_volatility_and_shocks(
    model_paths: tuple[np.ndarray, ...],
) -> None:
    paths, sigma, shocks = model_paths
    z = NormalDist().inv_cdf(LEVEL)
    for k in range(20):
        forecast = rolling_var_es(paths[k], method="ewma", level=LEVEL, window=WINDOW)
        np.testing.assert_allclose(forecast.scale, sigma[k], rtol=1e-12)
        # The forecast is the true conditional quantile, so the exceptions are
        # exactly the days with a shock below -z: i.i.d. Bernoulli(1%).
        np.testing.assert_array_equal(forecast.exceptions, shocks[k] < -z)


def test_size_matches_the_exact_null_rejection_rates(model_paths: tuple[np.ndarray, ...]) -> None:
    paths, _, _ = model_paths
    rejected = np.zeros(4)
    for k, path in enumerate(paths):
        forecast = rolling_var_es(path, method="ewma", level=LEVEL, window=WINDOW)
        result = backtest_var(forecast.exceptions, level=LEVEL)
        # 199 simulations make 5% an exact Monte Carlo cutoff: 0.05 * (199 + 1) = 10.
        z2 = es_backtest(forecast, n_simulations=199, seed=k)
        rejected += [
            result.unconditional.p_value < 0.05,
            result.independence.p_value < 0.05,
            result.conditional.p_value < 0.05,
            z2.p_value <= 0.05,
        ]
    rates = rejected / len(paths)
    exact = null_rejection_rates(250, level=LEVEL)
    for measured, expected in zip(
        rates, (exact.unconditional, exact.independence, exact.conditional, 0.05), strict=True
    ):
        assert abs(measured - expected) < 4 * standard_error(expected, len(paths))


def gaussian_rejection_rates(paths: np.ndarray, seed: int) -> np.ndarray:
    """Share of paths whose Gaussian VaR/ES forecasts fail LR_uc, LR_cc and Z2 at 5%."""
    rejected = np.zeros(3)
    for k, path in enumerate(paths):
        forecast = rolling_var_es(path, method="gaussian", level=LEVEL, window=WINDOW)
        result = backtest_var(forecast.exceptions, level=LEVEL)
        z2 = es_backtest(forecast, n_simulations=199, seed=seed + k)
        rejected += [
            result.unconditional.p_value < 0.05,
            result.conditional.p_value < 0.05,
            z2.p_value <= 0.05,
        ]
    rates: np.ndarray = rejected / len(paths)
    return rates


def test_gaussian_var_on_student_t_returns_is_rejected_far_more_often() -> None:
    rng = np.random.default_rng(7)
    shape = (200, WINDOW + 1_000)
    normal = 0.01 * rng.standard_normal(shape)
    fat = 0.01 * rng.standard_t(4, shape) * math.sqrt(2 / 4)  # unit variance before scaling
    size = gaussian_rejection_rates(normal, seed=0)
    power = gaussian_rejection_rates(fat, seed=0)
    # examples/var_size_power.py (1,000 paths) measures rejection rates of 2.2% (LR_uc),
    # 0.9% (LR_cc) and 3.9% (Z2) on normal returns, and 46.9%, 31.3% and 87.3% on
    # Student-t returns. Here (200 paths) these bounds leave at least four standard errors.
    assert (size < 0.12).all()
    assert power[0] > 0.30
    assert power[1] > 0.15
    assert power[2] > 0.65
    assert power[2] > power[0]  # Z2 sees the size of tail losses, not only their count

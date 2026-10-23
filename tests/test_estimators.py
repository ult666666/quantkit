"""Return and covariance estimators, including Ledoit-Wolf against scikit-learn."""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf

from quantkit.data import simulate_gbm_prices
from quantkit.exceptions import DataValidationError, InvalidParameterError
from quantkit.portfolio import (
    ReturnMethod,
    annualized_mean_returns,
    compute_returns,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)

PRICES = pd.DataFrame(
    {"X": [100.0, 110.0, 99.0, 99.0], "Y": [50.0, 50.0, 55.0, 44.0]},
    index=pd.date_range("2024-01-01", periods=4, freq="D"),
)


def test_simple_and_log_returns_by_hand() -> None:
    simple = compute_returns(PRICES)
    np.testing.assert_allclose(simple["X"], [0.10, -0.10, 0.0])
    np.testing.assert_allclose(simple["Y"], [0.0, 0.10, -0.20])
    assert list(simple.index) == list(PRICES.index[1:])
    log = compute_returns(PRICES, method="log")
    np.testing.assert_allclose(log, np.log1p(simple))


def test_rejects_unknown_return_method() -> None:
    with pytest.raises(InvalidParameterError, match="method"):
        compute_returns(PRICES, method=cast("ReturnMethod", "arithmetic"))


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(np.nan, "missing"), (-1.0, "positive"), (0.0, "positive"), (np.inf, "infinite")],
)
def test_rejects_bad_prices(bad_value: float, message: str) -> None:
    prices = PRICES.copy()
    prices.iloc[2, 1] = bad_value
    with pytest.raises(DataValidationError, match=message):
        compute_returns(prices)


def test_rejects_non_dataframes_and_tiny_tables() -> None:
    with pytest.raises(DataValidationError, match="DataFrame"):
        compute_returns(cast("pd.DataFrame", PRICES["X"]))  # wrong type on purpose
    with pytest.raises(DataValidationError, match="at least two rows"):
        compute_returns(PRICES.iloc[:1])


def test_sample_covariance_and_mean_are_annualized() -> None:
    returns = compute_returns(simulate_gbm_prices(n_days=300, seed=1))
    pd.testing.assert_frame_equal(sample_covariance(returns), returns.cov() * 252)
    pd.testing.assert_series_equal(
        annualized_mean_returns(returns, periods_per_year=12), returns.mean() * 12
    )


@pytest.mark.parametrize(("n_obs", "n_assets"), [(500, 5), (60, 10), (20, 40)])
def test_ledoit_wolf_matches_scikit_learn(n_obs: int, n_assets: int) -> None:
    rng = np.random.default_rng(n_obs)
    raw = rng.standard_normal((n_obs, n_assets)) @ rng.standard_normal((n_assets, n_assets))
    returns = pd.DataFrame(0.01 * raw, columns=[f"A{i}" for i in range(n_assets)])
    reference_cov, reference_shrinkage = ledoit_wolf(returns.to_numpy())
    assert ledoit_wolf_shrinkage(returns) == pytest.approx(reference_shrinkage, rel=1e-10)
    np.testing.assert_allclose(
        ledoit_wolf_covariance(returns, periods_per_year=1), reference_cov, rtol=1e-10, atol=1e-16
    )


def test_ledoit_wolf_is_well_conditioned_with_more_assets_than_observations() -> None:
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(0.01 * rng.standard_normal((15, 30)))
    sample = sample_covariance(returns).to_numpy()
    shrunk = ledoit_wolf_covariance(returns).to_numpy()
    # With 15 observations the 30x30 sample covariance has rank 14 at most...
    assert np.linalg.matrix_rank(sample) < 30
    # ...while the shrinkage estimate is strictly positive definite.
    assert np.linalg.eigvalsh(shrunk).min() > 0
    assert 0.0 < ledoit_wolf_shrinkage(returns) <= 1.0


def test_no_shrinkage_when_sample_covariance_is_already_a_scaled_identity() -> None:
    # Orthogonal, equal-length, zero-mean columns: S = I exactly.
    returns = pd.DataFrame([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    assert ledoit_wolf_shrinkage(returns) == 0.0
    np.testing.assert_allclose(ledoit_wolf_covariance(returns, periods_per_year=1), np.eye(2))


def test_estimators_validate_returns() -> None:
    with pytest.raises(InvalidParameterError, match="NaN"):
        sample_covariance(pd.DataFrame({"X": [0.01, np.nan, 0.02]}))
    with pytest.raises(InvalidParameterError, match="at least 2 rows"):
        ledoit_wolf_shrinkage(pd.DataFrame({"X": [0.01]}))
    with pytest.raises(InvalidParameterError, match="numeric"):
        ledoit_wolf_covariance(pd.DataFrame({"X": ["a", "b"]}))
    with pytest.raises(InvalidParameterError, match="periods_per_year"):
        sample_covariance(pd.DataFrame({"X": [0.01, 0.02]}), periods_per_year=0)

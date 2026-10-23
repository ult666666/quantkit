"""Synthetic GBM generator statistics and CSV loading/validation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantkit.data import load_prices_csv, simulate_gbm_prices
from quantkit.exceptions import DataValidationError, InvalidParameterError

TICKERS = ["AAA", "BBB", "CCC"]
CORR = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])


def log_returns_of(prices: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(prices.to_numpy()), axis=0)


def test_simulated_prices_have_the_expected_shape_and_calendar() -> None:
    prices = simulate_gbm_prices(tickers=TICKERS, n_days=30, start="2024-01-01", seed=0)
    assert prices.shape == (30, 3)
    assert list(prices.columns) == TICKERS
    assert prices.index[0] == pd.Timestamp("2024-01-01")
    assert prices.index.name == "date"
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert (prices.index.day_of_week < 5).all()  # business days only
    np.testing.assert_allclose(prices.iloc[0], 100.0)
    assert (prices > 0).all().all()


def test_seed_makes_simulation_reproducible() -> None:
    first = simulate_gbm_prices(n_days=50, seed=11)
    pd.testing.assert_frame_equal(first, simulate_gbm_prices(n_days=50, seed=11))
    assert not first.equals(simulate_gbm_prices(n_days=50, seed=12))


def test_simulated_log_returns_match_target_moments() -> None:
    drift = np.array([0.05, 0.10, 0.02])
    vol = np.array([0.10, 0.25, 0.40])
    prices = simulate_gbm_prices(
        tickers=TICKERS, n_days=50_001, drift=drift, vol=vol, correlation=CORR, seed=5
    )
    log_returns = log_returns_of(prices)
    n = len(log_returns)
    dt = 1 / 252
    # Standard errors: mean ~ sigma sqrt(dt / n); correlation ~ (1 - rho^2) / sqrt(n).
    mean_se = vol * np.sqrt(dt / n)
    assert np.all(np.abs(log_returns.mean(axis=0) - (drift - vol**2 / 2) * dt) < 4 * mean_se)
    np.testing.assert_allclose(log_returns.std(axis=0, ddof=1) * np.sqrt(252), vol, rtol=0.02)
    np.testing.assert_allclose(np.corrcoef(log_returns.T), CORR, atol=0.02)


def test_scalar_correlation_builds_a_constant_correlation_matrix() -> None:
    prices = simulate_gbm_prices(tickers=TICKERS, n_days=40_001, correlation=0.5, seed=8)
    corr = np.corrcoef(log_returns_of(prices).T)
    np.testing.assert_allclose(corr[np.triu_indices(3, 1)], 0.5, atol=0.02)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tickers": ["A", "A"]}, "unique"),
        ({"tickers": []}, "unique"),
        ({"n_days": 1}, "n_days"),
        ({"drift": [0.1, 0.2]}, "drift"),
        ({"vol": -0.2}, "vol"),
        ({"vol": [0.1, np.nan, 0.2]}, "vol"),
        ({"correlation": 1.5}, "positive definite"),
        ({"correlation": np.eye(2)}, "shape"),
        ({"correlation": 2 * np.eye(3)}, "diagonal"),
        ({"correlation": np.array([[1, 0.5, 0], [0.4, 1, 0], [0, 0, 1]])}, "symmetric"),
        ({"initial_price": 0.0}, "initial_price"),
    ],
)
def test_simulation_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    params: dict[str, Any] = {"tickers": TICKERS, "n_days": 10, "seed": 0, **kwargs}
    with pytest.raises(InvalidParameterError, match=message):
        simulate_gbm_prices(**params)


# --------------------------------------------------------------------- CSV loading


@pytest.fixture
def prices() -> pd.DataFrame:
    return simulate_gbm_prices(tickers=TICKERS, n_days=20, seed=3)


def test_wide_csv_round_trip(tmp_path: Path, prices: pd.DataFrame) -> None:
    path = tmp_path / "wide.csv"
    prices.to_csv(path, index_label="date")
    loaded = load_prices_csv(path)
    pd.testing.assert_frame_equal(loaded, prices, check_freq=False)


def test_long_csv_round_trip_in_any_row_order(tmp_path: Path, prices: pd.DataFrame) -> None:
    long = prices.reset_index().melt(id_vars="date", var_name="ticker", value_name="close")
    long = long.sample(frac=1.0, random_state=0)  # shuffle rows
    long.columns = ["Date", "Ticker", "Close"]  # headers are case-insensitive
    path = tmp_path / "long.csv"
    long.to_csv(path, index=False)
    loaded = load_prices_csv(path)
    pd.testing.assert_frame_equal(loaded, prices, check_freq=False)


def test_unsorted_wide_rows_are_sorted(tmp_path: Path, prices: pd.DataFrame) -> None:
    path = tmp_path / "unsorted.csv"
    prices.iloc[::-1].to_csv(path, index_label="date")
    pd.testing.assert_frame_equal(load_prices_csv(path), prices, check_freq=False)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prices.csv"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("date,AAA\n2024-01-02,10\n2024-01-03,\n", "missing"),
        ("date,AAA\n2024-01-02,10\n2024-01-03,-1\n", "positive"),
        ("date,AAA\n2024-01-02,10\n2024-01-02,11\n", "unique"),
        ("day,AAA\n2024-01-02,10\n2024-01-03,11\n", "'date' column"),
        ("date,AAA\nnot-a-date,10\n2024-01-03,11\n", "dates"),
        ("date,AAA\n2024-01-02,ten\n2024-01-03,11\n", "numeric"),
        ("", "could not parse"),
        (
            "date,ticker,close\n2024-01-02,AAA,10\n2024-01-02,AAA,11\n2024-01-03,AAA,12\n",
            "duplicate rows",
        ),
        (
            "date,ticker,close\n2024-01-02,AAA,10\n2024-01-02,BBB,20\n2024-01-03,AAA,11\n",
            "'BBB' on 2024-01-03",
        ),
    ],
)
def test_csv_validation_errors_are_specific(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_prices_csv(write(tmp_path, text))


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prices_csv(tmp_path / "nope.csv")

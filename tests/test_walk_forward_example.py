"""Exercise the walk-forward information boundary and real exported workflow."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import backtest_weights
from quantkit.data import simulate_gbm_prices
from quantkit.portfolio import compute_returns, ledoit_wolf_covariance, risk_contributions

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "walk_forward_risk_parity.py"


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("walk_forward_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prices() -> pd.DataFrame:
    return simulate_gbm_prices(
        tickers=["AAA", "BBB", "CCC"],
        n_days=420,
        drift=[0.04, 0.06, 0.08],
        vol=[0.10, 0.18, 0.28],
        correlation=0.3,
        seed=7,
    )


@pytest.mark.parametrize("cutoff", [65, 130, 260])
def test_current_and_future_closes_cannot_change_prior_targets(
    prices: pd.DataFrame, cutoff: int
) -> None:
    example = load_example()
    original = example.build_targets(prices, lookback=60)
    perturbed = prices.copy()
    rng = np.random.default_rng(cutoff)
    later = perturbed.iloc[cutoff:]
    perturbed.iloc[cutoff:] = later.to_numpy() * rng.uniform(0.8, 1.2, later.shape)
    changed = example.build_targets(perturbed, lookback=60)
    # Even date cutoff's target excludes that date's close.
    pd.testing.assert_frame_equal(
        original.risk_parity.iloc[: cutoff + 1], changed.risk_parity.iloc[: cutoff + 1]
    )
    pd.testing.assert_frame_equal(
        original.equal_weight.iloc[: cutoff + 1], changed.equal_weight.iloc[: cutoff + 1]
    )
    # Negative control: later estimates must actually respond to changed history.
    assert not original.risk_parity.iloc[cutoff + 1 :].equals(
        changed.risk_parity.iloc[cutoff + 1 :]
    )


def test_monthly_targets_are_feasible_and_equalize_estimated_risk(prices: pd.DataFrame) -> None:
    schedule = load_example().build_targets(prices, lookback=60)
    risk = schedule.risk_parity.dropna()
    equal = schedule.equal_weight.dropna()
    pd.testing.assert_index_equal(risk.index, equal.index)
    assert len(risk) > 10
    assert list(risk.columns) == ["AAA", "BBB", "CCC"]
    assert np.isfinite(risk.to_numpy()).all()
    assert ((risk >= 0) & (risk <= 1)).all().all()
    np.testing.assert_allclose(risk.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(equal.to_numpy(), 1 / 3)
    assert len(risk.index.to_period("M").unique()) == len(risk)
    # First eligible monthly close is April 1 (index 65), after 60 prior returns.
    assert risk.index[0] == pd.Timestamp("2020-04-01")
    cov = ledoit_wolf_covariance(compute_returns(prices.iloc[4:65]))
    np.testing.assert_allclose(risk_contributions(risk.iloc[0], cov), 1 / 3, atol=1e-8)


def test_windows_expose_exact_prior_returns_without_the_rebalance_close(
    prices: pd.DataFrame,
) -> None:
    windows = load_example().build_targets(prices, lookback=60).windows
    first = windows.iloc[0]
    assert first["rebalance_date"] == pd.Timestamp("2020-04-01")
    assert first["history_start"] == prices.index[4]
    assert first["history_end"] == prices.index[64]
    assert (windows["history_end"] < windows["rebalance_date"]).all()
    assert (windows["n_returns"] == 60).all()


def test_both_strategies_stay_in_cash_then_pay_the_same_entry_cost(
    prices: pd.DataFrame,
) -> None:
    schedule = load_example().build_targets(prices, lookback=60)
    for targets in (schedule.risk_parity, schedule.equal_weight):
        assert targets.iloc[:65].isna().all().all()
        result = backtest_weights(prices, targets, cost_bps=10)
        np.testing.assert_allclose(result.equity.iloc[:65], 1.0)
        np.testing.assert_allclose(result.weights.iloc[:66], 0.0)
        assert result.equity.iloc[65] == pytest.approx(0.999)
        np.testing.assert_allclose(result.weights.iloc[66], targets.iloc[65])


def test_costs_cannot_improve_wealth_for_identical_targets(prices: pd.DataFrame) -> None:
    schedule = load_example().build_targets(prices, lookback=60)
    for targets in (schedule.risk_parity, schedule.equal_weight):
        free = backtest_weights(prices, targets, cost_bps=0)
        costly = backtest_weights(prices, targets, cost_bps=10)
        assert (costly.equity <= free.equity + 1e-12).all()
        assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_actual_cli_exports_reproducible_csvs_and_matching_summary(tmp_path: Path) -> None:
    outputs = [tmp_path / "first", tmp_path / "second"]
    for directory in outputs:
        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--output-dir",
                str(directory),
                "--days",
                "120",
                "--lookback",
                "20",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "SYNTHETIC" in result.stdout
        assert "not evidence of investment performance" in result.stdout
    names = {
        "prices.csv",
        "risk_parity_targets.csv",
        "equal_weight_targets.csv",
        "wealth.csv",
        "summary.csv",
        "estimation_windows.csv",
    }
    assert {path.name for path in outputs[0].iterdir()} == names
    for name in names:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    wealth = pd.read_csv(outputs[0] / "wealth.csv", index_col="date")
    assert wealth.shape == (120, 2)
    np.testing.assert_allclose(wealth.iloc[:23], 1.0)
    assert (wealth.iloc[23] < 1.0).all()  # first monthly trade pays costs
    summary = pd.read_csv(outputs[0] / "summary.csv", index_col="metric")
    np.testing.assert_allclose(
        summary.loc["total_return"], wealth.iloc[-1].to_numpy() - 1, rtol=1e-8
    )
    # Evaluation begins with the entry cost, not the 23 cash warmup rows.
    expected_cagr = wealth.iloc[-1].to_numpy() ** (252 / 97) - 1
    np.testing.assert_allclose(summary.loc["cagr"], expected_cagr, rtol=1e-8)
    targets = pd.read_csv(outputs[0] / "risk_parity_targets.csv", index_col="date")
    assert targets.iloc[:23].isna().all().all()
    assert targets.iloc[23].notna().all()


@pytest.mark.parametrize("lookback", [0, 1, -1])
def test_invalid_estimation_window_is_rejected(prices: pd.DataFrame, lookback: int) -> None:
    with pytest.raises(ValueError, match="lookback"):
        load_example().build_targets(prices, lookback=lookback)


def test_cli_short_history_fails_before_writing_files(tmp_path: Path) -> None:
    output = tmp_path / "absent"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--days", "10", "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "enough history" in result.stderr
    assert not output.exists()

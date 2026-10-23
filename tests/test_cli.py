"""Command-line interface: every subcommand end to end."""

import runpy
from pathlib import Path

import pytest

from quantkit.cli import STRATEGY_NAMES, main
from quantkit.data import load_prices_csv

OPTION = ["--spot", "42", "--strike", "40", "--rate", "0.10", "--maturity", "0.5"]


@pytest.fixture(scope="module")
def prices_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "prices.csv"
    assert main(["simulate", "--output", str(path), "--days", "400", "--seed", "1"]) == 0
    return path


def test_price_compares_methods_and_prints_greeks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["price", *OPTION, "--vol", "0.20", "--paths", "20000"]) == 0
    out = capsys.readouterr().out
    assert "Black-Scholes-Merton                  4.7594" in out
    assert "CRR binomial (500 steps)" in out
    assert "Monte Carlo (20,000 paths)" in out
    assert "delta     0.7791" in out


def test_price_american_put_reports_the_early_exercise_premium(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["--spot", "50", "--strike", "50", "--rate", "0.1", "--maturity", "0.4167"]
    assert main(["price", *args, "--vol", "0.4", "--kind", "put", "--american"]) == 0
    out = capsys.readouterr().out
    assert "American put" in out
    assert "early-exercise premium" in out


def test_american_call_without_dividends_has_no_early_exercise_premium(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["price", *OPTION, "--vol", "0.20", "--american"]) == 0
    assert "early-exercise premium                    0.0000" in capsys.readouterr().out


def test_iv_round_trips_the_textbook_price(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["iv", "--price", "4.7594", *OPTION]) == 0
    assert "implied volatility: 0.199997 (20.00%)" in capsys.readouterr().out


def test_errors_are_reported_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["iv", "--price", "50", *OPTION]) == 1
    assert "no-arbitrage bounds" in capsys.readouterr().err
    assert main(["backtest", "does-not-exist.csv"]) == 1
    assert "does-not-exist.csv" in capsys.readouterr().err


def test_simulate_writes_a_loadable_csv(prices_csv: Path) -> None:
    prices = load_prices_csv(prices_csv)
    assert prices.shape == (400, 5)


@pytest.mark.parametrize("strategy", STRATEGY_NAMES)
def test_backtest_runs_every_strategy(
    prices_csv: Path, strategy: str, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["backtest", str(prices_csv), "--strategy", strategy, "--slow", "100"]
    assert main([*args, "--lookback", "60", "--window", "20"]) == 0
    out = capsys.readouterr().out
    assert strategy in out
    assert "buy-and-hold (equal weight)" in out
    assert "max drawdown" in out


def test_frontier_prints_a_table_and_saves_a_plot(
    prices_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot = tmp_path / "frontier.png"
    assert main(["frontier", str(prices_csv), "--shrinkage", "--plot", str(plot)]) == 0
    out = capsys.readouterr().out
    assert "Ledoit-Wolf covariance" in out
    assert "max Sharpe" in out
    assert plot.stat().st_size > 10_000


def test_python_dash_m_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["quantkit", "--version"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("quantkit", run_name="__main__")
    assert info.value.code == 0
    assert "quantkit" in capsys.readouterr().out

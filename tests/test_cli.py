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


def test_var_backtest_compares_every_method_on_common_days(
    prices_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["var-backtest", str(prices_csv), "--window", "100", "--es-simulations", "199"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "equal-weight portfolio of 5 assets, rebalanced daily: 399 daily returns" in out
    # FHS needs 2 x 100 returns of history, which leaves 199 days for every method.
    assert "backtest: 199 days" in out
    assert "the period every method covers" in out
    header = next(line for line in out.splitlines() if "historical" in line)
    assert header.split() == ["historical", "Gaussian", "EWMA", "FHS"]
    for row in ("Kupiec LR_uc p-value", "conditional coverage LR_cc p-value", "Z2 simulated"):
        assert row in out
    assert "n/a (< 250 days)" in out
    assert "a correct model fails the 5% tests" in out


def test_var_backtest_one_method_one_ticker(
    prices_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["var-backtest", str(prices_csv), "--method", "ewma", "--window", "100"]
    assert main([*base, "--ticker", "CCC", "--es-simulations", "99"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("CCC: 399 daily returns")
    assert "backtest: 299 days, " in out
    zone = next(line for line in out.splitlines() if line.startswith("Basel zone"))
    assert zone.split()[-2] in {"green", "yellow", "red"}
    assert main([*base, "--level", "0.975", "--es-simulations", "99"]) == 0
    out = capsys.readouterr().out
    assert "97.5% one-day VaR and ES, 100-day window, EWMA decay 0.94" in out
    assert "n/a (99% only)" in out


def test_var_backtest_handles_samples_too_long_for_the_exact_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Twenty years of daily data at 95%: too many expected exceptions for
    # null_rejection_rates to enumerate. The backtest must still succeed and only
    # leave out the exact size of the tests.
    path = tmp_path / "long.csv"
    assert main(["simulate", "--output", str(path), "--days", "5040", "--seed", "3"]) == 0
    capsys.readouterr()
    args = ["var-backtest", str(path), "--method", "historical", "--level", "0.95"]
    assert main([*args, "--es-simulations", "99"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "backtest: 4789 days" in captured.out
    assert "Kupiec LR_uc p-value" in captured.out
    assert "the exact size of the tests is not computed" in captured.out
    assert "a correct model fails the 5% tests" not in captured.out


def test_var_backtest_reports_bad_inputs(
    prices_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["var-backtest", str(prices_csv), "--ticker", "ZZZ"]) == 1
    assert "ticker 'ZZZ' is not in" in capsys.readouterr().err
    assert main(["var-backtest", str(prices_csv), "--method", "fhs"]) == 1
    assert "needs more than 500 returns" in capsys.readouterr().err


def test_python_dash_m_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["quantkit", "--version"])
    with pytest.raises(SystemExit) as info:
        runpy.run_module("quantkit", run_name="__main__")
    assert info.value.code == 0
    assert "quantkit" in capsys.readouterr().out

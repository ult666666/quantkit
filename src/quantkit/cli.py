"""Command-line interface: ``quantkit price | iv | simulate | backtest | frontier``.

Run ``quantkit <command> --help`` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quantkit import QuantkitError, __version__
from quantkit.backtest import (
    BuyAndHold,
    EqualWeight,
    InverseVolatility,
    MovingAverageCrossover,
    Strategy,
    TimeSeriesMomentum,
    run_backtest,
)
from quantkit.data import load_prices_csv, simulate_gbm_prices
from quantkit.portfolio import (
    annualized_mean_returns,
    compute_returns,
    efficient_frontier,
    ledoit_wolf_covariance,
    max_sharpe,
    min_variance,
    portfolio_performance,
    sample_covariance,
)
from quantkit.pricing import (
    OptionSpec,
    binomial_price,
    bsm_greeks,
    bsm_price,
    implied_vol,
    monte_carlo_price,
)

SIMULATED_TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
SIMULATED_DRIFTS = (0.03, 0.05, 0.07, 0.09, 0.11)
SIMULATED_VOLS = (0.08, 0.12, 0.18, 0.24, 0.32)
SIMULATED_CORRELATION = 0.3

STRATEGY_NAMES = ("buy-and-hold", "equal-weight", "ma-crossover", "momentum", "inverse-vol")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (QuantkitError, OSError) as exc:
        print(f"quantkit: error: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------- commands


def _cmd_price(args: argparse.Namespace) -> None:
    option = OptionSpec(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        vol=args.vol,
        maturity=args.maturity,
        dividend_yield=args.dividend_yield,
        kind=args.kind,
    )
    style = "American" if args.american else "European"
    print(
        f"{style} {option.kind}: S={option.spot:g} K={option.strike:g} T={option.maturity:g} "
        f"r={option.rate:g} q={option.dividend_yield:g} sigma={option.vol:g}\n"
    )
    bsm = bsm_price(option)
    tree = binomial_price(option, steps=args.steps)
    if args.american:
        american = binomial_price(option, steps=args.steps, exercise="american")
        print(f"{'method':<38}{'price':>10}")
        print(f"{f'CRR binomial, American ({args.steps} steps)':<38}{american:>10.4f}")
        print(f"{f'CRR binomial, European ({args.steps} steps)':<38}{tree:>10.4f}")
        print(f"{'Black-Scholes-Merton, European':<38}{bsm:>10.4f}")
        # Compare trees with the same steps so discretization error cancels out.
        print(f"{'early-exercise premium':<38}{american - tree:>10.4f}")
        return

    mc = monte_carlo_price(option, n_paths=args.paths, seed=args.seed)
    print(f"{'method':<34}{'price':>10}{'std. error':>13}")
    print(f"{'Black-Scholes-Merton':<34}{bsm:>10.4f}")
    print(f"{f'CRR binomial ({args.steps} steps)':<34}{tree:>10.4f}")
    print(f"{f'Monte Carlo ({args.paths:,} paths)':<34}{mc.price:>10.4f}{mc.std_error:>13.4f}")
    greeks = bsm_greeks(option)
    print("\nGreeks (Black-Scholes-Merton)")
    print(f"delta {greeks.delta:>10.4f}")
    print(f"gamma {greeks.gamma:>10.4f}")
    print(f"vega  {greeks.vega:>10.4f}   per 1.00 change in volatility")
    print(f"theta {greeks.theta:>10.4f}   per year")
    print(f"rho   {greeks.rho:>10.4f}   per 1.00 change in the rate")


def _cmd_iv(args: argparse.Namespace) -> None:
    vol = implied_vol(
        args.price,
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        maturity=args.maturity,
        kind=args.kind,
        dividend_yield=args.dividend_yield,
    )
    print(f"implied volatility: {vol:.6f} ({vol:.2%})")


def _cmd_simulate(args: argparse.Namespace) -> None:
    prices = simulate_gbm_prices(
        tickers=SIMULATED_TICKERS,
        n_days=args.days,
        drift=SIMULATED_DRIFTS,
        vol=SIMULATED_VOLS,
        correlation=SIMULATED_CORRELATION,
        start=args.start,
        seed=args.seed,
    )
    prices.to_csv(args.output, index_label="date", float_format="%.4f")
    print(
        f"wrote {len(prices)} days x {prices.shape[1]} tickers of synthetic GBM prices "
        f"to {args.output}"
    )


def _strategy_from(args: argparse.Namespace) -> tuple[str, Strategy]:
    name: str = args.strategy
    if name == "buy-and-hold":
        return name, BuyAndHold()
    if name == "equal-weight":
        return f"{name} ({args.rebalance})", EqualWeight(rebalance=args.rebalance)
    if name == "ma-crossover":
        strategy: Strategy = MovingAverageCrossover(fast=args.fast, slow=args.slow)
        return f"{name} ({args.fast}/{args.slow})", strategy
    if name == "momentum":
        strategy = TimeSeriesMomentum(
            lookback=args.lookback, rebalance=args.rebalance, long_only=args.long_only
        )
        return f"{name} ({args.lookback}d, {args.rebalance})", strategy
    strategy = InverseVolatility(window=args.window, rebalance=args.rebalance)
    return f"{name} ({args.window}d, {args.rebalance})", strategy


def _cmd_backtest(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.prices)
    label, strategy = _strategy_from(args)
    result = run_backtest(prices, strategy, cost_bps=args.cost_bps)
    benchmark = run_backtest(prices, BuyAndHold(), cost_bps=args.cost_bps)
    first, last = prices.index[0], prices.index[-1]
    print(
        f"{prices.shape[1]} assets, {len(prices)} days "
        f"({pd.Timestamp(first).date()} to {pd.Timestamp(last).date()}), "
        f"costs {args.cost_bps:g} bps of traded notional\n"
    )
    table = pd.DataFrame(
        {
            label: result.summary(risk_free_rate=args.risk_free),
            "buy-and-hold (equal weight)": benchmark.summary(risk_free_rate=args.risk_free),
        }
    )
    print(_format_summary(table))


def _format_summary(table: pd.DataFrame) -> str:
    labels = {
        "total_return": ("total return", "{:.1%}"),
        "cagr": ("CAGR", "{:.2%}"),
        "volatility": ("volatility", "{:.2%}"),
        "sharpe": ("Sharpe", "{:.2f}"),
        "sortino": ("Sortino", "{:.2f}"),
        "max_drawdown": ("max drawdown", "{:.1%}"),
        "max_drawdown_periods": ("max drawdown length (days)", "{:.0f}"),
        "calmar": ("Calmar", "{:.2f}"),
        "hit_rate": ("hit rate", "{:.1%}"),
        "annual_turnover": ("annual turnover", "{:.2f}x"),
    }
    formatted = pd.DataFrame(
        {
            column: [fmt.format(table.loc[key, column]) for key, (_, fmt) in labels.items()]
            for column in table.columns
        },
        index=[name for name, _ in labels.values()],
    )
    return formatted.to_string()


def _cmd_frontier(args: argparse.Namespace) -> None:
    prices = load_prices_csv(args.prices)
    returns = compute_returns(prices)
    mu = annualized_mean_returns(returns)
    cov = ledoit_wolf_covariance(returns) if args.shrinkage else sample_covariance(returns)
    bounds = (args.min_weight, args.max_weight)
    frontier = efficient_frontier(
        mu, cov, n_points=args.points, bounds=bounds, risk_free_rate=args.risk_free
    )
    special = {
        "min variance": min_variance(cov, bounds=bounds),
        "max Sharpe": max_sharpe(mu, cov, risk_free_rate=args.risk_free, bounds=bounds),
    }
    frontier.index = pd.RangeIndex(1, len(frontier) + 1)
    rows = [frontier]
    for name, weights in special.items():
        perf = portfolio_performance(weights, mu, cov, risk_free_rate=args.risk_free)
        row = {
            "expected_return": perf.expected_return,
            "volatility": perf.volatility,
            "sharpe": perf.sharpe,
        }
        row.update({str(asset): float(w) for asset, w in weights.items()})
        rows.append(pd.DataFrame([row], index=[name]))
    table = pd.concat(rows)
    estimator = "Ledoit-Wolf" if args.shrinkage else "sample"
    print(
        f"{len(returns)} daily returns of {prices.shape[1]} assets; expected returns = "
        f"annualized historical means, {estimator} covariance; weights in "
        f"[{args.min_weight:g}, {args.max_weight:g}]; risk-free rate {args.risk_free:.2%}\n"
    )
    display = table.rename(columns={"expected_return": "return", "sharpe": "Sharpe"})
    formatters = dict.fromkeys(display.columns, "{:.1%}".format)
    formatters["Sharpe"] = "{:.2f}".format
    print(display.to_string(formatters=formatters))
    if args.plot is not None:
        _save_frontier_plot(table, mu, cov, args.plot)
        print(f"\nsaved plot to {args.plot}")


def _save_frontier_plot(table: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # file output only; never open a window
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise QuantkitError(
            "plotting needs matplotlib: install the 'plot' extra (pip install 'quantkit[plot]')"
        ) from exc
    points = table.iloc[:-2]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        points["volatility"], points["expected_return"], "-o", ms=3, label="efficient frontier"
    )
    vols = np.sqrt(np.diag(cov.to_numpy()))
    ax.scatter(vols, mu, color="gray", label="individual assets")
    for name, x, y in zip(cov.index, vols, mu, strict=True):
        ax.annotate(str(name), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    for name, marker, color in (
        ("min variance", "s", "tab:green"),
        ("max Sharpe", "*", "tab:red"),
    ):
        ax.scatter(
            table.loc[name, "volatility"],
            table.loc[name, "expected_return"],
            marker=marker,
            color=color,
            s=120,
            zorder=3,
            label=name,
        )
    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized expected return")
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------ parser


def _add_option_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spot", type=float, required=True, help="underlying price S")
    parser.add_argument("--strike", type=float, required=True, help="strike price K")
    parser.add_argument("--rate", type=float, required=True, help="risk-free rate r (0.05 = 5%%)")
    parser.add_argument("--maturity", type=float, required=True, help="time to expiry T in years")
    parser.add_argument(
        "--dividend-yield", type=float, default=0.0, help="continuous dividend yield q"
    )
    parser.add_argument("--kind", choices=("call", "put"), default="call")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantkit",
        description="Options pricing, portfolio optimization and backtesting. "
        "Educational and research software, not investment advice.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    price = commands.add_parser("price", help="price a vanilla option with several methods")
    _add_option_arguments(price)
    price.add_argument("--vol", type=float, required=True, help="volatility sigma (0.2 = 20%%)")
    price.add_argument("--american", action="store_true", help="allow early exercise")
    price.add_argument("--steps", type=int, default=500, help="binomial tree steps")
    price.add_argument("--paths", type=int, default=100_000, help="Monte Carlo paths")
    price.add_argument("--seed", type=int, default=42, help="Monte Carlo seed")
    price.set_defaults(handler=_cmd_price)

    iv = commands.add_parser("iv", help="implied volatility from an option price")
    iv.add_argument("--price", type=float, required=True, help="observed option price")
    _add_option_arguments(iv)
    iv.set_defaults(handler=_cmd_iv)

    simulate = commands.add_parser(
        "simulate", help="write synthetic correlated GBM prices to a CSV file"
    )
    simulate.add_argument("--output", type=Path, required=True, help="CSV file to write")
    simulate.add_argument("--days", type=int, default=1260, help="number of business days")
    simulate.add_argument("--start", default="2020-01-01", help="first date (YYYY-MM-DD)")
    simulate.add_argument("--seed", type=int, default=42, help="random seed")
    simulate.set_defaults(handler=_cmd_simulate)

    backtest = commands.add_parser(
        "backtest", help="backtest a strategy on a CSV of daily closing prices"
    )
    backtest.add_argument("prices", type=Path, help="CSV in long (date,ticker,close) or wide form")
    backtest.add_argument("--strategy", choices=STRATEGY_NAMES, default="ma-crossover")
    backtest.add_argument("--cost-bps", type=float, default=10.0, help="cost per trade in bps")
    backtest.add_argument("--rebalance", choices=("D", "W", "M", "Q", "Y"), default="M")
    backtest.add_argument("--fast", type=int, default=50, help="ma-crossover fast window")
    backtest.add_argument("--slow", type=int, default=200, help="ma-crossover slow window")
    backtest.add_argument("--lookback", type=int, default=252, help="momentum lookback (days)")
    backtest.add_argument("--long-only", action="store_true", help="momentum without shorts")
    backtest.add_argument("--window", type=int, default=63, help="inverse-vol window (days)")
    backtest.add_argument("--risk-free", type=float, default=0.0, help="annual rate for Sharpe")
    backtest.set_defaults(handler=_cmd_backtest)

    frontier = commands.add_parser(
        "frontier", help="mean-variance efficient frontier from a CSV of prices"
    )
    frontier.add_argument("prices", type=Path, help="CSV in long (date,ticker,close) or wide form")
    frontier.add_argument("--points", type=int, default=8, help="number of frontier points")
    frontier.add_argument("--min-weight", type=float, default=0.0, help="lower weight bound")
    frontier.add_argument("--max-weight", type=float, default=1.0, help="upper weight bound")
    frontier.add_argument("--risk-free", type=float, default=0.0, help="annual risk-free rate")
    frontier.add_argument(
        "--shrinkage", action="store_true", help="use the Ledoit-Wolf covariance estimate"
    )
    frontier.add_argument("--plot", type=Path, help="also save a PNG chart (needs matplotlib)")
    frontier.set_defaults(handler=_cmd_frontier)
    return parser

"""Moving-average trend following vs. buy-and-hold on synthetic prices.

Run from the repository root:

    uv run python examples/backtest_trend.py

The prices are SYNTHETIC (correlated geometric Brownian motion). GBM returns are
independent from day to day, so there is no trend for a trend-following rule to
exploit: whatever difference one backtest shows is luck plus trading costs. To make
that concrete, the script also repeats the comparison over many independent
synthetic histories. It demonstrates the mechanics of a backtest, not a trading
edge. Writes docs/images/backtest_equity.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantkit.backtest import (
    BuyAndHold,
    MovingAverageCrossover,
    cagr,
    drawdown_series,
    run_backtest,
)
from quantkit.data import simulate_gbm_prices

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "backtest_equity.png"
COST_BPS = 10.0
STRATEGY = MovingAverageCrossover(fast=50, slow=200)


def simulate(seed: int) -> pd.DataFrame:
    """Ten years of daily prices for five synthetic assets."""
    return simulate_gbm_prices(
        tickers=["AAA", "BBB", "CCC", "DDD", "EEE"],
        n_days=2520,
        drift=[0.03, 0.05, 0.07, 0.09, 0.11],
        vol=[0.08, 0.12, 0.18, 0.24, 0.32],
        correlation=0.3,
        seed=seed,
    )


def main() -> None:
    """Backtest both strategies with costs, plot one history, then repeat over many."""
    prices = simulate(seed=42)
    results = {
        "MA crossover 50/200": run_backtest(prices, STRATEGY, cost_bps=COST_BPS),
        "buy-and-hold, equal weight": run_backtest(prices, BuyAndHold(), cost_bps=COST_BPS),
    }
    summary = pd.DataFrame({name: result.summary() for name, result in results.items()})
    print(f"seed 42, costs {COST_BPS:g} bps of traded notional")
    print(summary.round(3).to_string())

    # One backtest is one draw from a distribution. Repeat it on fresh histories.
    differences = []
    for seed in range(100):
        history = simulate(seed)
        trend = run_backtest(history, STRATEGY, cost_bps=COST_BPS).returns
        hold = run_backtest(history, BuyAndHold(), cost_bps=COST_BPS).returns
        differences.append(cagr(trend) - cagr(hold))
    diff = np.array(differences)
    print(
        f"\nover 100 independent synthetic histories, CAGR(MA) - CAGR(buy-and-hold): "
        f"mean {diff.mean():+.2%}, standard error {diff.std(ddof=1) / np.sqrt(diff.size):.2%}; "
        f"MA crossover came out ahead in {(diff > 0).sum()} of 100"
    )

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name, result in results.items():
        top.plot(result.equity.index, result.equity, lw=1.2, label=name)
        bottom.plot(result.returns.index, drawdown_series(result.returns), lw=1, label=name)
    top.set(ylabel="growth of $1", title="SYNTHETIC data: 5 GBM assets, 10 years, seed 42")
    top.legend()
    top.grid(alpha=0.3)
    bottom.set(ylabel="drawdown")
    bottom.yaxis.set_major_formatter("{x:.0%}")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110)
    print(f"saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

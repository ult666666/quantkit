"""Mean-variance efficient frontier for five synthetic assets.

Run from the repository root:

    uv run python examples/efficient_frontier.py

The prices are SYNTHETIC: five years of correlated geometric Brownian motion from
``quantkit.data.simulate_gbm_prices`` with a fixed seed. Expected returns and the
covariance matrix are estimated from those prices, exactly as they would be from
real data. Prints the key portfolios and writes docs/images/efficient_frontier.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantkit.data import simulate_gbm_prices
from quantkit.portfolio import (
    annualized_mean_returns,
    compute_returns,
    efficient_frontier,
    efficient_return_closed_form,
    ledoit_wolf_covariance,
    max_sharpe,
    min_variance,
    portfolio_performance,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "efficient_frontier.png"
RISK_FREE = 0.02


def main() -> None:
    """Estimate inputs from synthetic prices, optimize, and plot the frontier."""
    prices = simulate_gbm_prices(
        tickers=["AAA", "BBB", "CCC", "DDD", "EEE"],
        n_days=1260,
        drift=[0.03, 0.05, 0.07, 0.09, 0.11],
        vol=[0.08, 0.12, 0.18, 0.24, 0.32],
        correlation=0.3,
        seed=42,
    )
    returns = compute_returns(prices)
    mu = annualized_mean_returns(returns)
    cov = ledoit_wolf_covariance(returns)

    long_only = efficient_frontier(mu, cov, n_points=30, risk_free_rate=RISK_FREE)
    gmv = min_variance(cov)
    tangency = max_sharpe(mu, cov, risk_free_rate=RISK_FREE)
    tangency_perf = portfolio_performance(tangency, mu, cov, risk_free_rate=RISK_FREE)
    gmv_perf = portfolio_performance(gmv, mu, cov, risk_free_rate=RISK_FREE)

    # Without bounds the frontier is a hyperbola with a closed form; draw it over
    # the same range of returns for comparison.
    targets = np.linspace(gmv_perf.expected_return - 0.02, long_only["expected_return"].max(), 60)
    unbounded_vol = [
        portfolio_performance(efficient_return_closed_form(mu, cov, m), mu, cov).volatility
        for m in targets
    ]

    print("estimated annualized means:", (mu * 100).round(1).to_dict(), "(%)")
    print(
        f"min variance (long-only): return {gmv_perf.expected_return:.2%}, "
        f"volatility {gmv_perf.volatility:.2%}"
    )
    print(
        f"max Sharpe (long-only, rf {RISK_FREE:.0%}): return {tangency_perf.expected_return:.2%}, "
        f"volatility {tangency_perf.volatility:.2%}, Sharpe {tangency_perf.sharpe:.2f}"
    )
    print("max Sharpe weights:", tangency.round(3).to_dict())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        unbounded_vol, targets, color="0.6", ls="--", lw=1, label="shorting allowed (closed form)"
    )
    ax.plot(long_only["volatility"], long_only["expected_return"], lw=2, label="long-only (SLSQP)")
    asset_vols = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=cov.index)
    ax.scatter(asset_vols, mu, color="black", s=20, zorder=3, label="individual assets")
    for ticker in cov.index:
        ax.annotate(
            str(ticker),
            (asset_vols[ticker], mu[ticker]),
            xytext=(5, -3),
            textcoords="offset points",
        )
    cml_x = np.array([0.0, long_only["volatility"].max()])
    ax.plot(
        cml_x,
        RISK_FREE + tangency_perf.sharpe * cml_x,
        color="tab:red",
        lw=1,
        label="capital market line",
    )
    ax.scatter(
        gmv_perf.volatility,
        gmv_perf.expected_return,
        marker="s",
        s=70,
        color="tab:green",
        zorder=4,
        label="minimum variance",
    )
    ax.scatter(
        tangency_perf.volatility,
        tangency_perf.expected_return,
        marker="*",
        s=200,
        color="tab:red",
        zorder=4,
        label=f"maximum Sharpe ({tangency_perf.sharpe:.2f})",
    )
    ax.set(
        xlabel="annualized volatility",
        ylabel="annualized expected return",
        title="Efficient frontier (SYNTHETIC data: 5 GBM assets, 5 years, seed 42)",
        xlim=(0, None),
    )
    ax.xaxis.set_major_formatter("{x:.0%}")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110)
    print(f"saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

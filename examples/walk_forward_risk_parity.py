"""Walk-forward risk parity against equal weight on fictional prices.

Run ``uv run python examples/walk_forward_risk_parity.py``. Each monthly target
uses only closes preceding the rebalance date; the existing engine trades at
that date's close and holds the target during the following day. See
``docs/WALK_FORWARD.md`` for the timing, CSV contract and model limitations.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantkit import QuantkitError
from quantkit.backtest import backtest_weights, performance_summary, rebalance_days
from quantkit.data import simulate_gbm_prices
from quantkit.portfolio import compute_returns, ledoit_wolf_covariance, risk_parity


@dataclass(frozen=True)
class TargetSchedule:
    """Daily targets (NaN means no trade) and the estimation windows behind them."""

    risk_parity: pd.DataFrame
    equal_weight: pd.DataFrame
    windows: pd.DataFrame


def build_targets(prices: pd.DataFrame, lookback: int = 252) -> TargetSchedule:
    """Estimate monthly targets from exactly ``lookback`` returns ending at t-1.

    A return window needs one more closing price than returns. At rebalance
    position t the price window is [t-lookback-1, t), excluding close t. Both
    strategies remain in cash until the first monthly rebalance with that history.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2 returns")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices need a DatetimeIndex")
    # compute_returns validates the whole input. Its first row corresponds to
    # original price position 1, hence the additional -1 in the slice below.
    returns = compute_returns(prices)
    risk_targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    equal_targets = risk_targets.copy()
    windows: list[dict[str, object]] = []
    for offset in np.flatnonzero(rebalance_days(prices.index, "M")):
        position = int(offset)
        if position < lookback + 1:
            continue
        history = returns.iloc[position - lookback - 1 : position - 1]
        covariance = ledoit_wolf_covariance(history)
        risk_targets.iloc[position] = risk_parity(covariance).reindex(prices.columns)
        equal_targets.iloc[position] = 1.0 / len(prices.columns)
        windows.append(
            {
                "rebalance_date": prices.index[position],
                "history_start": prices.index[position - lookback - 1],
                "history_end": prices.index[position - 1],
                "n_returns": len(history),
            }
        )
    return TargetSchedule(
        risk_targets,
        equal_targets,
        pd.DataFrame(
            windows,
            columns=["rebalance_date", "history_start", "history_end", "n_returns"],
        ),
    )


def run_example(
    output_dir: Path,
    *,
    n_days: int = 1260,
    lookback: int = 252,
    cost_bps: float = 10.0,
    seed: int = 42,
) -> dict[str, Path]:
    """Simulate, estimate, backtest and export six inspectable CSV files."""
    prices = simulate_gbm_prices(
        tickers=["AAA", "BBB", "CCC"],
        n_days=n_days,
        drift=[0.04, 0.06, 0.08],
        vol=[0.10, 0.18, 0.28],
        correlation=0.3,
        seed=seed,
    )
    schedule = build_targets(prices, lookback)
    trade_positions = np.flatnonzero(schedule.risk_parity.notna().all(axis=1).to_numpy())
    if len(trade_positions) == 0 or int(trade_positions[0]) >= len(prices) - 1:
        raise ValueError("not enough history for a monthly rebalance and subsequent return")
    first_trade = int(trade_positions[0])
    targets = {
        "risk_parity": schedule.risk_parity,
        "equal_weight": schedule.equal_weight,
    }
    results = {
        name: backtest_weights(prices, weights, cost_bps=cost_bps)
        for name, weights in targets.items()
    }
    # Start both summaries at the first trade, including its entry cost. The
    # exported wealth still shows every day, including the shared cash warmup.
    summary = pd.DataFrame(
        {
            name: performance_summary(
                result.returns.iloc[first_trade:],
                turnover=result.turnover.iloc[first_trade:],
            )
            for name, result in results.items()
        }
    )
    summary.index.name = "metric"
    wealth = pd.DataFrame({name: result.equity for name, result in results.items()})
    tables = {
        "prices.csv": prices,
        "risk_parity_targets.csv": schedule.risk_parity,
        "equal_weight_targets.csv": schedule.equal_weight,
        "wealth.csv": wealth,
        "summary.csv": summary,
        "estimation_windows.csv": schedule.windows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(
            path,
            index=name != "estimation_windows.csv",
            date_format="%Y-%m-%d",
            float_format="%.12g",
            lineterminator="\n",
        )
        written[name] = path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fictional workflow; invalid inputs or I/O errors exit 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("walk-forward-output"))
    parser.add_argument("--days", type=int, default=1260)
    parser.add_argument("--lookback", type=int, default=252, help="number of prior daily returns")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        paths = run_example(
            args.output_dir,
            n_days=args.days,
            lookback=args.lookback,
            cost_bps=args.cost_bps,
            seed=args.seed,
        )
    except (QuantkitError, ValueError, OSError) as exc:
        print(f"walk-forward: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"SYNTHETIC data: {args.days} business-day closes, seed {args.seed}; "
        f"{args.lookback} prior returns, monthly rebalancing, costs {args.cost_bps:g} bps."
    )
    print("Both portfolios share the cash warmup and evaluation dates.")
    print("These exports demonstrate mechanics, not evidence of investment performance.")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

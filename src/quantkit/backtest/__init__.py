"""Daily vectorized backtesting: engine, example strategies and performance metrics.

See :mod:`quantkit.backtest.engine` for the exact timing (no-look-ahead) contract.
"""

from quantkit.backtest.engine import (
    BacktestResult,
    Strategy,
    backtest_weights,
    check_no_lookahead,
    run_backtest,
)
from quantkit.backtest.metrics import (
    annualized_turnover,
    annualized_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    hit_rate,
    max_drawdown,
    max_drawdown_duration,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from quantkit.backtest.strategies import (
    BuyAndHold,
    EqualWeight,
    Frequency,
    InverseVolatility,
    MovingAverageCrossover,
    TimeSeriesMomentum,
    rebalance_days,
)

__all__ = [
    "BacktestResult",
    "BuyAndHold",
    "EqualWeight",
    "Frequency",
    "InverseVolatility",
    "MovingAverageCrossover",
    "Strategy",
    "TimeSeriesMomentum",
    "annualized_turnover",
    "annualized_volatility",
    "backtest_weights",
    "cagr",
    "calmar_ratio",
    "check_no_lookahead",
    "drawdown_series",
    "hit_rate",
    "max_drawdown",
    "max_drawdown_duration",
    "performance_summary",
    "rebalance_days",
    "run_backtest",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
]

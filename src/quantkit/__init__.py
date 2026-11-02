"""quantkit: options pricing, portfolio optimization, backtesting and VaR validation.

Subpackages:
    quantkit.pricing: Black-Scholes-Merton, implied volatility, binomial trees and
        Monte Carlo for vanilla options.
    quantkit.portfolio: return and covariance estimation, mean-variance
        optimization, risk parity and risk metrics.
    quantkit.backtest: a daily vectorized backtester, example strategies and
        performance metrics.
    quantkit.var: rolling VaR and expected shortfall forecasts, exception
        backtests, the Basel traffic light and an expected shortfall backtest.
    quantkit.data: synthetic price generation and CSV loading.

This is educational and research software, not investment advice.
"""

from importlib.metadata import PackageNotFoundError, version

from quantkit.exceptions import (
    ArbitrageBoundError,
    ConvergenceError,
    DataValidationError,
    InvalidParameterError,
    QuantkitError,
)

try:
    __version__ = version("quantkit")
except PackageNotFoundError:  # pragma: no cover - only when run from a source tree
    __version__ = "0+unknown"

__all__ = [
    "ArbitrageBoundError",
    "ConvergenceError",
    "DataValidationError",
    "InvalidParameterError",
    "QuantkitError",
    "__version__",
]

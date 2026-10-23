"""Option pricing: Black-Scholes-Merton, implied volatility, binomial trees, Monte Carlo.

Every pricer takes an :class:`OptionSpec`, so the same contract can be priced by
several independent methods and the results compared.
"""

from quantkit.pricing.binomial import ExerciseStyle, binomial_price
from quantkit.pricing.black_scholes import Greeks, bsm_greeks, bsm_price
from quantkit.pricing.implied_vol import implied_vol, no_arbitrage_bounds
from quantkit.pricing.monte_carlo import MonteCarloResult, monte_carlo_price
from quantkit.pricing.option import OptionKind, OptionSpec

__all__ = [
    "ExerciseStyle",
    "Greeks",
    "MonteCarloResult",
    "OptionKind",
    "OptionSpec",
    "binomial_price",
    "bsm_greeks",
    "bsm_price",
    "implied_vol",
    "monte_carlo_price",
    "no_arbitrage_bounds",
]

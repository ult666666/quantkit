"""Exception hierarchy for quantkit.

Every error the library raises on purpose derives from :class:`QuantkitError`, so
callers can catch everything from this package with a single ``except`` clause.
Errors caused by bad inputs also derive from :class:`ValueError`, and solver
failures from :class:`RuntimeError`, matching standard-library conventions.
"""

from __future__ import annotations


class QuantkitError(Exception):
    """Base class for all errors raised deliberately by quantkit."""


class InvalidParameterError(QuantkitError, ValueError):
    """An input lies outside the domain on which a model is defined."""


class ArbitrageBoundError(InvalidParameterError):
    """An option price lies outside its model-free no-arbitrage bounds.

    No volatility can reproduce such a price, so implied volatility is undefined.

    Attributes:
        price: The offending option price.
        lower: The lower no-arbitrage bound (exclusive).
        upper: The upper no-arbitrage bound (exclusive).
    """

    def __init__(self, price: float, lower: float, upper: float) -> None:
        """Store the price and bounds and build a descriptive message."""
        self.price = price
        self.lower = lower
        self.upper = upper
        super().__init__(
            f"price {price:.6g} is outside the no-arbitrage bounds "
            f"({lower:.6g}, {upper:.6g}); no volatility reproduces it"
        )


class DataValidationError(InvalidParameterError):
    """Price data is malformed: missing values, duplicates, non-positive prices..."""


class ConvergenceError(QuantkitError, RuntimeError):
    """A numerical solver or optimizer did not reach a valid solution."""

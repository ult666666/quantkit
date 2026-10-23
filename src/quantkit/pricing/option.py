"""The option specification shared by every pricing method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantkit._validation import FloatArray, require_finite, require_positive
from quantkit.exceptions import InvalidParameterError

OptionKind = Literal["call", "put"]
"""Whether the option pays ``max(S - K, 0)`` (call) or ``max(K - S, 0)`` (put)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class OptionSpec:
    """A plain-vanilla option contract together with its market inputs.

    The same specification is used by every pricer; the exercise style (European
    or American) is an argument of the pricing call, not part of the data. Rates
    and yields are continuously compounded and annualized; time is in years.

    Fields are keyword-only so that ``spot`` and ``strike`` cannot be swapped by
    accident. The dataclass is frozen; use :func:`dataclasses.replace` to derive a
    bumped copy, e.g. ``replace(option, vol=0.25)``.

    Attributes:
        spot: Current price of the underlying, ``S > 0``.
        strike: Strike price, ``K > 0``.
        rate: Continuously compounded risk-free rate ``r`` (may be negative).
        vol: Annualized volatility of log returns, ``sigma > 0``.
        maturity: Time to expiry in years, ``T > 0``.
        dividend_yield: Continuous dividend (or carry) yield ``q``; default 0.
        kind: ``"call"`` or ``"put"``.

    Raises:
        InvalidParameterError: If any field is outside its domain.
    """

    spot: float
    strike: float
    rate: float
    vol: float
    maturity: float
    dividend_yield: float = 0.0
    kind: OptionKind

    def __post_init__(self) -> None:
        """Validate every field so that pricers can assume a well-posed problem."""
        validated = {
            "spot": require_positive("spot", self.spot),
            "strike": require_positive("strike", self.strike),
            "rate": require_finite("rate", self.rate),
            "vol": require_positive("vol", self.vol),
            "maturity": require_positive("maturity", self.maturity),
            "dividend_yield": require_finite("dividend_yield", self.dividend_yield),
        }
        if self.kind not in ("call", "put"):
            raise InvalidParameterError(f"kind must be 'call' or 'put', got {self.kind!r}")
        # Store plain floats (not ints or NumPy scalars) so reprs and arithmetic are
        # uniform; object.__setattr__ is the supported way to do this in a frozen
        # dataclass.
        for name, value in validated.items():
            object.__setattr__(self, name, value)

    def payoff(self, underlying: FloatArray) -> FloatArray:
        """Exercise value for an array of underlying prices.

        Args:
            underlying: Prices of the underlying at the exercise date.

        Returns:
            ``max(S - K, 0)`` for a call or ``max(K - S, 0)`` for a put, elementwise.
        """
        if self.kind == "call":
            return np.maximum(underlying - self.strike, 0.0)
        return np.maximum(self.strike - underlying, 0.0)
